"""装配 FastAPI 应用、启动依赖并托管已构建前端。

启动生命周期检查 PostgreSQL 迁移和 LangGraph Checkpointer，检查 Redis/Chroma/LLM，
并预热 BM25 与重排序模型。注册健康检查、业务 API，并将前端请求回退
到 ``frontend/dist`` 的 Vue 单页入口。
"""

import asyncio
import json
import secrets
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from sqlalchemy import text

from med_evidence_agentic_rag.agent.agent import create_chat_agent
from med_evidence_agentic_rag.api.admin_access_users import admin_access_users_router
from med_evidence_agentic_rag.api.admin_documents import admin_documents_router
from med_evidence_agentic_rag.api.admin_prompts import admin_prompts_router
from med_evidence_agentic_rag.api.auth import auth_router
from med_evidence_agentic_rag.api.chat import chat_router
from med_evidence_agentic_rag.api.conversations import conversation_router
from med_evidence_agentic_rag.api.documents import document_router
from med_evidence_agentic_rag.api.email_actions import email_actions_router
from med_evidence_agentic_rag.models_and_database.postgres import (
    email_lock_engine,
    engine,
    lock_engine,
)
from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import open_checkpointer
from med_evidence_agentic_rag.models_and_database.redis import redis_client
from med_evidence_agentic_rag.persistence.init_postgres import create_application_tables
from med_evidence_agentic_rag.persistence.migrations import MigrationError
from med_evidence_agentic_rag.services.coordination import local_guards, resources
from med_evidence_agentic_rag.services.email_approval import email_approvals
from med_evidence_agentic_rag.services.health import Readiness
from med_evidence_agentic_rag.services.recovery import recover
from med_evidence_agentic_rag.services.telemetry import RequestTelemetry, telemetry
from med_evidence_agentic_rag.tools.send_email import smtp_configured
from med_evidence_agentic_rag.utils.config import (
    DATA_ROOT,
    PROJECT_ROOT,
    diagnostics_token,
    llm_api_key,
    llm_base_url,
    llm_model_name,
    mineru_token,
)
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
# 组装应用、完成启动检查、注册 API、托管 frontend/dist
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


async def verify_llm_model() -> None:
    """启动时确认远端 LLM 服务存在当前配置的模型，不发送对话内容。"""
    if not llm_base_url or not llm_api_key or not llm_model_name:
        raise RuntimeError("LLM_BASE_URL、LLM_API_KEY 或 LLM_MODEL_NAME 未配置")
    endpoint = f"{llm_base_url.rstrip('/')}/models"

    payload_bytes = bytearray()
    async with asyncio.timeout(2.5), httpx.AsyncClient(timeout=2) as client:
        async with client.stream(
            "GET", endpoint, headers={"Authorization": f"Bearer {llm_api_key}"}
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=16384):
                payload_bytes.extend(chunk)
                if len(payload_bytes) > 262144:
                    raise RuntimeError("LLM model inventory exceeds the probe response limit")
    payload: object = json.loads(payload_bytes)

    models = cast(dict[str, Any], payload).get("data") if isinstance(payload, dict) else None
    model_names: set[str | None] = (
        {
            cast(dict[str, str], item).get("id")
            for item in cast(list[object], models)
            if isinstance(item, dict) and isinstance(cast(dict[str, object], item).get("id"), str)
        }
        if isinstance(models, list)
        else set()
    )
    if llm_model_name not in model_names:
        raise RuntimeError("Configured LLM model is unavailable")


async def probe_postgresql() -> None:
    async with asyncio.timeout(2.5), engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def probe_checkpointer(checkpointer: AsyncPostgresSaver, *, timeout: float = 2.5) -> None:
    connection = checkpointer.conn
    if not isinstance(connection, AsyncConnection):
        raise RuntimeError("Checkpoint connection unavailable")
    # The saver holds this lock through psycopg pipeline entry and exit.
    async with asyncio.timeout(timeout), checkpointer.lock:
        await connection.execute("SELECT 1")


async def probe_redis() -> None:
    async with asyncio.timeout(2.5):
        await redis_client.ping()  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]


def prewarm_retrieval() -> int:
    """预建 BM25 索引并加载 Cross-Encoder，缩短用户首个检索请求。"""
    from med_evidence_agentic_rag.models_and_database.reranker import reranker
    from med_evidence_agentic_rag.rag_retrieval.bm25 import prewarm_bm25_index

    chunk_count = prewarm_bm25_index()
    _ = reranker
    return chunk_count


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    接下来启动业务表和持久化 Agent 同时注入Checkpointer，
    而后续停止时 yield 释放 Checkpointer 与连接池。
    """
    monitor = Readiness()
    app.state.readiness = monitor
    app.state.retrieval_prewarm = None
    index_owner = None
    try:
        index_owner = await local_guards.obtain(("index-owner", "maintenance"))
        # Verify business migrations before opening runtime services.
        await create_application_tables()
        telemetry.recovery["emails_unknown"] += await email_approvals.recover_abandoned()
        # Checkpoint 保存 Agent 内部上下文；chat_turns 保存前端可见历史。
        async with open_checkpointer() as checkpointer, AsyncExitStack() as drain:
            drain.push_async_callback(email_approvals.wait_pending)
            drain.push_async_callback(resources.wait_pending)
            await probe_redis()
            from med_evidence_agentic_rag.models_and_database.chroma import vectorstore

            vectorstore.get(limit=1)
            recovery = await recover(
                DATA_ROOT / "uploads",
                DATA_ROOT / "raw",
                checkpointer.adelete_thread,
            )
            for outcome in ("completed", "busy", "failed"):
                telemetry.recovery[outcome] += recovery[outcome]
            await verify_llm_model()
            app.state.retrieval_prewarm = prewarm_retrieval()

            # 把正式 Agent 和 Checkpointer 放入 FastAPI 应用状态，后续每个请求共享它们：
            app.state.chat_agent = create_chat_agent(checkpointer)
            # 持久化 Agent 的同时，注入Checkpointer
            app.state.checkpointer = checkpointer

            async def probe_chroma() -> None:
                await asyncio.to_thread(vectorstore.get, limit=1, include=[])

            monitor.start({
                "postgresql": probe_postgresql,
                "checkpointer": lambda: probe_checkpointer(checkpointer),
                "redis": probe_redis,
                "chroma": probe_chroma,
                "llm": verify_llm_model,
            })
            drain.push_async_callback(monitor.stop)
            logger.info("Application startup complete")
            yield

    except MigrationError:
        monitor.phase = "failed"
        logger.error("Business schema is out of date; run the migrations check command")
        raise
    except Exception:
        monitor.phase = "failed"
        logger.exception("后端启动检查失败")
        raise

    # ========= yield 后是 FastAPI 的“应用生命周期管理器” 的关闭阶段==================
    finally:
        if hasattr(app.state, "checkpointer"):
            del app.state.checkpointer
        try:
            await email_approvals.wait_pending()
        finally:
            try:
                await redis_client.aclose()
            finally:
                try:
                    await engine.dispose()
                    await lock_engine.dispose()
                    await email_lock_engine.dispose()
                finally:
                    if index_owner is not None:
                        await index_owner.close()
        logger.info("Application stopped")


# ========================================================================
app = FastAPI(title="MedEvidence Agentic RAG", lifespan=lifespan)
# ========================================================================
app.state.readiness = Readiness()
app.add_middleware(RequestTelemetry, telemetry=telemetry)
for router in (
    auth_router,
    chat_router,
    conversation_router,
    document_router,
    email_actions_router,
    admin_documents_router,
    admin_prompts_router,
    admin_access_users_router,
):
    app.include_router(router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    """供开发启动脚本和人工验收读取的后端就绪状态。"""
    snapshot = readiness_snapshot()
    services = {
        **cast(dict[str, str], snapshot["services"]),
        "frontend_build": "available" if FRONTEND_INDEX.is_file() else "not_built",
        "smtp": "configured" if smtp_configured() else "not_configured",
        "mineru": "configured" if mineru_token else "not_configured",
    }
    chunks = getattr(app.state, "retrieval_prewarm", None)
    if isinstance(chunks, int):
        services["retrieval_prewarm"] = f"ready ({chunks} chunks)"
    return {"status": snapshot["status"], "services": services}


def readiness_snapshot() -> dict[str, str | dict[str, str]]:
    return cast(Readiness, app.state.readiness).snapshot()


@app.get("/livez", include_in_schema=False)
async def livez() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> JSONResponse:
    snapshot = readiness_snapshot()
    return JSONResponse(snapshot, status_code=200 if snapshot["status"] == "ready" else 503)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    if not diagnostics_token:
        raise HTTPException(404)
    expected = b"Bearer " + diagnostics_token.encode()
    presented = request.headers.get("Authorization", "").encode()
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(401, headers={"WWW-Authenticate": "Bearer"})
    snapshot = readiness_snapshot()
    return Response(
        telemetry.render(
            cast(dict[str, str], snapshot["services"]),
            ready=snapshot["status"] == "ready",
            protected=len(resources.active),
            deliveries=email_approvals.active_deliveries,
        ),
        media_type="text/plain; version=0.0.4",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_page(frontend_path: str) -> FileResponse:
    """返回构建产物；Vue 客户端路由统一回退到 index.html。"""
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(
            status_code=503, detail="前端尚未构建成功，请先执行 npm run build 来构建前端"
        )
    requested = (FRONTEND_DIST / frontend_path).resolve()
    if frontend_path and requested.is_relative_to(FRONTEND_DIST.resolve()) and requested.is_file():
        return FileResponse(requested)
    return FileResponse(FRONTEND_INDEX, headers={"Cache-Control": "no-cache"})
