"""装配 FastAPI 应用、启动依赖并托管已构建前端。

启动生命周期初始化 PostgreSQL 业务表和 LangGraph Checkpointer，检查 Redis/Chroma/LLM，
并预热 BM25 与重排序模型；成功后将持久化 Agent 和就绪状态放入
``app.state``。本模块注册全部 API 路由，提供 ``/api/health`` 诊断，并将非 API 请求回退
到 ``frontend/dist`` 的 Vue 单页入口。
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from med_evidence_agentic_rag.agent.agent import create_chat_agent
from med_evidence_agentic_rag.api.admin_access_users import admin_access_users_router
from med_evidence_agentic_rag.api.admin_documents import admin_documents_router
from med_evidence_agentic_rag.api.admin_prompts import admin_prompts_router
from med_evidence_agentic_rag.api.auth import auth_router
from med_evidence_agentic_rag.api.chat import chat_router
from med_evidence_agentic_rag.api.conversations import conversation_router
from med_evidence_agentic_rag.api.documents import document_router
from med_evidence_agentic_rag.api.email_actions import email_actions_router
from med_evidence_agentic_rag.models_and_database.postgres import engine
from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import open_checkpointer
from med_evidence_agentic_rag.models_and_database.redis import redis_client
from med_evidence_agentic_rag.persistence.init_postgres import create_application_tables
from med_evidence_agentic_rag.utils.config import (
    PROJECT_ROOT,
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

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(endpoint, headers={"Authorization": f"Bearer {llm_api_key}"})
    response.raise_for_status()
    payload = response.json()

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
        raise RuntimeError(f"LLM_MODEL_NAME 不存在：{llm_model_name}")


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
    services: dict[str, str] = {
        "frontend_build": "available" if FRONTEND_INDEX.is_file() else "not_built",
        "postgresql": "starting",
        "checkpointer": "starting",
        "redis": "starting",
        "chroma": "starting",
        "llm": "starting",
        "retrieval_prewarm": "starting",
        "smtp": "configured"
        if os.getenv("MY_SENDER_EMAIL") and os.getenv("MY_SENDER_PASSWORD")
        else "not_configured",
        "mineru": "configured" if mineru_token else "not_configured",
    }
    app.state.readiness = {"status": "starting", "services": services}

    try:
        # 创建新表，并迁移旧字段和旧问答状态：
        await create_application_tables()
        services["postgresql"] = "ready"
        # Checkpoint 保存 Agent 内部上下文；chat_turns 保存前端可见历史。
        async with open_checkpointer() as checkpointer:
            services["checkpointer"] = "ready"
            await redis_client.ping()  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
            services["redis"] = "ready"
            from med_evidence_agentic_rag.models_and_database.chroma import vectorstore

            vectorstore.get(limit=1)
            services["chroma"] = "ready"
            await verify_llm_model()
            services["llm"] = "ready"
            services["retrieval_prewarm"] = f"ready ({prewarm_retrieval()} chunks)"

            # 把正式 Agent 和 Checkpointer 放入 FastAPI 应用状态，后续每个请求共享它们：
            app.state.chat_agent = create_chat_agent(checkpointer)
            # 持久化 Agent 的同时，注入Checkpointer
            app.state.checkpointer = checkpointer

            app.state.readiness = {"status": "ready", "services": services}

            logger.info("后端服务已就绪，信息如下：\n%s", services)
            # ========= yield 前是 FastAPI 的“应用生命周期管理器“ 的启动阶段===================
            yield

    except Exception as error:
        app.state.readiness = {
            "status": "failed",
            "services": services,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        logger.exception("后端启动检查失败")
        raise

    # ========= yield 后是 FastAPI 的“应用生命周期管理器” 的关闭阶段==================
    finally:
        if hasattr(app.state, "checkpointer"):
            del app.state.checkpointer
        await redis_client.aclose()
        await engine.dispose()
        logger.info("Redis 内存数据库即时服务已关闭")
        logger.info("PostgreSQL 会话上下文数据库服务已关闭")


# ========================================================================
app = FastAPI(title="MedEvidence Agentic RAG", lifespan=lifespan)
# ========================================================================
app.state.readiness = {"status": "starting", "services": {}}
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
def health() -> dict[str, Any]:
    """供开发启动脚本和人工验收读取的后端就绪状态。"""
    return app.state.readiness


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
