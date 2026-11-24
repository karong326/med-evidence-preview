"""提供单一的流式 Agent 问答接口 ``POST /api/ask``。

路由验证会话归属，读取最近完成轮次与用户个性化资料，构造 Agent 输入后先保存问题轮次。
随后选择应用生命周期创建的持久化 Agent，并委托服务层把运行事件编码为 NDJSON；回答、
证据、执行记录及失败状态均由该服务层统一保存。
"""

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from med_evidence_agentic_rag.agent.agent import agent as default_agent
from med_evidence_agentic_rag.agent.agent import stream_chat
from med_evidence_agentic_rag.agent.context import (
    build_agent_input,
    build_user_context_trace,
    is_retry_request,
)
from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.persistence.conversations import (
    complete_chat_turn,
    create_chat_turn,
    get_latest_failed_turn,
    get_owned_conversation,
    list_recent_completed_turns,
    mark_chat_turn_failed,
)
from med_evidence_agentic_rag.persistence.operations import journal
from med_evidence_agentic_rag.persistence.user_preferences import get_user_personalization
from med_evidence_agentic_rag.services.chat_stream import StreamDeadline, encode_chat_stream
from med_evidence_agentic_rag.services.coordination import (
    LeasedStreamingResponse,
    resources,
    start_protected,
)
from med_evidence_agentic_rag.utils.config import (
    agent_turn_deadline_seconds,
    max_question_characters,
)
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
chat_router = APIRouter()


class AskRequest(BaseModel):
    """一次问答所需的原始问题与所属会话。"""

    question: str = Field(
        min_length=1, max_length=max_question_characters, description="用户的医学问题"
    )
    conversation_id: UUID


@chat_router.post("/api/ask")
async def ask(
    request_body: AskRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(require_user),
) -> StreamingResponse:
    """校验会话归属、先保存问题，再启动可持久化的 Agent 事件流。"""
    # 1. 校验会话归属:
    conversation = await get_owned_conversation(current_user.id, request_body.conversation_id)

    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    lease = await resources.obtain(
        ("conversation", str(current_user.id), str(request_body.conversation_id))
    )
    lease_transferred = False
    try:
        if await journal.conversation_deleting(request_body.conversation_id):
            raise HTTPException(409, "Conversation deletion awaits reconciliation")
        if await get_owned_conversation(current_user.id, request_body.conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        # 2. 读取最近三轮正常问答:
        turns = await list_recent_completed_turns(request_body.conversation_id, limit=3)

        # 3. 只在用户要求重试时读取失败轮次:
        failed_turn = (
            await get_latest_failed_turn(request_body.conversation_id)
            if is_retry_request(request_body.question)
            else None
        )

        # 4. 每轮读取用户偏好和长期资料。
        preference, memory = await get_user_personalization(current_user.id)
        context_trace = build_user_context_trace(preference, memory)

        # 5. 构造真正交给 Agent 的输入 入口：
        agent_input = build_agent_input(
            request_body.question,
            turns,
            preference=preference,
            long_term_memory=memory,
            failed_turn=failed_turn,
        )

        # 6. 先保存问题，再运行 Agent：
        creation = asyncio.create_task(
            create_chat_turn(request_body.conversation_id, request_body.question)
        )
        try:
            turn = await asyncio.shield(creation)
        except asyncio.CancelledError:
            lease_transferred = True

            async def finalize_abandoned_creation() -> None:
                try:
                    abandoned = await creation
                    await mark_chat_turn_failed(
                        abandoned.id,
                        failure_code="client_disconnected",
                        failure_message="客户端连接已断开，请重试。",
                    )
                except Exception:
                    logger.exception("Abandoned turn finalization failed")

            cleanup = start_protected(lease, finalize_abandoned_creation)
            await asyncio.shield(cleanup)
            raise

        # 7. 选择正式 Agent：
        chat_agent: Any = getattr(request.app.state, "chat_agent", default_agent)
        logger.info("Accepted chat turn")
        deadline = StreamDeadline(asyncio.get_running_loop().time() + agent_turn_deadline_seconds)

        async def mark_unstarted() -> None:
            timed_out = deadline.transport_expired
            await mark_chat_turn_failed(
                turn.id,
                failure_code="service_timeout" if timed_out else "client_disconnected",
                failure_message=(
                    "服务响应超时，请稍后重试。" if timed_out else "客户端连接已断开，请重试。"
                ),
            )

        # 8. 返回流式响应：
        return LeasedStreamingResponse(
            encode_chat_stream(
                request_body.question,
                request_body.conversation_id,
                turn.id,
                chat_agent,
                agent_input,
                current_user.id,
                turn.turn_number,
                stream_chat=stream_chat,
                complete_chat_turn=complete_chat_turn,
                mark_chat_turn_failed=mark_chat_turn_failed,
                context_trace=context_trace,
                deadline=deadline,
            ),
            lease=lease,
            on_unstarted=mark_unstarted,
            deadline=deadline,
            media_type="application/x-ndjson",
        )
    except BaseException:
        if not lease_transferred:
            await lease.close()
        raise
