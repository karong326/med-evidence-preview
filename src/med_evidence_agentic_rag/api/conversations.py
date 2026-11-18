"""提供当前登录用户的会话与问答历史接口。

本路由创建、列出和删除 Conversation，并将 ChatTurn 转为稳定的前端响应模型。读取与删除
始终校验会话所有权；删除时还会通过应用状态中的 Checkpointer 清除不可见的 LangGraph
会话上下文，避免数据库历史已删但 Agent 仍记得旧会话。
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.persistence.conversations import (
    create_conversation,
    delete_owned_conversation,
    get_owned_conversation,
    list_chat_turns,
    list_conversations,
)
from med_evidence_agentic_rag.services.coordination import resources, run_protected
from med_evidence_agentic_rag.utils.logger import setup_logger

conversation_router = APIRouter(prefix="/api/conversations", tags=["会话"])
logger = setup_logger(__name__)


class ConversationResponse(BaseModel):
    """历史会话列表所需的最小字段。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatTurnResponse(BaseModel):
    """前端可直接展示的一轮问答及时间。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    turn_number: int
    question: str
    answer: str | None
    status: str = "completed"
    failure_code: str | None = None
    failure_message: str | None = None
    evidence_snapshot: list[dict[str, object]] | None = None
    execution_trace: list[dict[str, object]] | None = None
    question_at: datetime
    answer_at: datetime | None


@conversation_router.post("", response_model=ConversationResponse)
async def create_conversation_endpoint(
    current_user: AuthenticatedUser = Depends(require_user),
) -> ConversationResponse:
    """用户首次发送问题前创建一个新会话。"""
    conversation = await create_conversation(current_user.id)
    return ConversationResponse.model_validate(conversation)


@conversation_router.get("", response_model=list[ConversationResponse])
async def list_conversations_endpoint(
    current_user: AuthenticatedUser = Depends(require_user),
) -> list[ConversationResponse]:
    """按最近更新时间返回当前用户自己的历史会话。"""
    conversations = await list_conversations(current_user.id)
    return [ConversationResponse.model_validate(item) for item in conversations]


@conversation_router.get("/{conversation_id}/turns", response_model=list[ChatTurnResponse])
async def list_chat_turns_endpoint(
    conversation_id: UUID,
    current_user: AuthenticatedUser = Depends(require_user),
) -> list[ChatTurnResponse]:
    """只读取当前用户指定会话的可见问答记录。"""
    turns = await list_chat_turns(current_user.id, conversation_id)
    if turns is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return [ChatTurnResponse.model_validate(item) for item in turns]


@conversation_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(
    conversation_id: UUID,
    request: Request,
    current_user: AuthenticatedUser = Depends(require_user),
) -> None:
    """删除当前用户的一条会话、可见问答与 Agent Checkpointer 上下文。"""
    owned = await get_owned_conversation(current_user.id, conversation_id)
    if owned is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    lease = resources.acquire(("conversation", str(current_user.id), str(conversation_id)))
    return await run_protected(
        lease, lambda: _delete_conversation(conversation_id, request, current_user)
    )


async def _delete_conversation(
    conversation_id: UUID,
    request: Request,
    current_user: AuthenticatedUser,
) -> None:
    if await get_owned_conversation(current_user.id, conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="会话上下文服务未就绪，暂时无法删除会话")

    # conversation_id 同时是 LangGraph thread_id，先清理不可见的 Agent 上下文。
    await checkpointer.adelete_thread(str(conversation_id))
    deleted = await delete_owned_conversation(current_user.id, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    logger.info("历史会话已删除：用户=%s，会话=%s", current_user.id, conversation_id)
