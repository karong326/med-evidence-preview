"""
2. 会话与问答:
读写用户、会话和 ChatTurn 的直接 PostgreSQL 操作。

本模块负责邮箱/访问令牌用户的获取或创建、会话归属校验、历史列表，以及问答轮次从
``processing`` 到 ``completed`` 或 ``failed`` 的状态保存。它同时更新会话时间与标题，但
不运行 Agent；LangGraph 的内部状态由独立 Checkpointer 管理。
"""

import hashlib
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select

from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.persistence.postgres_tables import (
    ChatTurn,
    Conversation,
    User,
)


def _now() -> datetime:
    """生成带时区的当前时间。"""
    return datetime.now(UTC)


# 核心方法顺序:
# get_or_create_user()      → 登录时获取或创建用户
#
# create_conversation()     → 创建“新对话”
#
# create_chat_turn()        → 先保存问题，状态 processing
#
# complete_chat_turn()      → 保存回答、证据、轨迹        → 状态 completed
#
# mark_chat_turn_failed()   → 保存部分回答和安全失败原因   → 状态 failed

async def get_or_create_user(email: str) -> User:
    """按原始邮箱获取用户；首次登录创建，重复登录更新时间。"""
    async with AsyncSessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email)
            session.add(user)
        else:
            user.last_login_at = _now()
        await session.commit()
        await session.refresh(user)
        return user


async def get_or_create_access_user(visitor_id: str) -> User:
    """按浏览器访问标识获取访问令牌用户，避免每次登录产生新账号。"""
    visitor_key_hash = hashlib.sha256(visitor_id.encode("utf-8")).hexdigest()
    async with AsyncSessionLocal() as session:
        user = await session.scalar(select(User).where(User.visitor_key_hash == visitor_key_hash))
        if user is None:
            user = User(
                email=f"access-{visitor_key_hash[:24]}@local",
                auth_type="access_token",
                visitor_key_hash=visitor_key_hash,
            )
            session.add(user)
        else:
            user.last_login_at = _now()
        await session.commit()
        await session.refresh(user)
        return user


async def create_conversation(user_id: UUID) -> Conversation:
    """为指定用户创建一个尚未提问的新会话。"""
    async with AsyncSessionLocal() as session:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation


async def list_conversations(user_id: UUID) -> list[Conversation]:
    """按最近更新时间倒序列出当前用户的会话。"""
    async with AsyncSessionLocal() as session:
        result = await session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        return list(result)


async def get_owned_conversation(user_id: UUID, conversation_id: UUID) -> Conversation | None:
    """只返回确实属于当前用户的会话。"""
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )


async def delete_owned_conversation(user_id: UUID, conversation_id: UUID) -> bool:
    """删除当前用户的一条会话及其级联问答记录，不删除用户偏好。"""
    async with AsyncSessionLocal() as session:
        conversation = await session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        if conversation is None:
            return False
        await session.delete(conversation)
        await session.commit()
        return True


async def create_chat_turn(conversation_id: UUID, question: str) -> ChatTurn:
    """先保存用户问题，并在首次提问时生成简短会话标题。"""
    async with AsyncSessionLocal() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise LookupError("会话不存在")
        now = _now()
        if conversation.title == "新对话":
            conversation.title = question.strip()[:50]
        conversation.updated_at = now
        last_turn_number = await session.scalar(
            select(func.max(ChatTurn.turn_number)).where(
                ChatTurn.conversation_id == conversation_id
            )
        )
        turn = ChatTurn(
            conversation_id=conversation_id,
            question=question,
            turn_number=(last_turn_number or 0) + 1,
            question_at=now,
        )
        session.add(turn)
        await session.commit()
        await session.refresh(turn)
        return turn


async def complete_chat_turn(
    turn_id: UUID,
    answer: str,
    evidence_snapshot: list[dict[str, object]] | None = None,
    execution_trace: list[dict[str, object]] | None = None,
) -> ChatTurn:
    """流式回答正常完成后保存回答、证据快照和完成时间。"""
    async with AsyncSessionLocal() as session:
        turn = await session.get(ChatTurn, turn_id)
        if turn is None:
            raise LookupError("问答记录不存在")
        now = _now()
        turn.answer = answer
        turn.answer_at = now
        turn.status = "completed"
        turn.failure_code = None
        turn.failure_message = None
        turn.evidence_snapshot = evidence_snapshot
        turn.execution_trace = execution_trace
        conversation = await session.get(Conversation, turn.conversation_id)
        if conversation is not None:
            conversation.updated_at = now
        await session.commit()
        await session.refresh(turn)
        return turn


async def mark_chat_turn_failed(
    turn_id: UUID,
    *,
    failure_code: str,
    failure_message: str,
    answer: str | None = None,
    evidence_snapshot: list[dict[str, object]] | None = None,
    execution_trace: list[dict[str, object]] | None = None,
) -> ChatTurn:
    """保存安全的失败类别与部分结果，不保存外部服务原始异常。"""
    async with AsyncSessionLocal() as session:
        turn = await session.get(ChatTurn, turn_id)
        if turn is None:
            raise LookupError("问答记录不存在")
        now = _now()
        turn.answer = answer
        turn.answer_at = now
        turn.status = "failed"
        turn.failure_code = failure_code
        turn.failure_message = failure_message
        turn.evidence_snapshot = evidence_snapshot
        turn.execution_trace = execution_trace
        conversation = await session.get(Conversation, turn.conversation_id)
        if conversation is not None:
            conversation.updated_at = now
        await session.commit()
        await session.refresh(turn)
        return turn


# get_latest_failed_turn()          只在用户输入“重试”等指令时调用。
# list_recent_completed_turns()     只取最近三轮 completed，供跨轮汇总使用。
# list_chat_turns()                 会先验证会话属于当前用户，再返回完整可见历史

async def get_latest_failed_turn(conversation_id: UUID) -> ChatTurn | None:
    """获取本会话最近失败的一轮，供用户明确要求重试时恢复原问题。"""
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(ChatTurn)
            .where(
                ChatTurn.conversation_id == conversation_id,
                ChatTurn.status == "failed",
            )
            .order_by(ChatTurn.question_at.desc())
            .limit(1)
        )


async def list_recent_completed_turns(
    conversation_id: UUID,
    *,
    limit: int = 3,
) -> list[ChatTurn]:
    """按时间返回最近完成的问答，供历史证据和偏好任务使用。"""
    async with AsyncSessionLocal() as session:
        result = await session.scalars(
            select(ChatTurn)
            .where(
                ChatTurn.conversation_id == conversation_id,
                ChatTurn.status == "completed",
            )
            .order_by(ChatTurn.question_at.desc())
            .limit(limit)
        )
        return list(reversed(list(result)))


async def list_chat_turns(user_id: UUID, conversation_id: UUID) -> list[ChatTurn] | None:
    """验证会话归属后，按提问时间返回可读问答历史。"""
    async with AsyncSessionLocal() as session:
        owned = await session.scalar(
            select(Conversation.id).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        if owned is None:
            return None
        result = await session.scalars(
            select(ChatTurn)
            .where(ChatTurn.conversation_id == conversation_id)
            .order_by(ChatTurn.question_at.asc())
        )
        return list(result)
