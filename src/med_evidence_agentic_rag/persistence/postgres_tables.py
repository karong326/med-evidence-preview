"""
1,数据表关系:
定义 PostgreSQL 业务 ORM 表及其关系。

``User`` 保存认证类型、长期偏好与联系人记忆；``DocumentRecord`` 登记用户私有资料；
``Conversation`` 与 ``ChatTurn`` 保存可展示的问答历史、证据快照、执行轨迹和失败信息。
外键级联覆盖用户删除和会话删除，字段默认值由持久化函数推进状态。
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """统一生成带时区的数据库时间。"""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """集中保存本项目业务表的 SQLAlchemy 元数据。"""


# 1,User：登录用户、偏好、称呼和联系人；
# 2,DocumentRecord：用户上传资料的处理状态
# 3,Conversation：一条独立会话；
# 4,ChatTurn：每轮问题、回答、状态、证据和轨迹；
#
# User
# ├── Conversation
# │   └── ChatTurn
# └── DocumentRecord

class User(Base):
    """邮箱或访问令牌登录的普通用户。"""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    auth_type: Mapped[str] = mapped_column(String(30), default="email", nullable=False)
    visitor_key_hash: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    last_login_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    preference: Mapped[str | None] = mapped_column(Text, nullable=True)
    long_term_memory: Mapped[dict[str, object] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class DocumentRecord(Base):
    """登记用户上传资料及其向量入库状态。"""

    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_owner_created", "owner_user_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="processing", nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Conversation(Base):
    """一个用户可以拥有多个相互隔离的聊天会话。"""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), default="新对话", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    turns: Mapped[list[ChatTurn]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class ChatTurn(Base):
    """保存一轮可直接展示的问题、回答及各自时间。"""

    __tablename__ = "chat_turns"
    __table_args__ = (
        Index("ix_chat_turns_conversation_question", "conversation_id", "question_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="processing", nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_snapshot: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    execution_trace: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )
    question_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    answer_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")
