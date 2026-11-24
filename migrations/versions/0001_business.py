"""Frozen business schema revision."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_business"
down_revision = None

schema = sa.MetaData()

sa.Table(
    "users",
    schema,
    sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
    sa.Column("email", sa.String(length=320), unique=True, nullable=False),
    sa.Column("auth_type", sa.String(length=30), nullable=False),
    sa.Column("visitor_key_hash", sa.String(length=64), unique=True, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("preference", sa.Text(), nullable=True),
    sa.Column("long_term_memory", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
)

sa.Table(
    "conversations",
    schema,
    sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
    sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("title", sa.String(length=200), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_conversations_user_updated", "user_id", "updated_at"),
)

sa.Table(
    "documents",
    schema,
    sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
    sa.Column(
        "owner_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("filename", sa.String(length=255), nullable=False),
    sa.Column("source_path", sa.Text(), unique=True, nullable=False),
    sa.Column("status", sa.String(length=20), nullable=False),
    sa.Column("chunk_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_documents_owner_created", "owner_user_id", "created_at"),
)

sa.Table(
    "chat_turns",
    schema,
    sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
    sa.Column(
        "conversation_id",
        sa.Uuid(),
        sa.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("question", sa.Text(), nullable=False),
    sa.Column("turn_number", sa.Integer(), nullable=False),
    sa.Column("answer", sa.Text(), nullable=True),
    sa.Column("status", sa.String(length=20), nullable=False),
    sa.Column("failure_code", sa.String(length=50), nullable=True),
    sa.Column("failure_message", sa.Text(), nullable=True),
    sa.Column("evidence_snapshot", sa.JSON(), nullable=True),
    sa.Column("execution_trace", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    sa.Column("question_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("answer_at", sa.DateTime(timezone=True), nullable=True),
    sa.Index("ix_chat_turns_conversation_question", "conversation_id", "question_at"),
)


def upgrade() -> None:
    schema.create_all(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    raise RuntimeError("Business data rollback requires a verified backup.")
