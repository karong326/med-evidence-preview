"""Frozen business schema revision."""

import sqlalchemy as sa
from alembic import op

revision = "0002_email"
down_revision = "0001_business"

schema = sa.MetaData()
for parent in ("users", "conversations"):
    sa.Table(parent, schema, sa.Column("id", sa.Uuid(), primary_key=True))

sa.Table(
    "email_actions",
    schema,
    sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
    sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column(
        "conversation_id",
        sa.Uuid(),
        sa.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("recipient", sa.String(length=320), nullable=False),
    sa.Column("subject", sa.String(length=300), nullable=False),
    sa.Column("body", sa.Text(), nullable=False),
    sa.Column("is_html", sa.Boolean(), nullable=False),
    sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("status", sa.String(length=20), nullable=False),
    sa.Column("claim_id", sa.Uuid(), nullable=True),
    sa.Column("failure_code", sa.String(length=50), nullable=True),
    sa.Index("ix_email_actions_owner_conversation", "owner_id", "conversation_id"),
    sa.Index("ix_email_actions_status_updated", "status", "updated_at"),
)


def upgrade() -> None:
    schema.tables["email_actions"].create(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    raise RuntimeError("Business data rollback requires a verified backup.")
