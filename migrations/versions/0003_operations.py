"""Recoverable cross-store operation intents."""

import sqlalchemy as sa
from alembic import op

revision = "0003_operations"
down_revision = "0002_email"


def upgrade() -> None:
    op.create_table(
        "operations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("owner_id", sa.Uuid()),
        sa.Column("resource_id", sa.Uuid()),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("reservation", sa.String(64), unique=True),
        sa.Column("phase", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(50)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_operations_pending", "operations", ["status", "updated_at"])


def downgrade() -> None:
    raise RuntimeError("Business data rollback requires a verified backup.")
