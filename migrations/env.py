"""Business migrations run inside the transaction owned by the maintenance command."""

from alembic import context

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("Use python -m med_evidence_agentic_rag.persistence.migrations upgrade")
context.configure(connection=connection)
with context.begin_transaction():
    context.run_migrations()
