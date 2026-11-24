"""Pinned execution connection and immutable email snapshots."""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from med_evidence_agentic_rag.persistence.postgres_tables import EmailActionRecord


@dataclass(frozen=True)
class EmailAction:
    id: UUID
    owner_id: UUID
    conversation_id: UUID
    recipient: str
    subject: str
    body: str
    is_html: bool
    deadline: datetime
    status: str


def snapshot(row: EmailActionRecord) -> EmailAction:
    deadline = row.deadline
    if deadline.tzinfo is None:  # SQLite stores UTC without an offset.
        deadline = deadline.replace(tzinfo=UTC)
    return EmailAction(
        row.id,
        row.owner_id,
        row.conversation_id,
        row.recipient,
        row.subject,
        row.body,
        row.is_html,
        deadline,
        row.status,
    )


@asynccontextmanager
async def execution_session(
    sessions: async_sessionmaker[AsyncSession], action_id: UUID, *, recovery: bool = False
) -> AsyncGenerator[AsyncSession | None]:
    """Acquire the same live-owner lock before claiming or recovering an action."""
    async with sessions() as probe:
        bind = probe.bind
        if not isinstance(bind, AsyncEngine):
            raise RuntimeError("Approval sessions require an AsyncEngine")
        if bind.dialect.name != "postgresql":
            # Offline SQLite tests exercise SQL CAS. Only PostgreSQL proves owner death.
            yield None if recovery else probe
            return
    from med_evidence_agentic_rag.services.coordination import local_guards

    try:
        guard = await local_guards.obtain(("email-action", str(action_id)))
    except HTTPException as busy:
        if busy.status_code != 409:
            raise
        yield None
        return
    try:
        async with AsyncExitStack() as stack:
            try:
                connection = await stack.enter_async_context(bind.connect())
            except PoolTimeout as error:
                raise HTTPException(503, "Email execution capacity exhausted") from error
            # Keep this physical connection across the claim commit and SMTP call. A dead
            # process releases its lock when PostgreSQL closes the session.
            key = int.from_bytes(action_id.bytes[:8], "big", signed=True)
            try:
                acquired = bool(
                    await connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": key})
                )
                await connection.commit()
            except BaseException:
                await connection.invalidate()
                raise
            try:
                if acquired:
                    async with AsyncSession(connection, expire_on_commit=False) as session:
                        yield session
                else:
                    yield None
            finally:
                try:
                    if acquired:
                        released = await connection.scalar(
                            text("SELECT pg_advisory_unlock(:key)"), {"key": key}
                        )
                        if not released:
                            raise RuntimeError("Lost approval execution connection")
                        await connection.commit()
                except BaseException:
                    # Never return an ambiguously locked backend to the pool.
                    await connection.invalidate()
                    raise
    finally:
        await guard.close()
