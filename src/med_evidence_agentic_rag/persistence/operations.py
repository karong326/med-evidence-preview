"""Durable intents and scoped source reservations."""

from hashlib import sha256
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.persistence.postgres_tables import Operation, utc_now


class Journal:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def begin(
        self, kind: str, owner: UUID | None, resource: UUID | None, filename: str
    ) -> Operation:
        key = f"{owner}/{filename.casefold()}"
        if kind == "conversation-delete":
            key = f"{owner}/conversation/{resource}"
        row = Operation(
            kind=kind,
            owner_id=owner,
            resource_id=resource,
            filename=filename,
            reservation=sha256(key.encode()).hexdigest(),
        )
        async with self.sessions.begin() as session:
            previous = await session.scalar(
                select(Operation).where(Operation.reservation == row.reservation)
            )
            if previous is not None:
                if (
                    kind in ("delete", "base-delete", "conversation-delete", "owner-delete")
                    and previous.status == "pending"
                    and previous.kind == kind
                    and previous.resource_id == resource
                ):
                    return previous
                raise HTTPException(409, "Resource awaits reconciliation")
            if kind == "owner-delete" and await session.scalar(
                select(Operation.id)
                .where(Operation.owner_id == owner, Operation.reservation.is_not(None))
                .limit(1)
            ):
                raise HTTPException(409, "Account has unfinished operations")
            session.add(row)
            try:
                await session.flush()
            except IntegrityError as error:
                raise HTTPException(409, "Resource awaits reconciliation") from error
        return row

    async def set_phase(self, identity: UUID, phase: str) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(Operation)
                .where(Operation.id == identity)
                .values(phase=phase, updated_at=utc_now())
            )

    async def finish(self, identity: UUID) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(Operation)
                .where(Operation.id == identity)
                .values(
                    status="done",
                    phase="finished",
                    reservation=None,
                    error_code=None,
                    updated_at=utc_now(),
                )
            )

    async def pending(self, limit: int) -> list[Operation]:
        async with self.sessions() as session:
            return list(
                await session.scalars(
                    select(Operation)
                    .where(Operation.status == "pending")
                    .order_by(Operation.updated_at, Operation.id)
                    .limit(limit)
                )
            )

    async def get(self, identity: UUID) -> Operation | None:
        async with self.sessions() as session:
            return await session.get(Operation, identity)

    async def conversation_deleting(self, conversation: UUID) -> bool:
        async with self.sessions() as session:
            return (
                await session.scalar(
                    select(Operation.id)
                    .where(
                        Operation.resource_id == conversation,
                        Operation.kind == "conversation-delete",
                        Operation.reservation.is_not(None),
                    )
                    .limit(1)
                )
                is not None
            )

    async def owner_deleting(self, owner: UUID) -> bool:
        async with self.sessions() as session:
            return (
                await session.scalar(
                    select(Operation.id)
                    .where(
                        Operation.owner_id == owner,
                        Operation.kind == "owner-delete",
                        Operation.reservation.is_not(None),
                    )
                    .limit(1)
                )
                is not None
            )

    async def failed(self, row: Operation, maximum: int) -> None:
        attempts = row.attempts + 1
        async with self.sessions.begin() as session:
            await session.execute(
                update(Operation)
                .where(Operation.id == row.id)
                .values(
                    attempts=attempts,
                    status="blocked" if attempts >= maximum else "pending",
                    error_code="cleanup_failed",
                    updated_at=utc_now(),
                )
            )


journal = Journal(AsyncSessionLocal)
