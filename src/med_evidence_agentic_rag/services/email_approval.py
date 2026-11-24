"""Durable, owner-bound approvals. Each claim permits at most one SMTP attempt."""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from med_evidence_agentic_rag.models_and_database.postgres import email_lock_engine
from med_evidence_agentic_rag.persistence.email_actions import (
    EmailAction as EmailAction,
)
from med_evidence_agentic_rag.persistence.email_actions import execution_session, snapshot
from med_evidence_agentic_rag.persistence.postgres_tables import EmailActionRecord as Row
from med_evidence_agentic_rag.persistence.postgres_tables import utc_now
from med_evidence_agentic_rag.services.coordination import resources

Sender = Callable[[str, str, str, bool], bool]


OwnerScope = Callable[[UUID], AbstractAsyncContextManager[None]]
ConversationScope = Callable[[UUID, UUID], AbstractAsyncContextManager[None]]


def uncoordinated_owner(owner: UUID) -> AbstractAsyncContextManager[None]:
    return nullcontext()


class DeliveryRejected(ValueError):
    """A definitive refusal before SMTP acceptance."""


class ApprovalError(ValueError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmailApprovalRegistry:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession] | None = None,
        *,
        ttl_seconds: float = 300,
        clock: Callable[[], datetime] = utc_now,
        lock_sessions: async_sessionmaker[AsyncSession] | None = None,
        owner_scope: OwnerScope = uncoordinated_owner,
        proposal_scope: ConversationScope | None = None,
        delivery_scope: ConversationScope | None = None,
    ) -> None:
        self._sessions = sessions
        self.lock_sessions = lock_sessions
        self.owner_scope = owner_scope
        self.proposal_scope = proposal_scope
        self.delivery_scope = delivery_scope
        self.clock = clock
        self.ttl = timedelta(seconds=ttl_seconds)
        self._tasks: set[asyncio.Task[EmailAction]] = set()

    @property
    def active_deliveries(self) -> int:
        """Number of local confirmation workers, including detached deliveries."""
        return len(self._tasks)

    @property
    def sessions(self) -> async_sessionmaker[AsyncSession]:
        if self._sessions is None:
            from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal

            return AsyncSessionLocal
        return self._sessions

    async def propose(
        self,
        owner: UUID,
        conversation: UUID,
        recipient: str,
        subject: str,
        body: str,
        is_html: bool = False,
    ) -> EmailAction:
        if not recipient or not subject.strip() or not body.strip():
            raise ApprovalError("收件人、主题和正文不能为空", 400)
        if len(recipient) > 320 or len(subject) > 300 or len(body) > 100_000:
            raise ApprovalError("邮件内容过长", 400)
        scope = (
            self.proposal_scope(owner, conversation)
            if self.proposal_scope is not None
            else self.owner_scope(owner)
        )
        async with scope, self.sessions.begin() as session:
            if session.get_bind().dialect.name == "postgresql":
                await session.execute(text("SELECT pg_advisory_xact_lock(1719283401, 2)"))
            now = self.clock()
            active = or_(Row.deadline > now, Row.status == "executing")
            count = select(func.count()).select_from(Row).where(active)
            if (await session.scalar(count) or 0) >= 200 or (
                await session.scalar(count.where(Row.owner_id == owner)) or 0
            ) >= 10:
                raise ApprovalError("待处理邮件过多，请先确认或取消已有邮件", 429)
            row = Row(
                id=uuid4(),
                owner_id=owner,
                conversation_id=conversation,
                recipient=recipient,
                subject=subject,
                body=body,
                is_html=is_html,
                status="pending",
                deadline=now + self.ttl,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            return snapshot(row)

    async def _expire(self, session: AsyncSession, owner: UUID) -> None:
        now = self.clock()
        await session.execute(
            update(Row)
            .where(
                Row.owner_id == owner,
                Row.status == "pending",
                Row.deadline <= now,
            )
            .values(status="expired", updated_at=now)
        )

    async def list_pending(self, owner: UUID, conversation: UUID) -> list[EmailAction]:
        async with self.owner_scope(owner), self.sessions.begin() as session:
            await self._expire(session, owner)
            rows = await session.scalars(
                select(Row)
                .where(
                    Row.owner_id == owner,
                    Row.conversation_id == conversation,
                    Row.status.in_(("pending", "executing", "unknown")),
                )
                .order_by(Row.created_at, Row.id)
            )
            return [snapshot(row) for row in rows]

    async def _owned(self, session: AsyncSession, owner: UUID, action_id: UUID) -> EmailAction:
        row = await session.scalar(select(Row).where(Row.id == action_id, Row.owner_id == owner))
        if (
            row is None
            or row.status == "expired"
            or (row.status == "pending" and snapshot(row).deadline <= self.clock())
        ):
            raise ApprovalError("确认事项不存在或已过期", 404)
        return snapshot(row)

    async def reject(self, owner: UUID, action_id: UUID) -> EmailAction:
        async with self.owner_scope(owner), self.sessions() as session:
            await self._expire(session, owner)
            row = await session.scalar(
                update(Row)
                .where(
                    Row.owner_id == owner,
                    Row.id == action_id,
                    Row.status == "pending",
                    Row.deadline > self.clock(),
                )
                .values(status="rejected", updated_at=self.clock())
                .returning(Row)
                .execution_options(populate_existing=True)
            )
            result = snapshot(row) if row else None
            await session.commit()
            if result is None:
                await self._owned(session, owner, action_id)
                raise ApprovalError("此邮件已处理或正在执行")
            return result

    async def reconcile(self, owner: UUID, action_id: UUID, *, delivered: bool) -> EmailAction:
        async with self.owner_scope(owner), self.sessions.begin() as session:
            row = await session.scalar(
                update(Row)
                .where(Row.id == action_id, Row.owner_id == owner, Row.status == "unknown")
                .values(
                    status="sent" if delivered else "failed",
                    failure_code="owner_verified_sent" if delivered else "owner_verified_unsent",
                    updated_at=self.clock(),
                )
                .returning(Row)
            )
            if row is None:
                await self._owned(session, owner, action_id)
                raise ApprovalError("Only uncertain deliveries can be reconciled")
            return snapshot(row)

    async def confirm(self, owner: UUID, action_id: UUID, sender: Sender) -> EmailAction:
        task = asyncio.create_task(self._execute(owner, action_id, sender))
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task: asyncio.Task[EmailAction]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Observe detached failures, without logging private DB parameters.

    async def _execute(self, owner: UUID, action_id: UUID, sender: Sender) -> EmailAction:
        if self.delivery_scope is not None:
            async with self.sessions() as business:
                action = await self._owned(business, owner, action_id)
            if action.status == "rejected":
                raise ApprovalError("此邮件已取消")
            if action.status != "pending":
                return action
            scope = self.delivery_scope(owner, action.conversation_id)
        else:
            scope = self.owner_scope(owner)
        async with (
            scope,
            execution_session(self.lock_sessions or self.sessions, action_id) as session,
        ):
            if session is None:
                async with self.sessions() as business:
                    return await self._owned(business, owner, action_id)
            now, claim = self.clock(), uuid4()
            await self._expire(session, owner)
            row = await session.scalar(
                update(Row)
                .where(
                    Row.owner_id == owner,
                    Row.id == action_id,
                    Row.status == "pending",
                    Row.deadline > now,
                )
                .values(status="executing", claim_id=claim, updated_at=now)
                .returning(Row)
                .execution_options(populate_existing=True)
            )
            action = snapshot(row) if row else None
            await session.commit()
            if action is None:
                existing = await self._owned(session, owner, action_id)
                if existing.status == "rejected":
                    raise ApprovalError("此邮件已取消")
                return existing
            try:
                accepted = await asyncio.to_thread(
                    sender, action.recipient, action.subject, action.body, action.is_html
                )
                status = "sent" if accepted else "failed"
            except DeliveryRejected:
                status = "failed"
            except Exception:
                status = "unknown"
            code = {"failed": "delivery_rejected", "unknown": "delivery_uncertain"}.get(status)
            row = await session.scalar(
                update(Row)
                .where(
                    Row.id == action_id,
                    Row.claim_id == claim,
                    Row.status == "executing",
                )
                .values(status=status, failure_code=code, updated_at=self.clock())
                .returning(Row)
                .execution_options(populate_existing=True)
            )
            if row is None:
                raise RuntimeError("Email execution claim changed")
            result = snapshot(row)
            await session.commit()
            return result

    async def recover_abandoned(self, limit: int = 100) -> int:
        async with self.sessions() as session:
            ids = list(
                (
                    await session.execute(
                        select(Row.id, Row.owner_id)
                        .where(
                            Row.status == "executing",
                        )
                        .order_by(Row.updated_at, Row.id)
                        .limit(limit)
                    )
                ).all()
            )
        recovered = 0
        for action_id, owner in ids:
            try:
                async with (
                    self.owner_scope(owner),
                    execution_session(
                        self.lock_sessions or self.sessions, action_id, recovery=True
                    ) as session,
                ):
                    if session is not None:
                        changed = await session.scalar(
                            update(Row)
                            .where(
                                Row.id == action_id,
                                Row.status == "executing",
                            )
                            .values(
                                status="unknown",
                                failure_code="execution_interrupted",
                                updated_at=self.clock(),
                            )
                            .returning(Row.id)
                        )
                        await session.commit()
                        recovered += int(changed is not None)
            except HTTPException as error:
                if error.status_code != 409:
                    raise
        return recovered

    async def wait_pending(self) -> None:
        if not self._tasks:
            return
        drain = asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        cancellation: asyncio.CancelledError | None = None
        # An SMTP thread cannot be cancelled. Keep its task and scopes alive until
        # it commits the outcome, even when the shutdown caller cancels repeatedly.
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as error:
                cancellation = error
        if cancellation is not None:
            raise cancellation


email_approvals = EmailApprovalRegistry(
    lock_sessions=async_sessionmaker(email_lock_engine, expire_on_commit=False),
    owner_scope=resources.owner_scope,
    proposal_scope=resources.proposal_scope,
    delivery_scope=resources.delivery_scope,
)
