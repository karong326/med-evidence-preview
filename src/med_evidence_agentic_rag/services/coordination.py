"""Owner-first PostgreSQL leases with a same-host guard for local index writes."""

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
    nullcontext,
)
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.types import Message, Receive, Scope, Send

from med_evidence_agentic_rag.models_and_database.postgres import lock_engine
from med_evidence_agentic_rag.persistence.operations import journal
from med_evidence_agentic_rag.utils.config import chroma_persist_directory

if TYPE_CHECKING:
    from med_evidence_agentic_rag.services.chat_stream import StreamDeadline

type ResourceKey = tuple[str, ...]


class ResourceLease:
    def __init__(
        self, stack: AsyncExitStack, registry: "ResourceRegistry", key: ResourceKey
    ) -> None:
        self.stack = stack
        self.registry = registry
        self.key = key
        self._closing = False
        self._borrowers = 0
        self._borrowers_released = asyncio.Event()
        self._borrowers_released.set()
        self._close_task: asyncio.Task[None] | None = None

    @contextmanager
    def borrow(self) -> Generator[None]:
        if self._closing:
            raise HTTPException(409, "Conversation is closing")
        self._borrowers += 1
        self._borrowers_released.clear()
        try:
            yield
        finally:
            self._borrowers -= 1
            if not self._borrowers:
                self._borrowers_released.set()

    def begin_close(self) -> None:
        self._closing = True

    async def _release(self) -> None:
        await self._borrowers_released.wait()
        await self.stack.aclose()

    async def close(self) -> None:
        self.begin_close()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._release())
        task = self._close_task
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = error
        task.result()
        if cancellation is not None:
            raise cancellation


_stream_lease: ContextVar[ResourceLease | None] = ContextVar("stream_lease", default=None)


class ResourceRegistry:
    def __init__(
        self,
        lock_root: Path,
        engine: AsyncEngine | None,
        owner_deleting: Callable[[UUID], Awaitable[bool]] | None = None,
        conversation_deleting: Callable[[UUID], Awaitable[bool]] | None = None,
    ) -> None:
        self.owner_deleting = owner_deleting
        self.conversation_deleting = conversation_deleting
        self.root = lock_root
        self.engine = engine
        self.active: set[ResourceLease] = set()
        self.tasks: set[asyncio.Task[object]] = set()

    @contextmanager
    def local_scope(self, scope: ResourceKey, *, exclusive: bool) -> Generator[int]:
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps(scope).encode()).digest()
        # The OS-backed lock survives a lost SQL connection while a worker thread runs.
        guard = sqlite3.connect(self.root / digest.hex(), timeout=0)
        try:
            try:
                guard.execute("BEGIN EXCLUSIVE" if exclusive else "BEGIN")
                guard.execute("SELECT count(*) FROM sqlite_master").fetchone()
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise
                raise HTTPException(409, "Resource busy") from error
            yield int.from_bytes(digest[:8], "big", signed=True)
        finally:
            guard.close()

    async def obtain(self, key: ResourceKey) -> ResourceLease:
        """Take the owner first, then the resource scopes in their canonical order."""
        kind, owner, *identity = key
        scopes: list[tuple[ResourceKey, bool]] = []
        has_owner = kind not in ("base-document", "index-owner", "email-action")
        if has_owner:
            scopes.append((("owner", owner), kind == "owner-delete"))
        if kind == "conversation-delete":
            scopes.append((("conversation", owner, *identity), True))
            scopes.append((("conversation-delivery", owner, *identity), True))
        elif kind not in ("owner-delete", "owner-read"):
            # Multiple deliveries can share the conversation while excluding deletion.
            scopes.append((key, kind != "conversation-delivery"))
        async with AsyncExitStack() as stack:
            connection = None
            if self.engine is not None:
                try:
                    connection = await stack.enter_async_context(self.engine.connect())
                except PoolTimeout as error:
                    raise HTTPException(503, "Coordination capacity exhausted") from error
            for scope, exclusive in scopes:
                advisory_key = stack.enter_context(self.local_scope(scope, exclusive=exclusive))
                if connection is not None:
                    fn = "pg_try_advisory_xact_lock" + ("" if exclusive else "_shared")
                    acquired = await connection.scalar(
                        text(f"SELECT {fn}(:key)"),
                        {"key": advisory_key},
                    )
                    if not acquired:
                        raise HTTPException(409, "Resource busy")
            if (
                self.owner_deleting is not None
                and has_owner
                and kind != "owner-delete"
                and await self.owner_deleting(UUID(owner))
            ):
                raise HTTPException(409, "Account deletion awaits reconciliation")
            if (
                kind == "conversation-delivery"
                and self.conversation_deleting is not None
                and await self.conversation_deleting(UUID(identity[0]))
            ):
                raise HTTPException(409, "Conversation deletion awaits reconciliation")
            lease = ResourceLease(stack.pop_all(), self, key)
            self.active.add(lease)
            lease.stack.callback(self.active.discard, lease)
            return lease

    @asynccontextmanager
    async def _scope(self, key: ResourceKey) -> AsyncGenerator[None]:
        lease = await self.obtain(key)
        try:
            yield
        finally:
            await lease.close()

    def owner_scope(self, owner: UUID) -> AbstractAsyncContextManager[None]:
        return self._scope(("owner-read", str(owner)))

    def delivery_scope(self, owner: UUID, conversation: UUID) -> AbstractAsyncContextManager[None]:
        return self._scope(("conversation-delivery", str(owner), str(conversation)))

    @asynccontextmanager
    async def proposal_scope(self, owner: UUID, conversation: UUID) -> AsyncGenerator[None]:
        lease = _stream_lease.get()
        if (
            lease is not None
            and lease.registry is self
            and lease.key == ("conversation", str(owner), str(conversation))
        ):
            with lease.borrow():
                yield
            return
        async with self.owner_scope(owner):
            yield

    async def wait_pending(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        while self.tasks:
            drain = asyncio.gather(*self.tasks, return_exceptions=True)
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError as error:
                    cancellation = error
        if cancellation is not None:
            raise cancellation


resources = ResourceRegistry(
    chroma_persist_directory / ".operation-locks",
    lock_engine,
    journal.owner_deleting,
    journal.conversation_deleting,
)


local_guards = ResourceRegistry(resources.root, None)


def start_protected[T](
    lease: ResourceLease, operation: Callable[[], Awaitable[T]]
) -> asyncio.Task[T]:
    async def run() -> T:
        try:
            return await operation()
        finally:
            await lease.close()

    task = asyncio.create_task(run())
    lease.registry.tasks.add(task)

    def done(completed: asyncio.Task[T]) -> None:
        lease.registry.tasks.discard(completed)
        if not completed.cancelled():
            completed.exception()  # Retrieve detached errors after a disconnected request.

    task.add_done_callback(done)
    return task


async def run_protected[T](lease: ResourceLease, operation: Callable[[], Awaitable[T]]) -> T:
    """Keep a lease until actual work completes, even if its HTTP request is cancelled."""
    return await asyncio.shield(start_protected(lease, operation))


async def run_stream_worker[T](operation: Callable[[], T]) -> T:
    """Keep the stream's resource lease while a synchronous tool thread is active."""
    lease = _stream_lease.get()
    with lease.borrow() if lease is not None else nullcontext():
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.exception()
            raise


class LeasedStreamingResponse(StreamingResponse):
    """Release only after stream persistence/closure, including early disconnects."""

    def __init__(
        self,
        stream: AsyncGenerator[str, None],
        lease: ResourceLease,
        *,
        on_unstarted: Callable[[], Awaitable[None]] | None = None,
        deadline: "StreamDeadline | None" = None,
        media_type: str = "application/x-ndjson",
    ) -> None:
        self.lease = lease
        self.stream = stream
        self.on_unstarted = on_unstarted
        self.deadline = deadline
        self.started = False
        self.close_task: asyncio.Task[None] | None = None
        super().__init__(self._iterate(), media_type=media_type)

    async def _iterate(self) -> AsyncGenerator[str, None]:
        self.started = True
        try:
            while True:
                token = _stream_lease.set(self.lease)
                try:
                    chunk = await anext(self.stream)
                except StopAsyncIteration:
                    break
                finally:
                    _stream_lease.reset(token)
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        async def close() -> None:
            await self.stream.aclose()
            if not self.started and self.on_unstarted is not None:
                await self.on_unstarted()

        if self.close_task is None:
            self.lease.begin_close()
            self.close_task = start_protected(self.lease, close)
        await asyncio.shield(self.close_task)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_before_deadline(message: Message) -> None:
            deadline = self.deadline
            if deadline is None:
                await send(message)
                return
            timeout = asyncio.timeout_at(deadline.expires_at)
            try:
                async with timeout:
                    await send(message)
            except TimeoutError:
                # The timer can fire within one clock tick of its scheduled time.
                if timeout.expired():
                    deadline.transport_expired = True
                raise

        try:
            await super().__call__(scope, receive, send_before_deadline)
        finally:
            await self.aclose()
