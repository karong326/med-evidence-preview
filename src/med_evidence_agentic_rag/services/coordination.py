"""Fail-fast resource leases for the single-worker preview application.

Acquisition contains no await: competing tasks in one event loop cannot interleave.
Tasks which outlive their request remain strongly referenced until they finish.
This is deliberately not a distributed lock.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

type ResourceKey = tuple[str, ...]


class ResourceLease:
    def __init__(self, registry: "ResourceRegistry", key: ResourceKey) -> None:
        self.registry = registry
        self.key = key
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.registry.active.remove(self.key)


class ResourceRegistry:
    def __init__(self) -> None:
        self.active: set[ResourceKey] = set()
        self.tasks: set[asyncio.Task[object]] = set()

    def acquire(self, key: ResourceKey) -> ResourceLease:
        if key in self.active:
            raise HTTPException(status_code=409, detail="该资源正在处理中，请稍后重试")
        self.active.add(key)
        return ResourceLease(self, key)


resources = ResourceRegistry()


def start_protected[T](
    lease: ResourceLease, operation: Callable[[], Awaitable[T]]
) -> asyncio.Task[T]:
    async def run() -> T:
        try:
            return await operation()
        finally:
            lease.release()

    task = asyncio.create_task(run())
    resources.tasks.add(task)

    def done(completed: asyncio.Task[T]) -> None:
        resources.tasks.discard(completed)
        if not completed.cancelled():
            completed.exception()  # Retrieve detached errors after a disconnected request.

    task.add_done_callback(done)
    return task


async def run_protected[T](lease: ResourceLease, operation: Callable[[], Awaitable[T]]) -> T:
    """Keep a lease until actual work completes, even if its HTTP request is cancelled."""
    return await asyncio.shield(start_protected(lease, operation))


class LeasedStreamingResponse(StreamingResponse):
    """Release only after stream persistence/closure, including early disconnects."""

    def __init__(
        self,
        stream: AsyncGenerator[str, None],
        lease: ResourceLease,
        *,
        on_unstarted: Callable[[], Awaitable[None]] | None = None,
        media_type: str = "application/x-ndjson",
    ) -> None:
        self.lease = lease
        self.stream = stream
        self.on_unstarted = on_unstarted
        self.started = False
        self.close_task: asyncio.Task[None] | None = None
        super().__init__(self._iterate(), media_type=media_type)

    async def _iterate(self) -> AsyncGenerator[str, None]:
        self.started = True
        try:
            async for chunk in self.stream:
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        async def close() -> None:
            await self.stream.aclose()
            if not self.started and self.on_unstarted is not None:
                await self.on_unstarted()

        if self.close_task is None:
            self.close_task = start_protected(self.lease, close)
        await asyncio.shield(self.close_task)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()
