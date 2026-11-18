import asyncio
import threading

import pytest
from fastapi import HTTPException


def test_resource_conflicts_cleanup_and_owner_isolation():
    from med_evidence_agentic_rag.services.coordination import resources

    lease = resources.acquire(("document", "owner1", "a.txt"))
    with pytest.raises(HTTPException) as error:
        resources.acquire(("document", "owner1", "a.txt"))
    assert error.value.status_code == 409
    resources.acquire(("document", "owner2", "a.txt")).release()
    lease.release()
    lease.release()
    assert not resources.active


def test_cancelled_request_keeps_lease_until_worker_really_finishes():
    from med_evidence_agentic_rag.services.coordination import resources, run_protected

    async def run():
        started, finish = threading.Event(), threading.Event()
        key = ("document", "owner", "a.txt")

        def worker():
            started.set()
            finish.wait(5)

        task = asyncio.create_task(
            run_protected(resources.acquire(key), lambda: asyncio.to_thread(worker))
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(HTTPException):
            resources.acquire(key)
        finish.set()
        for _ in range(1000):
            if not resources.active:
                break
            await asyncio.sleep(0.001)
        assert not resources.active

    asyncio.run(run())


def test_unstarted_response_close_is_idempotent_and_persists_failure():
    from unittest.mock import AsyncMock

    from med_evidence_agentic_rag.services.coordination import LeasedStreamingResponse, resources

    async def run():
        async def stream():
            yield "test"

        abandoned = AsyncMock()
        response = LeasedStreamingResponse(
            stream(),
            resources.acquire(("conversation", "owner", "unstarted")),
            on_unstarted=abandoned,
        )
        await asyncio.gather(response.aclose(), response.aclose())
        abandoned.assert_awaited_once()
        assert not resources.active

    asyncio.run(run())


def test_response_keeps_lease_through_finalization_and_disconnect():
    from med_evidence_agentic_rag.services.coordination import LeasedStreamingResponse, resources

    async def run():
        finalizing, finish = asyncio.Event(), asyncio.Event()

        async def stream():
            try:
                yield "test"
            finally:
                finalizing.set()
                await finish.wait()

        response = LeasedStreamingResponse(
            stream(),
            resources.acquire(("conversation", "owner", "started")),
        )
        assert await anext(response.body_iterator) == "test"
        closing = asyncio.create_task(response.aclose())
        await finalizing.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert resources.active
        finish.set()
        await response.aclose()
        assert not resources.active

    asyncio.run(run())
