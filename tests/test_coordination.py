import asyncio
import threading

import pytest
from fastapi import HTTPException


def test_resource_conflicts_cleanup_and_owner_isolation():
    from med_evidence_agentic_rag.services.coordination import resources

    async def scenario():
        lease = await resources.obtain(("document", "owner1", "a.txt"))
        with pytest.raises(HTTPException) as error:
            await resources.obtain(("document", "owner1", "a.txt"))
        assert error.value.status_code == 409
        await (await resources.obtain(("document", "owner1", "b.txt"))).close()
        await (await resources.obtain(("document", "owner2", "a.txt"))).close()
        await lease.close()
        await lease.close()
        assert not resources.active

    asyncio.run(scenario())


def test_cancelled_request_keeps_lease_until_worker_really_finishes():
    from med_evidence_agentic_rag.services.coordination import resources, run_protected

    async def run():
        started, finish = threading.Event(), threading.Event()
        key = ("document", "owner", "a.txt")

        def worker():
            started.set()
            finish.wait(5)

        task = asyncio.create_task(
            run_protected(await resources.obtain(key), lambda: asyncio.to_thread(worker))
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(HTTPException):
            await resources.obtain(key)
        finish.set()
        for _ in range(1000):
            if not resources.active:
                break
            await asyncio.sleep(0.001)
        assert not resources.active

    asyncio.run(run())


def test_registry_drains_its_detached_workers_even_if_shutdown_is_cancelled(tmp_path):
    from med_evidence_agentic_rag.services.coordination import ResourceRegistry, run_protected

    async def run():
        registry = ResourceRegistry(tmp_path, None)
        owner, started, finish = "owner", asyncio.Event(), asyncio.Event()
        lease = await registry.obtain(("document", owner, "test.txt"))

        async def worker():
            started.set()
            await finish.wait()

        request = asyncio.create_task(run_protected(lease, worker))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        drain = asyncio.create_task(registry.wait_pending())
        try:
            for _ in range(2):
                await asyncio.sleep(0)
                assert not drain.done()
                drain.cancel()
            with pytest.raises(HTTPException):
                await registry.obtain(("owner-delete", owner))
        finally:
            finish.set()
            await registry.wait_pending()
        with pytest.raises(asyncio.CancelledError):
            await drain
        await (await registry.obtain(("owner-delete", owner))).close()

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
            await resources.obtain(("conversation", "owner", "unstarted")),
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
            await resources.obtain(("conversation", "owner", "started")),
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
