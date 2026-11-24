"""Independent PostgreSQL lock backends and isolated local fence directories."""

import asyncio
import os
import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from med_evidence_agentic_rag.services.coordination import ResourceRegistry, run_protected

CHILD = """
import asyncio, os, sys
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from med_evidence_agentic_rag.services.coordination import ResourceRegistry
async def run():
    engine = create_async_engine(os.environ["MEDEVIDENCE_TEST_POSTGRES_URL"])
    locks = ResourceRegistry(Path(sys.argv[1]), engine)
    lease = await locks.obtain(("conversation", sys.argv[2], "thread"))
    print("owned", flush=True)
    await asyncio.Event().wait()
asyncio.run(run())
"""


def test_process_kill_releases_postgres_and_host_locks(tmp_path):
    url = os.getenv("MEDEVIDENCE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("explicit disposable PostgreSQL URL required")

    async def scenario():
        owner = str(uuid4())
        engine = create_async_engine(url)
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, str(tmp_path), owner],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 15) == "owned\n"
            peer = ResourceRegistry(tmp_path / "peer", engine)
            with pytest.raises(HTTPException) as busy:
                await peer.obtain(("owner-delete", owner))
            assert busy.value.status_code == 409
        finally:
            child.kill()
            child.wait(timeout=10)
            child.stdout.close()
        same_host = ResourceRegistry(tmp_path, engine)
        for _ in range(50):
            try:
                lease = await same_host.obtain(("owner-delete", owner))
                await lease.close()
                break
            except HTTPException:
                await asyncio.sleep(0.05)
        else:
            pytest.fail("Dead process retained ownership")
        await engine.dispose()

    asyncio.run(scenario())


def test_lock_pool_pressure_and_backend_loss_while_worker_active(tmp_path):
    url = os.getenv("MEDEVIDENCE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("explicit disposable PostgreSQL URL required")

    async def scenario():
        name = "coordination_test_" + uuid4().hex
        owner = str(uuid4())
        business = create_async_engine(url, pool_size=1, max_overflow=0, pool_timeout=1)
        pool = create_async_engine(
            url,
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.1,
            connect_args={"server_settings": {"application_name": name}},
        )
        locks = ResourceRegistry(tmp_path, pool)
        lease = await locks.obtain(("document", owner, "a.txt"))
        from threading import Event

        started, gate = Event(), Event()

        def worker():
            started.set()
            assert gate.wait(15)

        request = asyncio.create_task(run_protected(lease, lambda: asyncio.to_thread(worker)))
        assert await asyncio.to_thread(started.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        try:
            with pytest.raises(HTTPException) as full:
                await locks.obtain(("document", owner, "b.txt"))
            assert full.value.status_code == 503
            async with business.begin() as connection:
                assert await connection.scalar(text("SELECT 1")) == 1
                assert await connection.scalar(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity WHERE application_name=:name"
                    ),
                    {"name": name},
                )
            peer = ResourceRegistry(tmp_path, business)
            with pytest.raises(HTTPException) as blocked:
                await peer.obtain(("owner-delete", owner))
            assert blocked.value.status_code == 409
        finally:
            gate.set()
            await locks.wait_pending()
            await pool.dispose()
        retry = await peer.obtain(("owner-delete", owner))
        await retry.close()
        await business.dispose()

    asyncio.run(scenario())


def test_approval_pool_saturation_preserves_business_transactions(tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from test_postgres_approvals import test_schema

    from med_evidence_agentic_rag.persistence.migrations import migrate
    from med_evidence_agentic_rag.persistence.postgres_tables import Conversation, User
    from med_evidence_agentic_rag.services.email_approval import EmailApprovalRegistry

    async def scenario():
        async with test_schema() as business:
            async with business.begin() as connection:
                await connection.run_sync(migrate, "upgrade")
                schema = await connection.scalar(text("SELECT current_schema()"))
            execution = create_async_engine(
                business.url,
                pool_size=1,
                max_overflow=0,
                pool_timeout=0.1,
                connect_args={"server_settings": {"search_path": schema}},
            )
            sessions = async_sessionmaker(business, expire_on_commit=False)
            locks = ResourceRegistry(tmp_path, business)
            registry = EmailApprovalRegistry(
                sessions,
                lock_sessions=async_sessionmaker(execution, expire_on_commit=False),
                owner_scope=locks.owner_scope,
            )
            owner, chat = uuid4(), uuid4()
            async with sessions.begin() as session:
                session.add(User(id=owner, email="test@example.org"))
                await session.flush()
                session.add(Conversation(id=chat, user_id=owner))
            first = await registry.propose(owner, chat, "test@example.org", "test", "one")
            second = await registry.propose(owner, chat, "test@example.org", "test", "two")
            from threading import Event

            started, finish = Event(), Event()
            sends = []

            def sender(recipient, subject, body, is_html):
                sends.append(body)
                started.set()
                return finish.wait(10)

            delivery = asyncio.create_task(registry.confirm(owner, first.id, sender))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                assert (await registry.confirm(owner, first.id, sender)).status == "executing"
                with pytest.raises(HTTPException) as full:
                    await registry.confirm(owner, second.id, sender)
                assert full.value.status_code == 503
                rejected = await registry.reject(owner, second.id)
                assert rejected.status == "rejected"
                with pytest.raises(HTTPException) as busy:
                    await locks.obtain(("owner-delete", str(owner)))
                assert busy.value.status_code == 409
            finally:
                finish.set()
                await registry.wait_pending()
                await execution.dispose()
            assert (await delivery).status == "sent"
            assert sends == ["one"]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["document", "conversation"])
def test_admin_route_waits_for_protected_owner_then_recovery_preserves_intent(
    tmp_path, kind, monkeypatch
):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    from sqlalchemy.ext.asyncio import async_sessionmaker
    from test_postgres_approvals import test_schema

    from med_evidence_agentic_rag.api import admin_access_users as routes
    from med_evidence_agentic_rag.persistence.migrations import migrate
    from med_evidence_agentic_rag.persistence.operations import Journal
    from med_evidence_agentic_rag.persistence.postgres_tables import Conversation, Operation, User
    from med_evidence_agentic_rag.services.recovery import recover

    async def scenario():
        async with test_schema() as engine:
            async with engine.begin() as connection:
                await connection.run_sync(migrate, "upgrade")
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            store = Journal(sessions)
            locks = ResourceRegistry(tmp_path / "locks", engine, store.owner_deleting)
            owner, chat = uuid4(), uuid4()
            async with sessions.begin() as session:
                session.add(User(id=owner, email="test@example.org", auth_type="access_token"))
                await session.flush()
                session.add(Conversation(id=chat, user_id=owner))
            monkeypatch.setattr(routes, "resources", locks)
            monkeypatch.setattr(routes, "AsyncSessionLocal", sessions)
            monkeypatch.setattr(routes, "journal", store)
            monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)

            class BrokenCheckpointer:
                async def adelete_thread(self, thread):
                    raise RuntimeError("interrupted")

            @asynccontextmanager
            async def checkpointer():
                yield BrokenCheckpointer()

            monkeypatch.setattr(routes, "open_checkpointer", checkpointer)
            holder = await locks.obtain((kind, str(owner), str(chat)))
            with pytest.raises(HTTPException) as busy:
                await routes.delete_access_user(owner)
            assert busy.value.status_code == 409
            await holder.close()
            with pytest.raises(RuntimeError, match="interrupted"):
                await routes.delete_access_user(owner)
            (pending,) = await store.pending(2)
            with pytest.raises(HTTPException):
                await locks.obtain(("document", str(owner), "new.txt"))
            cleanup = AsyncMock()
            for _ in range(2):
                await recover(tmp_path, tmp_path, cleanup, store=store, locks=locks)
            cleanup.assert_awaited_once_with(str(chat))
            async with sessions() as session:
                assert await session.get(User, owner) is None
                assert (await session.get(Operation, pending.id)).status == "done"

    asyncio.run(scenario())
