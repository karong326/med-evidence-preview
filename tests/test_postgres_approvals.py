"""Run against a disposable PostgreSQL service selected explicitly by its test URL."""

import asyncio
import os
from contextlib import asynccontextmanager
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from med_evidence_agentic_rag.persistence.migrations import migrate
from med_evidence_agentic_rag.persistence.postgres_tables import (
    Base,
    Conversation,
    EmailActionRecord,
    User,
)
from med_evidence_agentic_rag.services.email_approval import EmailApprovalRegistry


@asynccontextmanager
async def test_schema():
    url = os.getenv("MEDEVIDENCE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("requires an explicit disposable PostgreSQL URL")
    schema = "approval_test_" + uuid4().hex
    admin = create_async_engine(url)
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


test_schema.__test__ = False


def test_checkpointer_probe_deadline_includes_saver_pipeline_lock():
    import sys

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from sqlalchemy.engine import make_url

    from med_evidence_agentic_rag.api.app import probe_checkpointer

    async def scenario():
        async with test_schema() as engine:
            async with engine.connect() as connection:
                schema = await connection.scalar(text("SELECT current_schema()"))
            url = make_url(os.environ["MEDEVIDENCE_TEST_POSTGRES_URL"])
            target = url.set(drivername="postgresql").update_query_dict(
                {"options": f"-csearch_path={schema}"}
            )
            async with AsyncPostgresSaver.from_conn_string(target.render_as_string(False)) as saver:
                await saver.setup()
                async with saver._cursor(pipeline=True) as cursor:
                    await cursor.execute("SELECT 1 AS value")
                    assert saver.lock.locked()
                    with pytest.raises(TimeoutError):
                        await probe_checkpointer(saver, timeout=0.05)
                    assert await cursor.fetchone() == {"value": 1}
                await probe_checkpointer(saver)
                await saver.adelete_thread(str(uuid4()))

    with asyncio.Runner(
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None
    ) as runner:
        runner.run(scenario())


def test_concurrent_contact_updates(monkeypatch):
    from med_evidence_agentic_rag.persistence import user_preferences as prefs

    async def scenario():
        async with test_schema() as engine:
            async with engine.begin() as conn:
                await conn.run_sync(migrate, "upgrade")
            factory = async_sessionmaker(engine, expire_on_commit=False)
            monkeypatch.setattr(prefs, "AsyncSessionLocal", factory)
            owner = uuid4()
            async with factory.begin() as session:
                session.add(User(id=owner, email="owner@example.org"))
            aliases = {f"alias{i}": f"test{i}@example.org" for i in range(12)}
            await asyncio.gather(*(prefs.save_user_contact(owner, name, address)
                                   for name, address in aliases.items()))
            _, memory = await prefs.get_user_personalization(owner)
            assert memory["contacts"] == aliases

    asyncio.run(scenario())


def test_postgres_fresh_claim_cancellation_and_orphan_recovery(monkeypatch):
    from med_evidence_agentic_rag.api import email_actions
    from med_evidence_agentic_rag.api.auth import AuthenticatedUser
    from med_evidence_agentic_rag.tools import send_email as mail

    async def scenario():
        async with test_schema() as engine:
            async with engine.begin() as conn:
                await conn.run_sync(migrate, "upgrade")
                await conn.run_sync(migrate, "check")
            factory = async_sessionmaker(engine, expire_on_commit=False)
            first, second = EmailApprovalRegistry(factory), EmailApprovalRegistry(factory)
            owner, chat = uuid4(), uuid4()
            async with factory.begin() as session:
                session.add(User(id=owner, email="owner@example.org"))
                await session.flush()
                session.add(Conversation(id=chat, user_id=owner))
            action = await first.propose(owner, chat, "draft@example.org", "draft", "private")
            entered, release = Event(), Event()
            calls = []

            def sender(*args):
                calls.append(args)
                entered.set()
                return release.wait(15)

            monkeypatch.setattr(mail, "send_email", sender)
            monkeypatch.setattr(email_actions, "email_approvals", first)
            request = asyncio.create_task(
                email_actions.confirm_email(
                    action.id, AuthenticatedUser(owner, "owner@example.org")
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert (await second.confirm(owner, action.id, sender)).status == "executing"
                assert await second.recover_abandoned() == 0
                drain = asyncio.create_task(first.wait_pending())
                await asyncio.sleep(0)
                for _ in range(2):
                    drain.cancel()
                    await asyncio.sleep(0)
                    assert await second.recover_abandoned() == 0
                    assert not drain.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await drain
            finally:
                release.set()
                await first.wait_pending()
            assert (await second.confirm(owner, action.id, sender)).status == "sent"
            assert len(calls) == 1
            orphan = await first.propose(owner, chat, "draft@example.org", "draft", "private")
            async with factory.begin() as session:
                await session.execute(
                    update(EmailActionRecord)
                    .where(EmailActionRecord.id == orphan.id)
                    .values(status="executing", claim_id=uuid4())
                )
            assert await second.recover_abandoned() == 1
            assert (await first.confirm(owner, orphan.id, sender)).status == "unknown"
            assert len(calls) == 1

    asyncio.run(scenario())


def test_postgres_validated_legacy_adoption():
    async def scenario():
        async with test_schema() as engine:
            async with engine.begin() as conn:
                # Build the pre-versioning ORM schema like the original startup command.
                legacy = [
                    Base.metadata.tables[t]
                    for t in ("users", "conversations", "documents", "chat_turns")
                ]
                await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=legacy))
                await conn.execute(text("CREATE TABLE checkpoint_writes (version integer)"))
                with pytest.raises(RuntimeError, match="adopt-legacy"):
                    await conn.run_sync(migrate, "upgrade")
                await conn.execute(text("ALTER TABLE users ALTER COLUMN email DROP NOT NULL"))
                with pytest.raises(RuntimeError, match="legacy"):
                    await conn.run_sync(migrate, "adopt-legacy")
                assert not await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
                await conn.execute(text("ALTER TABLE users ALTER COLUMN email SET NOT NULL"))
                await conn.execute(
                    text("ALTER TABLE conversations DROP CONSTRAINT conversations_user_id_fkey")
                )
                await conn.execute(
                    text("INSERT INTO conversations VALUES (:id, :owner, 'orphan', now(), now())"),
                    {"id": uuid4(), "owner": uuid4()},
                )
                await conn.execute(
                    text(
                        "ALTER TABLE conversations ADD CONSTRAINT conversations_user_id_fkey "
                        "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE NOT VALID"
                    )
                )
                with pytest.raises(RuntimeError, match="legacy"):
                    await conn.run_sync(migrate, "adopt-legacy")
                assert not await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
                await conn.execute(text("DELETE FROM conversations WHERE title = 'orphan'"))
                await conn.execute(
                    text("ALTER TABLE conversations VALIDATE CONSTRAINT conversations_user_id_fkey")
                )
                await conn.run_sync(migrate, "adopt-legacy")
                await conn.run_sync(migrate, "upgrade")
                await conn.run_sync(migrate, "check")
                assert await conn.run_sync(lambda c: inspect(c).has_table("checkpoint_writes"))

    asyncio.run(scenario())
