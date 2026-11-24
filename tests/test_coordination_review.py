"""Exercise reviewed crash and ownership boundaries with disposable state."""

import asyncio
import io
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import Request
from test_postgres_approvals import test_schema

from med_evidence_agentic_rag.api.auth import AuthenticatedUser
from med_evidence_agentic_rag.persistence.migrations import migrate
from med_evidence_agentic_rag.persistence.postgres_tables import Conversation, DocumentRecord, User
from med_evidence_agentic_rag.services.coordination import ResourceRegistry, resources


@asynccontextmanager
async def migrated_owner():
    async with test_schema() as engine:
        async with engine.begin() as connection:
            await connection.run_sync(migrate, "upgrade")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        owner = AuthenticatedUser(uuid4(), "owner@example.org")
        conversation = uuid4()
        async with sessions.begin() as session:
            session.add(User(id=owner.id, email=owner.email))
            await session.flush()
            session.add(Conversation(id=conversation, user_id=owner.id))
        yield engine, sessions, owner, conversation


@pytest.mark.parametrize("filename", ["a.txt", "A.txt"])
def test_registered_failed_upload_rejects_retry_before_writing(tmp_path, monkeypatch, filename):
    from med_evidence_agentic_rag.api import documents as routes
    from med_evidence_agentic_rag.persistence import documents as persistence
    from med_evidence_agentic_rag.services import document_deletion

    async def run():
        async with migrated_owner() as (_, sessions, owner, _):
            monkeypatch.setattr(persistence, "AsyncSessionLocal", sessions)
            monkeypatch.setattr(routes.journal, "sessions", sessions)
            monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)
            monkeypatch.setattr(routes, "index_document", lambda *a, **kw: 1)
            monkeypatch.setattr(document_deletion, "delete_indexed_document", lambda _: None)
            source = tmp_path / str(owner.id) / "a.txt"
            registered = await persistence.create_document(
                owner_user_id=owner.id, filename=source.name, source_path=str(source)
            )
            await persistence.mark_document_failed(registered.id)
            with pytest.raises(HTTPException) as conflict:
                await routes.upload_document(
                    UploadFile(io.BytesIO(b"retry"), filename=filename), owner
                )
            assert conflict.value.status_code == 409
            assert not source.parent.exists()
            assert await routes.journal.pending(10) == []
            assert await routes.delete_document(registered.id, owner) == {"deleted": True}
            result = await routes.upload_document(
                UploadFile(io.BytesIO(b"retry"), filename=filename), owner
            )
            assert result["status"] == "ready"

    asyncio.run(run())


@pytest.mark.parametrize("action", ["delete", "recover"])
def test_old_case_alias_cleanup_preserves_ready_source(tmp_path, monkeypatch, action):
    from med_evidence_agentic_rag.api import documents as routes
    from med_evidence_agentic_rag.persistence import documents as persistence
    from med_evidence_agentic_rag.services import document_deletion
    from med_evidence_agentic_rag.services.recovery import recover

    async def run():
        async with migrated_owner() as (_, sessions, owner, _):
            monkeypatch.setattr(routes.journal, "sessions", sessions)
            monkeypatch.setattr(persistence, "AsyncSessionLocal", sessions)
            monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)
            folder = tmp_path / str(owner.id)
            folder.mkdir()
            source = folder / "A.txt"
            source.write_bytes(b"ready")
            if not (folder / "a.txt").exists():
                pytest.skip("requires a case-insensitive filesystem")
            old, current = uuid4(), uuid4()
            async with sessions.begin() as session:
                session.add_all(
                    [
                        DocumentRecord(
                            id=old,
                            owner_user_id=owner.id,
                            filename="a.txt",
                            source_path=str(folder / "a.txt"),
                            status="failed",
                        ),
                        DocumentRecord(
                            id=current,
                            owner_user_id=owner.id,
                            filename="A.txt",
                            source_path=str(source),
                            status="ready",
                        ),
                    ]
                )
            vectors = {str(old), str(current)}
            monkeypatch.setattr(document_deletion, "delete_indexed_document", vectors.discard)
            if action == "delete":
                assert await routes.delete_document(old, owner) == {"deleted": True}
            else:
                await recover(
                    tmp_path,
                    tmp_path,
                    AsyncMock(),
                    store=routes.journal,
                    remove=lambda selector: vectors.discard(selector[1]),
                )
                await recover(
                    tmp_path,
                    tmp_path,
                    AsyncMock(),
                    store=routes.journal,
                    remove=lambda selector: vectors.discard(selector[1]),
                )
            assert source.read_bytes() == b"ready"
            assert vectors == {str(current)}
            assert (await persistence.get_owned_document(owner.id, current)).status == "ready"

    asyncio.run(run())


@pytest.mark.parametrize("action", ["delete", "recover"])
def test_cleanup_removes_only_selected_case_sensitive_legacy_source(
    tmp_path, monkeypatch, email_sessions, isolated_journal, action
):
    from med_evidence_agentic_rag.api import documents as routes
    from med_evidence_agentic_rag.persistence import documents as persistence
    from med_evidence_agentic_rag.services import document_deletion
    from med_evidence_agentic_rag.services.recovery import recover

    owner = AuthenticatedUser(uuid4(), "owner@example.org")
    folder = tmp_path / str(owner.id)
    folder.mkdir()
    old_path, current_path = folder / "a.txt", folder / "A.txt"
    current_path.write_bytes(b"keep")
    if old_path.exists():
        pytest.skip("requires a case-sensitive filesystem")
    old_path.write_bytes(b"remove")
    monkeypatch.setattr(persistence, "AsyncSessionLocal", email_sessions)
    monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)

    async def run():
        old, current = uuid4(), uuid4()
        async with email_sessions.begin() as session:
            session.add(User(id=owner.id, email=owner.email))
            session.add_all(
                [
                    DocumentRecord(
                        id=identity,
                        owner_user_id=owner.id,
                        filename=source.name,
                        source_path=str(source),
                        status=status,
                    )
                    for identity, source, status in (
                        (old, old_path, "delete_failed"),
                        (current, current_path, "ready"),
                    )
                ]
            )
        vectors = {str(old), str(current)}
        monkeypatch.setattr(document_deletion, "delete_indexed_document", vectors.discard)
        if action == "delete":
            assert await routes.delete_document(old, owner) == {"deleted": True}
        else:
            await recover(
                tmp_path,
                tmp_path,
                AsyncMock(),
                store=routes.journal,
                remove=lambda selector: vectors.discard(selector[1]),
            )
        assert await persistence.get_owned_document(owner.id, old) is None
        assert not old_path.exists()
        assert current_path.read_bytes() == b"keep"
        assert vectors == {str(current)}
        assert (await persistence.get_owned_document(owner.id, current)).status == "ready"

    asyncio.run(run())


def test_active_delivery_blocks_conversation_delete_and_duplicate_is_read_only(monkeypatch):
    from med_evidence_agentic_rag.api import conversations as routes
    from med_evidence_agentic_rag.persistence import conversations as persistence
    from med_evidence_agentic_rag.services.email_approval import email_approvals

    async def run():
        async with migrated_owner() as (engine, sessions, owner, conversation):
            monkeypatch.setattr(persistence, "AsyncSessionLocal", sessions)
            monkeypatch.setattr(routes.journal, "sessions", sessions)
            monkeypatch.setattr(resources, "engine", engine)
            monkeypatch.setattr(email_approvals, "_sessions", sessions)
            monkeypatch.setattr(email_approvals, "lock_sessions", sessions)
            action = await email_approvals.propose(
                owner.id, conversation, "recipient@example.org", "Subject", "Synthetic"
            )
            started, finish = Event(), Event()
            sent = []

            def sender(*args):
                sent.append(args)
                started.set()
                return finish.wait(15)

            checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
            req = Request(
                {
                    "type": "http",
                    "app": SimpleNamespace(state=SimpleNamespace(checkpointer=checkpointer)),
                }
            )
            execution = asyncio.create_task(email_approvals.confirm(owner.id, action.id, sender))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                assert (
                    await email_approvals.confirm(owner.id, action.id, sender)
                ).status == "executing"
                with pytest.raises(HTTPException) as busy:
                    await routes.delete_conversation_endpoint(conversation, req, owner)
                assert busy.value.status_code == 409
                checkpointer.adelete_thread.assert_not_awaited()
            finally:
                finish.set()
                await asyncio.gather(execution, return_exceptions=True)
            assert execution.result().status == "sent"
            assert len(sent) == 1
            await routes.delete_conversation_endpoint(conversation, req, owner)
            checkpointer.adelete_thread.assert_awaited_once_with(str(conversation))

    asyncio.run(run())


def test_chat_email_tool_reuses_owner_at_full_coordination_pool(monkeypatch):
    from med_evidence_agentic_rag.services.coordination import LeasedStreamingResponse
    from med_evidence_agentic_rag.services.email_approval import email_approvals
    from med_evidence_agentic_rag.tools import send_email as mail

    async def run():
        async with migrated_owner() as (engine, sessions, owner, conversation):
            lock_pool = create_async_engine(
                engine.url, pool_size=1, max_overflow=0, pool_timeout=0.1
            )
            monkeypatch.setattr(resources, "engine", lock_pool)
            monkeypatch.setattr(email_approvals, "_sessions", sessions)
            monkeypatch.setattr(
                mail, "resolve_email_recipient", AsyncMock(return_value="test@example.org")
            )
            lease = await resources.obtain(("conversation", str(owner.id), str(conversation)))
            context = SimpleNamespace(
                context={"user_id": str(owner.id), "conversation_id": str(conversation)}
            )

            async def stream():
                yield await mail.send_email_tool.coroutine(
                    "recipient", "Subject", "Synthetic", context
                )

            response = LeasedStreamingResponse(stream(), lease)
            try:
                with pytest.raises(HTTPException) as full:
                    await email_approvals.propose(
                        owner.id, conversation, "test@example.org", "outside", "Synthetic"
                    )
                assert full.value.status_code == 503
                assert '"requires_confirmation": true' in await anext(response.body_iterator)
            finally:
                await response.aclose()
                await lock_pool.dispose()
            assert (await email_approvals.list_pending(owner.id, conversation))[
                0
            ].subject == "Subject"

    asyncio.run(run())


def test_stream_close_keeps_fences_until_borrowed_proposal_commits(email_sessions, tmp_path):
    from med_evidence_agentic_rag.services.coordination import LeasedStreamingResponse
    from med_evidence_agentic_rag.services.email_approval import EmailApprovalRegistry

    async def run():
        owner, conversation = uuid4(), uuid4()
        async with email_sessions.begin() as session:
            session.add(User(id=owner, email="owner@example.org"))
            session.add(Conversation(id=conversation, user_id=owner))
        registry = ResourceRegistry(tmp_path / "locks", None)
        peer = ResourceRegistry(registry.root, None)
        borrowed, finish_proposal = asyncio.Event(), asyncio.Event()
        closing_stream, finish_stream = asyncio.Event(), asyncio.Event()
        start_late = asyncio.Event()

        @asynccontextmanager
        async def proposal_scope(owner, conversation):
            async with registry.proposal_scope(owner, conversation):
                if asyncio.current_task() is proposal:
                    borrowed.set()
                    await finish_proposal.wait()
                yield

        approvals = EmailApprovalRegistry(
            email_sessions, owner_scope=registry.owner_scope, proposal_scope=proposal_scope
        )

        async def propose(subject):
            return await approvals.propose(owner, conversation, "test@example.org", subject, "demo")

        async def propose_late():
            await start_late.wait()
            return await propose("late")

        async def stream():
            nonlocal proposal, late
            proposal = asyncio.create_task(propose("borrowed"))
            late = asyncio.create_task(propose_late())
            try:
                await asyncio.wait_for(borrowed.wait(), 5)
                yield "started"
            finally:
                closing_stream.set()
                await finish_stream.wait()

        proposal = late = None
        lease = await registry.obtain(("conversation", str(owner), str(conversation)))
        response = LeasedStreamingResponse(stream(), lease)
        closing = None
        try:
            assert await anext(response.body_iterator) == "started"
            closing = asyncio.create_task(response.aclose())
            await asyncio.wait_for(closing_stream.wait(), 5)
            start_late.set()
            (late_outcome,) = await asyncio.wait_for(
                asyncio.gather(late, return_exceptions=True), 5
            )
            finish_stream.set()
            await asyncio.sleep(0)
            for key in (
                ("owner-delete", str(owner)),
                ("conversation-delete", str(owner), str(conversation)),
            ):
                with pytest.raises(HTTPException) as busy:
                    await (await peer.obtain(key)).close()
                assert busy.value.status_code == 409
            assert not closing.done()
            assert isinstance(late_outcome, HTTPException) and late_outcome.status_code == 409
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            drain = asyncio.create_task(registry.wait_pending())
            for _ in range(2):
                await asyncio.sleep(0)
                assert not drain.done()
                drain.cancel()
            finish_proposal.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(drain, 5)
            assert [item.subject for item in await approvals.list_pending(owner, conversation)] == [
                "borrowed"
            ]
            await (await peer.obtain(("owner-delete", str(owner)))).close()
        finally:
            finish_stream.set()
            finish_proposal.set()
            if late is not None and not late.done():
                late.cancel()
            await asyncio.gather(
                *(task for task in (proposal, late, closing) if task), return_exceptions=True
            )
            await response.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("entrypoint", ["startup", "cli"])
def test_recovery_cancellation_drains_before_closing_fences(
    tmp_path, monkeypatch, isolated_journal, entrypoint
):
    from med_evidence_agentic_rag.models_and_database import postgres_checkpointer
    from med_evidence_agentic_rag.persistence import init_postgres, operations
    from med_evidence_agentic_rag.services import recovery, recovery_cli
    from med_evidence_agentic_rag.services.coordination import local_guards
    from med_evidence_agentic_rag.services.email_approval import email_approvals
    from med_evidence_agentic_rag.utils import config

    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(local_guards, "root", resources.root)
    uploads, base = tmp_path / "data" / "uploads", tmp_path / "data" / "raw"
    uploads.mkdir(parents=True)
    base.mkdir()
    source = base / "base.txt"
    source.write_bytes(b"synthetic")
    entered, release = Event(), Event()
    checkpoint_closed = asyncio.Event()

    def remove(selector):
        assert selector == ("source", str(source))
        entered.set()
        assert release.wait(15)

    real_recover = recovery.recover

    async def recover(uploads, base, delete_thread, **kwargs):
        return await real_recover(uploads, base, delete_thread, remove=remove, **kwargs)

    @asynccontextmanager
    async def open_checkpointer():
        try:
            yield SimpleNamespace(adelete_thread=AsyncMock())
        finally:
            checkpoint_closed.set()

    monkeypatch.setattr(recovery, "recover", recover)
    monkeypatch.setattr(postgres_checkpointer, "open_checkpointer", open_checkpointer)
    monkeypatch.setattr(init_postgres, "create_application_tables", AsyncMock())
    monkeypatch.setattr(email_approvals, "recover_abandoned", AsyncMock(return_value=0))
    if entrypoint == "startup":
        chroma = ModuleType("synthetic_chroma")
        chroma.vectorstore = SimpleNamespace(get=lambda **_: None)
        monkeypatch.setitem(
            sys.modules, "med_evidence_agentic_rag.models_and_database.chroma", chroma
        )
        from med_evidence_agentic_rag.api import app

        monkeypatch.setattr(app, "DATA_ROOT", tmp_path / "data")
        monkeypatch.setattr(app, "recover", recover)
        monkeypatch.setattr(app, "open_checkpointer", open_checkpointer)
        monkeypatch.setattr(app, "create_application_tables", AsyncMock())
        monkeypatch.setattr(
            app, "redis_client", SimpleNamespace(ping=AsyncMock(), aclose=AsyncMock())
        )

        async def start():
            async with app.lifespan(app.app):
                pytest.fail("Startup unexpectedly finished")

        start = start()
    else:
        monkeypatch.setattr(
            sys, "argv", ["recovery", "--apply", "--uploads", str(uploads), "--base", str(base)]
        )
        pending = []
        with monkeypatch.context() as patch:
            patch.setattr(recovery_cli, "run_recovery", pending.append)
            from med_evidence_agentic_rag.utils import logger

            patch.setattr(logger, "configure_server_logging", lambda: None)
            recovery_cli.main()
        (start,) = pending

    async def run():
        await operations.journal.begin("base-upload", None, None, source.name)
        run = asyncio.create_task(start)
        peer = ResourceRegistry(resources.root, None)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            run.cancel()
            for _ in range(2):
                await asyncio.sleep(0)
                assert not run.done()
                assert not checkpoint_closed.is_set()
                for key in (("index-owner", "maintenance"), ("base-document", source.name)):
                    with pytest.raises(HTTPException) as busy:
                        await peer.obtain(key)
                    assert busy.value.status_code == 409
                run.cancel()
        finally:
            release.set()
            await asyncio.gather(run, return_exceptions=True)
            await resources.wait_pending()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert checkpoint_closed.is_set()
        assert not source.exists()
        for key in (("index-owner", "maintenance"), ("base-document", source.name)):
            await (await peer.obtain(key)).close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "module",
    [
        "med_evidence_agentic_rag.rag_creation.base_corpus",
        "med_evidence_agentic_rag.api.app",
    ],
)
def test_contending_entry_points_stop_before_chroma_import(tmp_path, module):
    root = tmp_path / "chroma"
    env = dict(os.environ, CHROMA_PERSIST_DIRECTORY=str(root))
    source = """
import asyncio, importlib, runpy, sys
from fastapi import HTTPException
from med_evidence_agentic_rag.services.coordination import local_guards

class DetectChroma:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "med_evidence_agentic_rag.models_and_database.chroma":
            raise AssertionError("Chroma imported before ownership")

sys.meta_path.insert(0, DetectChroma())
module = sys.argv[1]
lease = asyncio.run(local_guards.obtain(("index-owner", "maintenance")))
try:
    try:
        if module.endswith(".app"):
            app = importlib.import_module(module)
            async def start():
                async with app.lifespan(app.app):
                    pass
            asyncio.run(start())
        else:
            runpy.run_module(module, run_name="__main__")
    except HTTPException as error:
        assert error.status_code == 409
    else:
        raise AssertionError("Younger entry point acquired a held index")
finally:
    asyncio.run(lease.close())
"""
    result = subprocess.run(
        [sys.executable, "-c", source, module],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert not (root / "chroma.sqlite3").exists()
