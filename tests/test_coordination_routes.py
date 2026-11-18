"""Real routes, isolated stores, and controlled worker/stream lifetimes."""

import asyncio
import importlib.util
import io
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.requests import Request

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.services.coordination import resources


def load_routes(name, filename):
    path = Path(__file__).parents[1] / "src/med_evidence_agentic_rag/api" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chat(monkeypatch):
    agent = ModuleType("isolated_coordination_agent")
    agent.agent = object()
    agent.stream_chat = AsyncMock()
    monkeypatch.setitem(sys.modules, "med_evidence_agentic_rag.agent.agent", agent)
    routes = load_routes("isolated_coordination_chat", "chat.py")
    monkeypatch.setattr(routes, "get_owned_conversation", AsyncMock(return_value=object()))
    monkeypatch.setattr(routes, "list_recent_completed_turns", AsyncMock(return_value=[]))
    monkeypatch.setattr(routes, "get_user_personalization", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(routes, "build_agent_input", lambda *a, **k: "question")
    monkeypatch.setattr(routes, "build_user_context_trace", lambda *a: None)
    monkeypatch.setattr(routes, "mark_chat_turn_failed", AsyncMock())
    created = AsyncMock(return_value=SimpleNamespace(id=uuid4(), turn_number=1))
    monkeypatch.setattr(routes, "create_chat_turn", created)
    return routes


def test_ask_conflict_precedes_new_turn_and_unstarted_close_releases(chat):
    async def run():
        user = AuthenticatedUser(uuid4(), "a@example.org")
        body = chat.AskRequest(question="question", conversation_id=uuid4())
        request = Request({"type": "http", "app": FastAPI()})
        response = await chat.ask(body, request, user)
        with pytest.raises(HTTPException) as conflict:
            await chat.ask(body, request, user)
        assert conflict.value.status_code == 409
        assert chat.create_chat_turn.await_count == 1
        await response.aclose()
        chat.mark_chat_turn_failed.assert_awaited_once()
        assert not resources.active

    asyncio.run(run())


def test_ask_preparation_failure_releases_lease(chat, monkeypatch):
    monkeypatch.setattr(
        chat, "get_user_personalization", AsyncMock(side_effect=RuntimeError("test"))
    )

    async def run():
        user = AuthenticatedUser(uuid4(), "a@example.org")
        body = chat.AskRequest(question="question", conversation_id=uuid4())
        with pytest.raises(RuntimeError):
            await chat.ask(body, Request({"type": "http", "app": FastAPI()}), user)
        assert not resources.active
        chat.create_chat_turn.assert_not_awaited()

    asyncio.run(run())


def test_cancel_during_committed_turn_creation_keeps_lease_until_failure_saved(chat, monkeypatch):
    async def run():
        committed, refresh = asyncio.Event(), asyncio.Event()
        turn = SimpleNamespace(id=uuid4(), turn_number=1)

        async def create(*args):
            committed.set()
            await refresh.wait()
            return turn

        monkeypatch.setattr(chat, "create_chat_turn", create)
        user = AuthenticatedUser(uuid4(), "a@example.org")
        body = chat.AskRequest(question="question", conversation_id=uuid4())
        task = asyncio.create_task(
            chat.ask(body, Request({"type": "http", "app": FastAPI()}), user)
        )
        await committed.wait()
        task.cancel()
        await asyncio.sleep(0)
        try:
            assert resources.active
        finally:
            refresh.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        chat.mark_chat_turn_failed.assert_awaited_once()
        assert chat.mark_chat_turn_failed.await_args.args[0] == turn.id
        assert not resources.active

    asyncio.run(run())


def test_delete_http_checks_ownership_before_busy_response(monkeypatch):
    from med_evidence_agentic_rag.api import conversations

    user = AuthenticatedUser(uuid4(), "a@example.org")
    conversation_id = uuid4()
    app = FastAPI()
    app.include_router(conversations.conversation_router)
    app.dependency_overrides[require_user] = lambda: user
    owned = AsyncMock(return_value=None)
    monkeypatch.setattr(conversations, "get_owned_conversation", owned)
    lease = resources.acquire(("conversation", str(user.id), str(conversation_id)))
    try:
        with TestClient(app) as client:
            assert client.delete(f"/api/conversations/{conversation_id}").status_code == 404
            owned.return_value = object()
            assert client.delete(f"/api/conversations/{conversation_id}").status_code == 409
    finally:
        lease.release()


def test_cancelled_upload_blocks_delete_until_vectors_and_status_finish(monkeypatch, tmp_path):
    ingestion = ModuleType("isolated_coordination_ingestion")
    started, finish = threading.Event(), threading.Event()

    def index(*args, **kwargs):
        started.set()
        assert finish.wait(5)
        return 2

    ingestion.index_document = index
    ingestion.delete_indexed_document = lambda *a: None
    deletion = ModuleType("isolated_coordination_deletion")
    deletion.DocumentDeletionError = RuntimeError
    deletion.delete_document_assets = AsyncMock()
    monkeypatch.setitem(sys.modules, "med_evidence_agentic_rag.rag_creation.ingestion", ingestion)
    monkeypatch.setitem(
        sys.modules, "med_evidence_agentic_rag.services.document_deletion", deletion
    )
    routes = load_routes("isolated_coordination_documents", "documents.py")
    user = AuthenticatedUser(uuid4(), "a@example.org")
    record = SimpleNamespace(id=uuid4(), filename="test.txt")
    monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)
    monkeypatch.setattr(routes, "create_document", AsyncMock(return_value=record))
    monkeypatch.setattr(routes, "get_owned_document", AsyncMock(return_value=record))
    ready = AsyncMock()
    monkeypatch.setattr(routes, "mark_document_ready", ready)

    async def run():
        upload = UploadFile(io.BytesIO(b"synthetic"), filename="test.txt")
        task = asyncio.create_task(routes.upload_document(upload, user))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(HTTPException) as error:
                await routes.delete_document(record.id, user)
            assert error.value.status_code == 409
            deletion.delete_document_assets.assert_not_awaited()
        finally:
            finish.set()
        for _ in range(1000):
            if not resources.active:
                break
            await asyncio.sleep(0.001)
        ready.assert_awaited_once_with(record.id, 2)
        assert not resources.active
        assert await routes.delete_document(record.id, user) == {"deleted": True}

    asyncio.run(run())


@pytest.mark.parametrize("base", [False, True])
def test_closed_upload_cleans_partial_file_and_releases_lease(monkeypatch, tmp_path, base):
    ingestion = ModuleType("isolated_closed_upload_ingestion")
    ingestion.index_document = lambda *a, **k: 1
    ingestion.delete_indexed_document = lambda *a: None
    monkeypatch.setitem(sys.modules, "med_evidence_agentic_rag.rag_creation.ingestion", ingestion)
    user = AuthenticatedUser(uuid4(), "a@example.org")
    if base:
        corpus = ModuleType("isolated_closed_base_corpus")
        corpus.BASE_CORPUS_DIRECTORY = tmp_path
        corpus.discover_base_documents = lambda: []
        monkeypatch.setitem(
            sys.modules, "med_evidence_agentic_rag.rag_creation.base_corpus", corpus
        )
        routes = load_routes("isolated_closed_base_upload", "admin_documents.py")
        monkeypatch.setattr(routes, "BASE_CORPUS_DIRECTORY", tmp_path)
        upload = routes.upload_base_document
        target = tmp_path / "test.txt"
    else:
        routes = load_routes("isolated_closed_upload", "documents.py")
        monkeypatch.setattr(routes, "UPLOAD_DIRECTORY", tmp_path)
        upload = routes.upload_document
        target = tmp_path / str(user.id) / "test.txt"
    file = UploadFile(io.BytesIO(b""), filename="test.txt")
    monkeypatch.setattr(file, "read", AsyncMock(side_effect=[b"partial", ValueError("closed")]))
    with pytest.raises(ValueError):
        asyncio.run(upload(file, user))
    assert not resources.active
    assert not target.exists()
