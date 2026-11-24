"""Exercise real routes and deletion service with temporary files and injected stores."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user

pytestmark = pytest.mark.usefixtures("isolated_journal")


def load_file(name, relative):
    path = Path(__file__).parents[1] / "src" / "med_evidence_agentic_rag" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def documents(monkeypatch, tmp_path):
    ingestion = ModuleType("isolated_ingestion")
    ingestion.index_document = lambda *args, **kwargs: 1
    ingestion.delete_indexed_document = lambda *args: None
    deletion = ModuleType("isolated_deletion")
    deletion.DocumentDeletionError = RuntimeError
    deletion.delete_document_assets = AsyncMock()
    monkeypatch.setitem(sys.modules, "med_evidence_agentic_rag.rag_creation.ingestion", ingestion)
    monkeypatch.setitem(
        sys.modules, "med_evidence_agentic_rag.services.document_deletion", deletion
    )
    module = load_file("isolated_document_routes", Path("api/documents.py"))
    monkeypatch.setattr(module, "UPLOAD_DIRECTORY", tmp_path)
    user = AuthenticatedUser(uuid4(), "test@example.org")
    app = FastAPI()
    app.include_router(module.document_router)
    app.dependency_overrides[require_user] = lambda: user
    created = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    monkeypatch.setattr(module, "create_document", created)
    monkeypatch.setattr(module, "mark_document_ready", AsyncMock())
    monkeypatch.setattr(module, "mark_document_failed", AsyncMock())
    with TestClient(app) as client:
        yield client, module, user, created


def test_upload_success_uses_owner_directory_and_rejects_duplicate(documents):
    client, module, user, created = documents
    result = client.post("/api/documents/upload", files={"file": ("demo.txt", b"synthetic text")})
    assert result.status_code == 200
    assert result.json()["status"] == "ready"
    assert (module.UPLOAD_DIRECTORY / str(user.id) / "demo.txt").read_bytes() == b"synthetic text"
    assert created.await_args.kwargs["owner_user_id"] == user.id
    assert (
        client.post("/api/documents/upload", files={"file": ("demo.txt", b"duplicate")}).status_code
        == 409
    )


@pytest.mark.parametrize("name,content", [("demo.exe", b"bad format"), ("empty.txt", b"")])
def test_invalid_upload_does_not_register_document(documents, name, content):
    client, _, _, created = documents
    assert client.post("/api/documents/upload", files={"file": (name, content)}).status_code == 400
    created.assert_not_awaited()


def test_failed_upload_cleans_file_and_vectors_and_marks_failure(documents, monkeypatch):
    client, module, user, _ = documents

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic indexing failure")

    monkeypatch.setattr(module, "index_document", fail)
    removed = []
    monkeypatch.setattr(module, "delete_indexed_document", removed.append)
    assert (
        client.post("/api/documents/upload", files={"file": ("demo.txt", b"test")}).status_code
        == 500
    )
    assert not (module.UPLOAD_DIRECTORY / str(user.id) / "demo.txt").exists()
    module.mark_document_failed.assert_awaited_once()
    assert len(removed) == 1


def test_foreign_document_is_not_deleted(documents, monkeypatch):
    client, module, user, _ = documents
    owned = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_owned_document", owned)
    document_id = uuid4()
    assert client.delete(f"/api/documents/{document_id}").status_code == 404
    owned.assert_awaited_once_with(user.id, document_id)
    module.delete_document_assets.assert_not_awaited()


def test_real_deletion_blocks_outside_path_and_is_retryable(monkeypatch, tmp_path):
    ingestion = ModuleType("test_deletion_ingestion")
    ingestion.delete_indexed_document = lambda *args: None
    monkeypatch.setitem(sys.modules, "med_evidence_agentic_rag.rag_creation.ingestion", ingestion)
    module = load_file("isolated_document_deletion", Path("services/document_deletion.py"))
    mark = AsyncMock(return_value=True)
    failed = AsyncMock()
    removed = AsyncMock()
    monkeypatch.setattr(module, "mark_document_deleting", mark)
    monkeypatch.setattr(module, "mark_document_delete_failed", failed)
    monkeypatch.setattr(module, "delete_document_record", removed)
    root = tmp_path / "owner"
    root.mkdir()
    foreign = tmp_path / "foreign.txt"
    foreign.write_text("keep", encoding="utf-8")
    record = SimpleNamespace(id=uuid4(), source_path=str(foreign))
    with pytest.raises(module.DocumentDeletionError):
        asyncio.run(module.delete_document_assets(record, upload_root=root))
    assert foreign.exists()
    mark.assert_not_awaited()
    own = root / "own.txt"
    own.write_text("test", encoding="utf-8")
    record.source_path = str(own)

    def fail(*args):
        raise RuntimeError("synthetic vector failure")

    monkeypatch.setattr(module, "delete_indexed_document", fail)
    with pytest.raises(module.DocumentDeletionError):
        asyncio.run(module.delete_document_assets(record, upload_root=root))
    assert own.exists()
    removed.assert_not_awaited()
    failed.assert_awaited_once()
    monkeypatch.setattr(module, "delete_indexed_document", lambda *args: None)
    asyncio.run(module.delete_document_assets(record, upload_root=root))
    assert not own.exists()
    removed.assert_awaited_once_with(record.id)
