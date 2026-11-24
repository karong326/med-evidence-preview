import asyncio
import os
import subprocess
import sys

import pytest
from fastapi import HTTPException

from med_evidence_agentic_rag.services.coordination import resources


@pytest.mark.parametrize("option", ["absent-apply", "dotenv-only", "unowned-root"])
def test_recovery_entry_point_requires_explicit_owned_target(tmp_path, option):
    uploads, base = tmp_path / "data" / "uploads", tmp_path / "data" / "raw"
    for root in (uploads, base):
        root.mkdir(parents=True)
    source = uploads / "untouched.txt"
    source.write_text("synthetic")
    env = os.environ.copy()
    target = {
        "POSTGRESQL_URL": env["POSTGRESQL_URL"],
        "CHROMA_PERSIST_DIRECTORY": str(tmp_path / "chroma"),
        "CHROMA_COLLECTION_NAME": "synthetic",
    }
    env.update(target)
    if option == "dotenv-only":
        (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in target.items()))
        for key in target:
            del env[key]
    cmd = [
        sys.executable,
        "-m",
        "med_evidence_agentic_rag.services.recovery",
        "--uploads",
        str(uploads if option != "unowned-root" else base),
        "--base",
        str(base),
    ]
    if option != "absent-apply":
        cmd.append("--apply")
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stderr
    assert source.read_text() == "synthetic"
    assert not (tmp_path / "chroma").exists()


@pytest.mark.parametrize("kind", ["conversation", "document"])
def test_owner_deletion_conflicts_in_both_directions(kind):
    async def scenario():
        lease = await resources.obtain((kind, "owner", "key"))
        with pytest.raises(HTTPException) as busy:
            await resources.obtain(("owner-delete", "owner"))
        assert busy.value.status_code == 409
        await lease.close()
        exclusive = await resources.obtain(("owner-delete", "owner"))
        with pytest.raises(HTTPException):
            await resources.obtain((kind, "owner", "other"))
        await exclusive.close()

    asyncio.run(scenario())


def test_pending_upload_blocks_filename_reuse_until_compensation(email_sessions, tmp_path):
    from uuid import uuid4

    from med_evidence_agentic_rag.persistence.operations import Journal
    from med_evidence_agentic_rag.services.recovery import repair_files

    async def scenario():
        owner = uuid4()
        journal = Journal(email_sessions)
        op = await journal.begin("upload", owner, uuid4(), "partial.txt")
        with pytest.raises(HTTPException):
            await journal.begin("upload", owner, uuid4(), "partial.txt")
        folder = tmp_path / str(owner)
        folder.mkdir()
        (folder / "partial.txt").write_bytes(b"partial")
        staging = folder / f".{op.id}.part"
        staging.write_bytes(b"staged")
        deleted = []
        for _ in range(2):
            for pending in await journal.pending(10):
                await repair_files(pending, tmp_path, tmp_path, deleted.append)
                await journal.finish(pending.id)
        assert deleted == [("document_id", str(op.resource_id))]
        assert not (folder / "partial.txt").exists()
        assert not staging.exists()

    asyncio.run(scenario())


def test_recovery_rejects_escaping_owner_directory(email_sessions, tmp_path):
    from uuid import uuid4

    from med_evidence_agentic_rag.persistence.operations import Journal
    from med_evidence_agentic_rag.services.recovery import repair_files

    async def scenario():
        journal = Journal(email_sessions)
        pending = await journal.begin("upload", uuid4(), uuid4(), "../escape.txt")
        deleted = []
        with pytest.raises(ValueError):
            await repair_files(pending, tmp_path, tmp_path, deleted.append)
        assert deleted == []

    asyncio.run(scenario())


def test_owner_reconciles_unknown_once_without_redelivery(email_sessions):
    from uuid import uuid4

    from med_evidence_agentic_rag.services.email_approval import (
        ApprovalError,
        EmailApprovalRegistry,
    )

    async def scenario():
        owner = uuid4()
        registry = EmailApprovalRegistry(email_sessions)
        item = await registry.propose(owner, uuid4(), "test@example.org", "test", "demo")
        calls = []

        def uncertain(*args):
            calls.append(1)
            raise RuntimeError("lost acknowledgment")

        assert (await registry.confirm(owner, item.id, uncertain)).status == "unknown"
        with pytest.raises(ApprovalError):
            await registry.reconcile(uuid4(), item.id, delivered=True)
        result = await registry.reconcile(owner, item.id, delivered=False)
        assert result.status == "failed"
        with pytest.raises(ApprovalError):
            await registry.reconcile(owner, item.id, delivered=True)
        assert (await registry.confirm(owner, item.id, uncertain)).status == "failed"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_restart_recovers_legacy_delete_chat_and_stops_after_attempt_limit(
    email_sessions, tmp_path
):
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from sqlalchemy import select

    from med_evidence_agentic_rag.persistence.operations import Journal
    from med_evidence_agentic_rag.persistence.postgres_tables import (
        ChatTurn,
        Conversation,
        DocumentRecord,
        Operation,
        User,
    )
    from med_evidence_agentic_rag.services.recovery import recover

    async def scenario():
        owner, conversation, turn, doc = (uuid4() for _ in range(4))
        store = Journal(email_sessions)
        source = tmp_path / str(owner) / "legacy.txt"
        source.parent.mkdir()
        source.write_bytes(b"demo")
        async with email_sessions.begin() as session:
            session.add(User(id=owner, email="owner@example.org"))
            session.add(Conversation(id=conversation, user_id=owner))
            session.add(
                ChatTurn(
                    id=turn,
                    conversation_id=conversation,
                    question="test",
                    turn_number=1,
                    status="processing",
                )
            )
            session.add(
                DocumentRecord(
                    id=doc,
                    owner_user_id=owner,
                    filename=source.name,
                    source_path=str(source),
                    status="delete_failed",
                )
            )
        bad = await store.begin("base-upload", None, None, "../unsafe.txt")
        deleted = []

        async def run():
            return await recover(
                tmp_path, tmp_path, AsyncMock(), store=store, remove=deleted.append, attempts=2
            )

        lease = await resources.obtain(("conversation", str(owner), str(conversation)))
        first = await run()
        assert first["busy"] == 1
        assert first["failed"] == 1
        assert not source.exists()
        assert deleted == [("document_id", str(doc))]
        await lease.close()
        await run()
        assert (await run()) == {"completed": 0, "busy": 0, "failed": 0}
        async with email_sessions() as session:
            assert (await session.get(ChatTurn, turn)).failure_code == "worker_interrupted"
            blocked = await session.get(Operation, bad.id)
            assert blocked.status == "blocked" and blocked.attempts == 2
            assert await session.scalar(select(DocumentRecord.id)) is None

    asyncio.run(scenario())


def test_vector_cleanup_uses_existing_collection_without_embedding(tmp_path, monkeypatch):
    from med_evidence_agentic_rag.models_and_database.chroma_client import persistent_client
    from med_evidence_agentic_rag.services.recovery import delete_vectors
    from med_evidence_agentic_rag.utils import config

    client = persistent_client(tmp_path)
    assert not client.get_settings().anonymized_telemetry
    collection = client.create_collection("recovery_test", embedding_function=None)
    collection.add(
        ids=["partial", "keep"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"document_id": "abandoned"}, {"document_id": "other"}],
    )
    monkeypatch.setattr(config, "chroma_persist_directory", tmp_path)
    monkeypatch.setattr(config, "chroma_collection_name", "recovery_test")
    for _ in range(2):
        delete_vectors(("document_id", "abandoned"))
    assert collection.get()["ids"] == ["keep"]
