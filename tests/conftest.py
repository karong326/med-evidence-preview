"""All pytest processes use synthetic service settings, never the local business databases."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

_runtime = tempfile.TemporaryDirectory(prefix="medevidence-tests-")
os.environ.update(
    {
        "POSTGRESQL_URL": "postgresql+asyncpg://test:test@127.0.0.1:1/medevidence_test",
        "REDIS_URL": "redis://127.0.0.1:1/0",
        "CHROMA_PERSIST_DIRECTORY": str(Path(_runtime.name) / "chroma"),
        "CHROMA_COLLECTION_NAME": "isolated_tests",
        "JWT_SECRET_KEY": "isolated-tests-only-" * 3,
        "LLM_MODEL_NAME": "isolated-test-model",
        "LLM_BASE_URL": "http://127.0.0.1:1/v1",
        "LLM_API_KEY": "isolated-tests-only-api-key",
        "EMBEDDING_MODEL_PATH": str(Path(_runtime.name) / "unused-embedding"),
        "RERANKER_MODEL_PATH": str(Path(_runtime.name) / "unused-reranker"),
        "MULTILINGUAL_EMBEDDING_MODEL_PATH": "",
        "ACCESS_TOKEN_CODE": "isolated-user-code",
        "ADMIN_ACCESS_CODE": "isolated-admin-code",
        "MY_SENDER_EMAIL": "",
        "MY_SENDER_PASSWORD": "",
        "MINERU_TOKEN": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)


@pytest.fixture
def email_sessions(tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from med_evidence_agentic_rag.persistence.postgres_tables import Base

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}", poolclass=NullPool
    )

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    yield async_sessionmaker(engine, expire_on_commit=False)
    asyncio.run(engine.dispose())


@pytest.fixture(autouse=True)
def offline_resource_locks(monkeypatch, tmp_path):
    from med_evidence_agentic_rag.services.coordination import resources

    monkeypatch.setattr(resources, "root", tmp_path / "locks")
    monkeypatch.setattr(resources, "engine", None)
    monkeypatch.setattr(resources, "owner_deleting", None)


@pytest.fixture
def isolated_journal(email_sessions, monkeypatch):
    from med_evidence_agentic_rag.persistence.operations import journal

    monkeypatch.setattr(journal, "sessions", email_sessions)
