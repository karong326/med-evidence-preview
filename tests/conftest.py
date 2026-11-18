"""All pytest processes use synthetic service settings, never the local business databases."""

import os
import tempfile
from pathlib import Path

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
