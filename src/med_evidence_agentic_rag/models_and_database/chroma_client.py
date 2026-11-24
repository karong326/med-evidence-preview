"""One configuration for the runtime index and model-free recovery."""

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings


def persistent_client(path: Path) -> ClientAPI:
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
