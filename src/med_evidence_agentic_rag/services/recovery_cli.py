"""Run bounded offline recovery for a selected deployment."""

import argparse
import asyncio
import os
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any


def run_recovery(operation: Coroutine[Any, Any, None]) -> None:
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(operation)
    else:
        asyncio.run(operation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--uploads", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    if not args.apply or not all(
        os.getenv(key)
        for key in ("POSTGRESQL_URL", "CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME")
    ):
        parser.error("Explicit PostgreSQL and Chroma environment plus --apply required")
    if not all(path.is_absolute() and path.is_dir() for path in (args.uploads, args.base)):
        parser.error("Existing absolute source roots required")

    from med_evidence_agentic_rag.utils.config import DATA_ROOT

    if (
        args.uploads.resolve() != (DATA_ROOT / "uploads").resolve()
        or args.base.resolve() != (DATA_ROOT / "raw").resolve()
    ):
        parser.error("Source roots must match the selected application deployment")

    if not 1 <= args.limit <= 500 or not 1 <= args.attempts <= 20:
        parser.error("Invalid recovery bounds")

    async def run() -> None:
        from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import (
            open_checkpointer,
        )
        from med_evidence_agentic_rag.persistence.init_postgres import create_application_tables
        from med_evidence_agentic_rag.services.coordination import local_guards, resources
        from med_evidence_agentic_rag.services.email_approval import email_approvals
        from med_evidence_agentic_rag.services.recovery import recover

        lease = await local_guards.obtain(("index-owner", "maintenance"))
        try:
            await create_application_tables()
            async with open_checkpointer() as saver:
                try:
                    result = await recover(
                        args.uploads,
                        args.base,
                        saver.adelete_thread,
                        limit=args.limit,
                        attempts=args.attempts,
                    )
                finally:
                    await resources.wait_pending()
            result["emails_unknown"] = await email_approvals.recover_abandoned(args.limit)
            print(result)
            if result["failed"]:
                raise SystemExit(1)
        finally:
            await lease.close()

    from med_evidence_agentic_rag.utils.logger import configure_server_logging, setup_logger

    configure_server_logging()
    try:
        run_recovery(run())
    except Exception:
        setup_logger(__name__).exception("Recovery did not complete")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
