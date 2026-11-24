"""Bounded maintenance for one application/index owner and its local storage."""

if __name__ == "__main__":
    # Parse the target before config can populate missing variables from a local .env.
    from med_evidence_agentic_rag.services.recovery_cli import main

    main()
    raise SystemExit

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, select, update

from med_evidence_agentic_rag.persistence.documents import source_is_registered
from med_evidence_agentic_rag.persistence.operations import Journal, journal
from med_evidence_agentic_rag.persistence.postgres_tables import (
    ChatTurn,
    Conversation,
    DocumentRecord,
    Operation,
    User,
    utc_now,
)
from med_evidence_agentic_rag.services.coordination import (
    ResourceRegistry,
    resources,
    run_protected,
)
from med_evidence_agentic_rag.services.paths import scoped_source


async def register_legacy_documents(store: Journal, locks: ResourceRegistry, limit: int) -> None:
    """Discover pre-journal registrations without superseding a newer named asset."""
    abandoned = ("processing", "failed", "deleting", "delete_failed")
    recorded = select(Operation.id).where(Operation.resource_id == DocumentRecord.id).exists()
    async with store.sessions() as session:
        legacy = list(
            await session.scalars(
                select(DocumentRecord)
                .where(DocumentRecord.status.in_(abandoned), ~recorded)
                .order_by(DocumentRecord.updated_at, DocumentRecord.id)
                .limit(limit)
            )
        )
    for candidate in legacy:
        try:
            lease = await locks.obtain(
                ("document", str(candidate.owner_user_id), candidate.filename.casefold())
            )
        except HTTPException as error:
            if error.status_code != 409:
                raise
            continue
        try:
            async with store.sessions() as session:
                row = await session.scalar(
                    select(DocumentRecord).where(
                        DocumentRecord.id == candidate.id,
                        DocumentRecord.status.in_(abandoned),
                        ~recorded,
                    )
                )
            if row:
                kind = "delete" if row.status in ("deleting", "delete_failed") else "upload"
                try:
                    await store.begin(kind, row.owner_user_id, row.id, row.filename)
                except HTTPException as conflict:
                    if conflict.status_code != 409:
                        raise
        finally:
            await lease.close()


def delete_vectors(selector: tuple[str, str]) -> None:
    from med_evidence_agentic_rag.models_and_database.chroma_client import persistent_client
    from med_evidence_agentic_rag.utils.config import (
        chroma_collection_name,
        chroma_persist_directory,
    )

    if not (chroma_persist_directory / "chroma.sqlite3").is_file():
        raise RuntimeError("Configured vector storage missing")
    collection = persistent_client(chroma_persist_directory).get_collection(
        chroma_collection_name, embedding_function=None
    )
    collection.delete(where={selector[0]: selector[1]})


async def repair_files(
    row: Operation,
    uploads: Path,
    base: Path,
    remove: Callable[[tuple[str, str]], None],
    *,
    remove_source: bool = True,
) -> None:
    if row.kind in ("upload", "delete") and row.owner_id and row.resource_id:
        source = scoped_source(uploads, row.owner_id, row.filename)
        selector = ("document_id", str(row.resource_id))
    elif row.kind in ("base-upload", "base-delete") and row.owner_id is None:
        source = scoped_source(base, None, row.filename)
        selector = ("source", str(source))
    else:
        raise ValueError("Invalid operation identity")
    await asyncio.to_thread(remove, selector)
    if row.kind in ("upload", "base-upload"):
        source.with_name(f".{row.id}.part").unlink(missing_ok=True)
    if remove_source:
        source.unlink(missing_ok=True)


async def remove_account(
    owner: UUID,
    uploads: Path,
    store: Journal,
    remove: Callable[[tuple[str, str]], None],
    delete_thread: Callable[[str], Awaitable[None]],
) -> None:
    async with store.sessions() as session:
        user = await session.get(User, owner)
        if user and user.auth_type != "access_token":
            raise ValueError("Account type mismatch")
        documents = list(
            await session.scalars(
                select(DocumentRecord).where(DocumentRecord.owner_user_id == owner)
            )
        )
        threads = list(
            await session.scalars(select(Conversation.id).where(Conversation.user_id == owner))
        )
    for document in documents:
        source = scoped_source(uploads, owner, document.filename)
        if source != Path(document.source_path).resolve():
            raise ValueError("Source registration mismatch")
        await asyncio.to_thread(remove, ("document_id", str(document.id)))
        source.unlink(missing_ok=True)
    for thread in threads:
        await delete_thread(str(thread))
    directory = uploads.resolve() / str(owner)
    if directory.resolve() != directory:
        raise ValueError("Owner directory escapes upload root")
    if directory.exists():
        # Unexpected unregistered content needs an operator, never a recursive purge.
        directory.rmdir()
    async with store.sessions.begin() as session:
        await session.execute(
            delete(User).where(User.id == owner, User.auth_type == "access_token")
        )


async def recover(
    uploads: Path,
    base: Path,
    delete_thread: Callable[[str], Awaitable[None]],
    *,
    limit: int = 20,
    attempts: int = 3,
    store: Journal = journal,
    locks: ResourceRegistry = resources,
    remove: Callable[[tuple[str, str]], None] = delete_vectors,
) -> dict[str, int]:
    if not 1 <= limit <= 500 or not 1 <= attempts <= 20:
        raise ValueError("Invalid recovery bounds")
    counts = {"completed": 0, "busy": 0, "failed": 0}
    await register_legacy_documents(store, locks, limit)
    for candidate in await store.pending(limit):
        owner = candidate.owner_id
        if candidate.kind == "owner-delete":
            key = ("owner-delete", str(owner))
        elif candidate.kind == "conversation-delete":
            key = ("conversation-delete", str(owner), str(candidate.resource_id))
        elif owner:
            key = ("document", str(owner), candidate.filename.casefold())
        else:
            key = ("base-document", candidate.filename.casefold())
        try:
            lease = await locks.obtain(key)
        except HTTPException as error:
            if error.status_code != 409:
                raise
            counts["busy"] += 1
            continue

        async def reconcile(identity: UUID = candidate.id) -> None:
            row = await store.get(identity)
            if row is None or row.status != "pending":
                return
            try:
                if row.kind == "owner-delete" and row.owner_id:
                    await remove_account(row.owner_id, uploads, store, remove, delete_thread)
                elif row.kind == "conversation-delete" and row.resource_id and row.owner_id:
                    async with store.sessions.begin() as session:
                        chat = await session.get(Conversation, row.resource_id)
                        if chat and chat.user_id != row.owner_id:
                            raise ValueError("Conversation owner mismatch")
                        if chat:
                            await delete_thread(str(chat.id))
                            await session.delete(chat)
                else:
                    async with store.sessions() as session:
                        doc = (
                            await session.get(DocumentRecord, row.resource_id)
                            if row.resource_id
                            else None
                        )
                        shared_source = False
                        if doc and row.owner_id is None:
                            raise ValueError("Source registration mismatch")
                        if row.owner_id is not None:
                            source = scoped_source(uploads, row.owner_id, row.filename)
                            if doc and (
                                doc.owner_user_id != row.owner_id
                                or doc.filename != row.filename
                                or Path(doc.source_path).resolve() != source
                            ):
                                raise ValueError("Source registration mismatch")
                            shared_source = await source_is_registered(
                                session, row.owner_id, source, excluding=row.resource_id
                            )
                    if not (doc and row.kind == "upload" and doc.status == "ready"):
                        await repair_files(
                            row, uploads, base, remove, remove_source=not shared_source
                        )
                        if doc:
                            async with store.sessions.begin() as session:
                                if row.kind == "delete":
                                    await session.execute(
                                        delete(DocumentRecord).where(DocumentRecord.id == doc.id)
                                    )
                                else:
                                    await session.execute(
                                        update(DocumentRecord)
                                        .where(DocumentRecord.id == doc.id)
                                        .values(status="failed", updated_at=utc_now())
                                    )
                await store.finish(identity)
                counts["completed"] += 1
            except Exception:
                await store.failed(row, attempts)
                counts["failed"] += 1

        await run_protected(lease, reconcile)
    async with store.sessions() as session:
        turns = (
            await session.execute(
                select(ChatTurn.id, ChatTurn.conversation_id, Conversation.user_id)
                .join(Conversation)
                .where(ChatTurn.status == "processing")
                .order_by(ChatTurn.question_at)
                .limit(limit)
            )
        ).all()
    for turn, conversation, owner in turns:
        try:
            lease = await locks.obtain(("conversation", str(owner), str(conversation)))
        except HTTPException as error:
            if error.status_code != 409:
                raise
            counts["busy"] += 1
            continue
        try:
            async with store.sessions.begin() as session:
                changed = await session.scalar(
                    update(ChatTurn)
                    .where(ChatTurn.id == turn, ChatTurn.status == "processing")
                    .values(
                        status="failed",
                        failure_code="worker_interrupted",
                        failure_message="服务已中断，请重试。",
                        answer_at=utc_now(),
                    )
                    .returning(ChatTurn.id)
                )
                counts["completed"] += int(changed is not None)
        finally:
            await lease.close()
    return counts
