"""处理登录用户个人资料的上传、列表与删除。

上传接口校验扩展名、文件名安全性和 200MB 上限，将文件保存到用户专属 uploads 目录，再在
工作线程执行加载、切分和 Chroma 入库；失败时清理向量并记录可重试失败状态。删除接口通过
统一资料删除服务依次清理向量、源文件和 PostgreSQL 登记，保证用户隔离与清理顺序一致。
"""

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.persistence.documents import (
    create_document,
    get_owned_document,
    list_user_documents,
    mark_document_failed,
    mark_document_ready,
    source_is_registered,
)
from med_evidence_agentic_rag.persistence.operations import journal
from med_evidence_agentic_rag.rag_creation.ingestion import (
    delete_indexed_document,
    index_document,
)
from med_evidence_agentic_rag.rag_creation.loader import DocumentLoader
from med_evidence_agentic_rag.services.coordination import resources, run_protected
from med_evidence_agentic_rag.services.document_deletion import (
    DocumentDeletionError,
    delete_document_assets,
)
from med_evidence_agentic_rag.services.paths import scoped_source
from med_evidence_agentic_rag.utils.config import DATA_ROOT
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
document_router = APIRouter(prefix="/api/documents", tags=["资料"])

UPLOAD_DIRECTORY = DATA_ROOT / "uploads"
UPLOAD_MAX_BYTES = 200 * 1024 * 1024
SUPPORTED_SUFFIXES = DocumentLoader.MINERU_SUFFIXES | DocumentLoader.TEXT_SUFFIXES


class DocumentResponse(BaseModel):
    """返回当前用户可管理的资料状态。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: str
    chunk_count: int
    created_at: datetime


@document_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: AuthenticatedUser = Depends(require_user),
) -> dict[str, str | int]:
    """安全保存一份上传文件，然后在线程中完成加载、切块和入库。"""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持该上传文件格式")

    lease = await resources.obtain(("document", str(current_user.id), filename.casefold()))
    return await run_protected(lease, lambda: _upload_document(file, current_user, filename))


async def _upload_document(
    file: UploadFile,
    current_user: AuthenticatedUser,
    filename: str,
) -> dict[str, str | int]:
    try:
        target = scoped_source(UPLOAD_DIRECTORY, current_user.id, filename)
    except ValueError as error:
        raise HTTPException(400, "上传文件名不安全") from error
    if target.exists():
        raise HTTPException(status_code=409, detail="同名文件已存在")
    async with journal.sessions() as session:
        if await source_is_registered(session, current_user.id, target, match_filename=True):
            raise HTTPException(status_code=409, detail="请先删除同名资料登记")
    identity = uuid4()
    op = await journal.begin("upload", current_user.id, identity, filename)
    staging = target.with_name(f".{op.id}.part")
    target.parent.mkdir(parents=True, exist_ok=True)

    # 分块写入并限制总大小，避免一次把大文件全部读入内存。
    total_bytes = 0
    try:
        with staging.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > UPLOAD_MAX_BYTES:
                    output.close()
                    staging.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="上传文件超过 200MB")
                output.write(chunk)
    except BaseException:
        staging.unlink(missing_ok=True)
        await journal.finish(op.id)
        raise
    if total_bytes == 0:
        staging.unlink(missing_ok=True)
        await journal.finish(op.id)
        raise HTTPException(status_code=400, detail="上传文件不能为空")

    staging.replace(target)

    record = None
    try:
        record = await create_document(
            owner_user_id=current_user.id,
            filename=filename,
            source_path=str(target),
            document_id=identity,
        )
        await journal.set_phase(op.id, "indexing")
        chunk_count = await asyncio.to_thread(
            index_document,
            target,
            "upload",
            document_id=str(record.id),
            owner_user_id=str(current_user.id),
        )
    except Exception as exc:
        compensated = False
        if record is not None:
            try:
                await asyncio.to_thread(delete_indexed_document, str(record.id))
                compensated = True
            except Exception:
                logger.exception("Failed upload vector cleanup failed")
            await mark_document_failed(record.id)
        target.unlink(missing_ok=True)
        if compensated:
            await journal.finish(op.id)
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail="上传资料解析或入库失败") from exc

    await mark_document_ready(record.id, chunk_count)
    await journal.finish(op.id)
    logger.info("Document indexed: chunks=%d", chunk_count)
    return {
        "id": str(record.id),
        "filename": filename,
        "chunks": chunk_count,
        "status": "ready",
    }


@document_router.get("")
async def get_documents(
    current_user: AuthenticatedUser = Depends(require_user),
) -> list[DocumentResponse]:
    """列出当前用户上传过的资料。"""
    records = await list_user_documents(current_user.id)
    return [DocumentResponse.model_validate(record) for record in records]


@document_router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: AuthenticatedUser = Depends(require_user),
) -> dict[str, bool]:
    """只允许资料所有者同步删除向量、原文件和登记记录。"""
    record = await get_owned_document(current_user.id, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="资料不存在")

    lease = await resources.obtain(("document", str(current_user.id), record.filename.casefold()))
    return await run_protected(lease, lambda: _delete_document(document_id, current_user))


async def _delete_document(
    document_id: UUID,
    current_user: AuthenticatedUser,
) -> dict[str, bool]:
    record = await get_owned_document(current_user.id, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="资料不存在")
    target = scoped_source(UPLOAD_DIRECTORY, current_user.id, record.filename)
    if Path(record.source_path).resolve() != target:
        raise HTTPException(409, "Unsafe document registration")
    async with journal.sessions() as session:
        shared_source = await source_is_registered(
            session, current_user.id, target, excluding=document_id
        )
    op = await journal.begin("delete", current_user.id, document_id, record.filename)
    try:
        await delete_document_assets(
            record,
            upload_root=UPLOAD_DIRECTORY / str(current_user.id),
            remove_source=not shared_source,
        )
    except DocumentDeletionError as exc:
        raise HTTPException(status_code=500, detail="资料删除未完成，可重试") from exc
    await journal.finish(op.id)
    logger.info("Personal document deleted")
    return {"deleted": True}
