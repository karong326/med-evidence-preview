"""管理员维护共享基础资料库的路由。

基础资料位于唯一的 ``data/raw`` 目录，所有操作要求管理员 JWT。上传先执行与个人资料相同的
格式、大小和路径安全检查，再调用资料创建阶段写入 Chroma；删除按真实 source 路径清除向量后
才移除原文件。该路由不操作用户私有资料或其 PostgreSQL DocumentRecord。
"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_admin
from med_evidence_agentic_rag.api.documents import (
    SUPPORTED_SUFFIXES,
    UPLOAD_MAX_BYTES,
)
from med_evidence_agentic_rag.rag_creation.base_corpus import (
    BASE_CORPUS_DIRECTORY,
    discover_base_documents,
)
from med_evidence_agentic_rag.services.coordination import resources, run_protected
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
admin_documents_router = APIRouter(prefix="/api/admin", tags=["管理员"])


def _base_path(filename: str) -> Path:
    """将上传文件名限制在唯一基础资料目录内。"""
    path = (BASE_CORPUS_DIRECTORY / Path(filename).name).resolve()
    if path.parent != BASE_CORPUS_DIRECTORY.resolve():
        raise HTTPException(status_code=400, detail="资料文件名不安全")
    return path


@admin_documents_router.get("/base-documents")
async def list_base_documents(
    _: AuthenticatedUser = Depends(require_admin),
) -> list[dict[str, str]]:
    """列出唯一 data/raw 目录中的基础资料。"""
    return [{"filename": path.name, "source": str(path)} for path in discover_base_documents()]


@admin_documents_router.post("/base-documents")
async def upload_base_document(
    file: UploadFile = File(...),
    _: AuthenticatedUser = Depends(require_admin),
) -> dict[str, str | int]:
    """保存并立即走现有 MinerU、切块和 Chroma 入库流程。"""
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持该基础资料格式")
    target = _base_path(filename)
    lease = resources.acquire(("base-document", target.name.casefold()))
    return await run_protected(lease, lambda: _upload_base_document(file, target, filename))


async def _upload_base_document(
    file: UploadFile,
    target: Path,
    filename: str,
) -> dict[str, str | int]:
    if target.exists():
        raise HTTPException(status_code=409, detail="同名基础资料已存在")
    total = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > UPLOAD_MAX_BYTES:
                    output.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="基础资料超过 200MB")
                output.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="基础资料不能为空")
    try:
        from med_evidence_agentic_rag.rag_creation.ingestion import index_document

        chunks = await asyncio.to_thread(index_document, target, "base")
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="基础资料解析或入库失败") from exc
    logger.info("管理员基础资料入库完成：文件=%s，切块数=%d", filename, chunks)
    return {"filename": filename, "chunks": chunks}


@admin_documents_router.delete("/base-documents/{filename}")
async def delete_base_document(
    filename: str,
    _: AuthenticatedUser = Depends(require_admin),
) -> dict[str, bool]:
    """按真实 source 清理向量，再删除基础资料原文件。"""
    source = _base_path(filename)
    lease = resources.acquire(("base-document", source.name.casefold()))
    return await run_protected(lease, lambda: _delete_base_document(source))


async def _delete_base_document(source: Path) -> dict[str, bool]:
    if not source.is_file():
        raise HTTPException(status_code=404, detail="基础资料不存在")
    from med_evidence_agentic_rag.rag_creation.ingestion import delete_indexed_source

    await asyncio.to_thread(delete_indexed_source, str(source))
    source.unlink()
    logger.info("管理员基础资料已删除：文件=%s", source.name)
    return {"deleted": True}
