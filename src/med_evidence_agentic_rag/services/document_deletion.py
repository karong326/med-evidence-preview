"""
4. 资料删除：
以可重试顺序清理一份用户资料的向量、文件与登记记录。

该服务先校验源文件严格位于所属 uploads 根目录，再将登记标记为 deleting，依次删除 Chroma
document_id 向量、原文件和 PostgreSQL 记录。任何中途错误都会把资料标记为 delete_failed 并
抛出统一异常；普通用户删除与管理员删除都复用此链，下一次请求可安全重试。
"""

import asyncio
from pathlib import Path

from med_evidence_agentic_rag.persistence.documents import (
    delete_document_record,
    mark_document_delete_failed,
    mark_document_deleting,
)
from med_evidence_agentic_rag.persistence.postgres_tables import DocumentRecord
from med_evidence_agentic_rag.rag_creation.ingestion import delete_indexed_document


# 删除顺序：
# 验证 source_path 位于用户上传目录
#   → 状态改为 deleting
#   → 删除 Chroma 向量
#   → 删除原文件
#   → 最后删除 PostgreSQL 登记
# 数据库记录最后删除，因为它是失败重试时定位其他资源的依据。
class DocumentDeletionError(RuntimeError):
    """资料清理未完成，但可由相同请求安全重试。"""


async def delete_document_assets(
    record: DocumentRecord, *, upload_root: Path, remove_source: bool = True
) -> None:
    """依次物理清理向量、源文件和 PostgreSQL 登记。"""
    source = Path(record.source_path).resolve()
    root = upload_root.resolve()
    if not source.is_relative_to(root):
        raise DocumentDeletionError("资料路径不安全")

    try:
        exists = await mark_document_deleting(record.id)
    except Exception as exc:
        raise DocumentDeletionError("资料删除状态登记失败") from exc
    if not exists:
        raise DocumentDeletionError("资料不存在")

    try:
        await asyncio.to_thread(delete_indexed_document, str(record.id))
        if remove_source:
            source.unlink(missing_ok=True)
        await delete_document_record(record.id)
    except Exception as exc:
        await mark_document_delete_failed(record.id)
        raise DocumentDeletionError("资料删除未完成，可重试") from exc
