"""
3. 用户上传资料状态：
维护用户上传资料在 PostgreSQL 中的登记与状态。

每份文件先创建 ``processing`` 记录，入库成功后记录 chunk 数并标为 ``ready``；删除和异常
补偿可标为 deleting/delete_failed/failed。文件本体与 Chroma 向量不在此处操作，由 API 和
document_deletion 服务按安全顺序协调。
"""

from uuid import UUID

from sqlalchemy import delete, select

from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.persistence.postgres_tables import DocumentRecord, utc_now


# 资料状态变化：
# processing
# ├── ready
# └── failed
#
# ready / failed
# → deleting
# ├── 删除成功：登记记录消失
# └── 删除失败：delete_failed
async def create_document(
    *, owner_user_id: UUID, filename: str, source_path: str
) -> DocumentRecord:
    """先登记待处理资料，使解析失败也有明确状态。"""
    async with AsyncSessionLocal() as session:
        record = DocumentRecord(
            owner_user_id=owner_user_id,
            filename=filename,
            source_path=source_path,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record


async def mark_document_ready(document_id: UUID, chunk_count: int) -> None:
    """入库成功后记录文本块数量和完成状态。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(DocumentRecord, document_id)
        if record is not None:
            record.status = "ready"
            record.chunk_count = chunk_count
            record.updated_at = utc_now()
            await session.commit()


async def mark_document_failed(document_id: UUID) -> None:
    """解析或入库失败时保留失败状态。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(DocumentRecord, document_id)
        if record is not None:
            record.status = "failed"
            record.updated_at = utc_now()
            await session.commit()


async def mark_document_deleting(document_id: UUID) -> bool:
    """记录删除开始，并返回资料是否仍存在。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(DocumentRecord, document_id)
        if record is None:
            return False
        record.status = "deleting"
        record.updated_at = utc_now()
        await session.commit()
        return True


async def mark_document_delete_failed(document_id: UUID) -> None:
    """保留未清完的资料登记，支持通过同一入口重试删除。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(DocumentRecord, document_id)
        if record is not None:
            record.status = "delete_failed"
            record.updated_at = utc_now()
            await session.commit()


async def list_user_documents(user_id: UUID) -> list[DocumentRecord]:
    """只列出当前用户自己的资料。"""
    async with AsyncSessionLocal() as session:
        result = await session.scalars(
            select(DocumentRecord)
            .where(DocumentRecord.owner_user_id == user_id)
            .order_by(DocumentRecord.created_at.desc())
        )
        return list(result)


async def get_owned_document(user_id: UUID, document_id: UUID) -> DocumentRecord | None:
    """同时校验资料编号和所属用户，阻止越权读取或删除。"""
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(DocumentRecord).where(
                DocumentRecord.id == document_id,
                DocumentRecord.owner_user_id == user_id,
            )
        )


async def delete_document_record(document_id: UUID) -> None:
    """删除已完成向量和文件清理的资料登记。"""
    async with AsyncSessionLocal() as session:
        await session.execute(delete(DocumentRecord).where(DocumentRecord.id == document_id))
        await session.commit()
