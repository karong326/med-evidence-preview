"""执行一份文档的加载、切分、Chroma 写入和按身份删除。

``index_document`` 可自行加载文件或复用已加载的 Document，生成稳定 document_id 和 chunk ID，
补齐 source、来源类型、标题与可选 owner metadata 后替换同路径旧 chunks。删除函数分别按
document_id（私有资料）或 source（基础资料）清理向量，并令 BM25 缓存下次重建。
"""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

from med_evidence_agentic_rag.models_and_database.chroma import vectorstore
from med_evidence_agentic_rag.rag_creation.loader import DocumentLoader
from med_evidence_agentic_rag.rag_creation.splitter import split_documents
from med_evidence_agentic_rag.rag_retrieval.bm25 import invalidate_bm25_index
from med_evidence_agentic_rag.utils.config import PROJECT_ROOT
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


def index_document(
    file_path: str | Path,
    source_type: str = "base",
    documents: Sequence[Document] | None = None,
    *,
    document_id: str | None = None,
    owner_user_id: str | None = None,
) -> int:
    """将一份文档写入 Chroma，并返回写入的 chunk 数。"""
    # 先删除该文件旧 chunks，再写入新 chunks：

    # 文件内容哈希作为稳定 document_id，同一内容可得到相同身份。
    source = Path(file_path).expanduser().resolve()
    # 上传资料显式传入身份，基础资料使用内容哈希。
    document_id = document_id or sha256(source.read_bytes()).hexdigest()
    # 既支持自行加载，也支持复用批量 Loader 已得到的 Document。
    loaded_documents = (
        DocumentLoader(source).load()
        if documents is None
        else list(documents)  # 复用批量解析，避免重复调用 MinerU。
    )
    # 文档名称直接取真实输入文件名，不再把正文中的章节标题误当成书名。
    title = source.stem
    # 获取文本块：
    chunks = split_documents(loaded_documents)

    # 为每个 chunk 补齐检索、引用和按文档删除所需的元数据。
    chunk_ids: list[str] = []
    for chunk in chunks:
        # 给每个 chunk 补齐以下元数据：
        chunk_id = f"{document_id}:{chunk.metadata['chunk_index']}"  # 文档编号:切块顺序
        chunk.metadata.update(
            {
                "source": str(source),  # 原文件绝对路径
                "source_type": source_type,  # base 或 upload
                "document_id": document_id,  # 文档编号
                "title": title,  # 原文件名(不含扩展名)
            }
        )
        if owner_user_id:
            # 用户资料额外获得:
            chunk.metadata["owner_user_id"] = owner_user_id
        chunk_ids.append(chunk_id)

    # 解析和切分成功后才清理同一路径的旧 chunks 然后准备替换，避免重复或陈旧数据。
    vectorstore.delete(where={"source": str(source)})
    # 一次性写入当前文档的全部 chunks。
    vectorstore.add_documents(chunks, ids=chunk_ids)

    # 使从 Chroma 构建的内存索引失效。
    invalidate_bm25_index()

    logger.info("文档入库完成：%s，切块数=%d", source.name, len(chunks))
    return len(chunks)


def delete_indexed_document(document_id: str) -> None:
    """按资料编号删除全部向量文本块，并让 BM25 下次自动重建。"""
    # 按 document_id 删除一份用户资料：
    vectorstore.delete(where={"document_id": document_id})
    invalidate_bm25_index()
    logger.info("‘用户资料向量’已删除：document_id=%s", document_id)


def delete_indexed_source(source: str) -> None:
    """按基础资料的真实路径删除全部向量文本块。"""
    # 按真实绝对路径 source 删除一份基础资料：
    vectorstore.delete(where={"source": source})
    invalidate_bm25_index()
    logger.info("‘基础资料向量’已删除：source=%s", source)


if __name__ == "__main__":
    sample = PROJECT_ROOT / "data" / "raw" / "who_patient_safety_fact_sheet.html"
    chunk_count = index_document(sample)
    logger.info("’ingestion‘功能实现的独立测试-成功：%s，切块数=%d", sample.name, chunk_count)
