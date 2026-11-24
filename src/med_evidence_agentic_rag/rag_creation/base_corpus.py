"""发现并串行初始化 ``data/raw`` 中的共享基础语料。

模块筛选支持的本地文件，批量加载后逐份调用 ``index_document`` 写入 Chroma。串行处理便于在
启动、管理员管理和独立运行时记录明确的文件级成功/失败日志；共享资料使用 source_type=base，
与用户私有上传资料的所有者可见性隔离。
"""

from pathlib import Path

from med_evidence_agentic_rag.rag_creation.loader import DocumentLoader
from med_evidence_agentic_rag.utils.config import DATA_ROOT
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

BASE_CORPUS_DIRECTORY = DATA_ROOT / "raw"
SUPPORTED_SUFFIXES = DocumentLoader.MINERU_SUFFIXES | DocumentLoader.TEXT_SUFFIXES


def discover_base_documents(base_directory: str | Path = BASE_CORPUS_DIRECTORY) -> list[Path]:
    """发现和返回基础语料目录中的全部支持文件，并按路径稳定排序返回。"""

    root = Path(base_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"该基础语料目录路径不存在: {root}")

    # 递归发现支持格式，忽略目录中的其他文件。
    paths = (
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = sorted(paths, key=lambda path: str(path).lower())
    logger.info("Base documents discovered: count=%d", len(documents))
    return documents


def index_base_corpus(base_directory: str | Path = BASE_CORPUS_DIRECTORY) -> dict[str, int]:
    """顺序处理基础语料；单份文件失败不会中断后续文件。"""
    from med_evidence_agentic_rag.services.coordination import local_guards

    with local_guards.local_scope(("index-owner", "maintenance"), exclusive=True):
        return _index_base_corpus(base_directory)


def _index_base_corpus(base_directory: str | Path) -> dict[str, int]:
    from med_evidence_agentic_rag.models_and_database.chroma import vectorstore
    from med_evidence_agentic_rag.rag_creation.ingestion import index_document
    from med_evidence_agentic_rag.rag_retrieval.bm25 import invalidate_bm25_index

    documents = discover_base_documents(base_directory)

    # 先确保全部原文解析成功，再替换公共向量，避免失败时损失可用索引。
    # 上传资料有 source_type=upload，绝不能随着基础语料重建而被删除。
    loaded_documents = DocumentLoader.load_many(documents)
    vectorstore.delete(where={"source_type": "base"})
    invalidate_bm25_index()
    logger.info("旧基础向量已清空，开始写入完整基础语料")

    # 逐份切块入库并汇总结果，单份写入失败不会阻断后续文档。
    summary = {"discovered": len(documents), "indexed": 0, "chunks": 0, "failed": 0}
    for file_path in documents:
        try:
            chunk_count = index_document(
                file_path,
                source_type="base",
                documents=loaded_documents[file_path],
            )
        except Exception:
            summary["failed"] += 1
            logger.exception("Base document ingestion failed")
            continue

        summary["indexed"] += 1
        summary["chunks"] += chunk_count

    logger.info("Base corpus initialization finished")
    return summary


if __name__ == "__main__":
    index_base_corpus()
