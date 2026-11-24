"""将本地文本或复杂文档统一加载为 LangChain Document。

``DocumentLoader`` 根据扩展名选择普通文本读取或 MinerU 解析，将页码、来源等信息规范到
Document metadata。它只负责文件到文本的转换和格式支持判断；分块、向量写入及用户资料状态
由后续创建阶段和 API 层负责。
"""

from collections.abc import Sequence
from pathlib import Path
from unicodedata import category

from langchain_core.documents import Document

from med_evidence_agentic_rag.rag_creation.mineru_loader import MinerUClient
from med_evidence_agentic_rag.utils.config import mineru_token
from med_evidence_agentic_rag.utils.logger import configure_server_logging, setup_logger

logger = setup_logger(__name__)


def _clean_content(content: str) -> str:
    """删除不可显示的控制字符，同时保留换行和制表符。"""
    return "".join(
        character for character in content if character in "\n\r\t" or category(character) != "Cc"
    )


class DocumentLoader:
    """按文件类型加载单个本地文档；复杂格式统一交给 MinerU。"""

    # 按格式选择文本加载或复杂文档解析。
    # 这里我们分为两个大类：
    MINERU_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".html", ".htm"}
    # 以下是直接加载然后即可使用的文本形式：
    TEXT_SUFFIXES = {".md", ".txt"}


    def __init__(self, file_path: str | Path) -> None:
        """保存规范化后的单文件绝对路径。"""
        self.file_path = Path(file_path).expanduser().resolve()

    def load(self) -> list[Document]:
        """加载单份文件并返回统一的 ``list[Document]``。"""
        if not self.file_path.is_file():
            raise FileNotFoundError(f"文档不存在: {self.file_path}")

        # 纯文本直接读取，复杂格式统一由 MinerU 转成 Markdown。
        suffix = self.file_path.suffix.lower()
        if suffix in self.TEXT_SUFFIXES:
            content = self.file_path.read_text(encoding="utf-8")

        elif suffix in self.MINERU_SUFFIXES:
            content = MinerUClient(mineru_token).parse(self.file_path)

        else:
            raise ValueError(f"暂不支持的文档格式: {suffix}")

        documents = [
            Document(page_content=_clean_content(content), metadata={"source": str(self.file_path)})
        ]
        logger.info("Document loaded: count=%d", len(documents))
        return documents

    @classmethod
    def load_many(cls, file_paths: Sequence[str | Path]) -> dict[Path, list[Document]]:
        """批量加载本地文档；复杂格式只提交一次 MinerU 批量解析。"""
        # 先规范化并分类文件，避免为每份复杂文档单独提交请求。
        sources = [Path(path).expanduser().resolve() for path in file_paths]
        contents: dict[Path, str] = {}
        mineru_sources: list[Path] = []

        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"文档不存在: {source}")
            suffix = source.suffix.lower()
            if suffix in cls.TEXT_SUFFIXES:
                contents[source] = source.read_text(encoding="utf-8")
            elif suffix in cls.MINERU_SUFFIXES:
                mineru_sources.append(source)
            else:
                raise ValueError(f"暂不支持的文档格式: {suffix}")

        # 所有复杂格式一次性交给 MinerU 批量解析。
        if mineru_sources:
            contents.update(MinerUClient(mineru_token).parse_many(mineru_sources))

        # 无论原始格式如何，统一返回“绝对路径 → Document 列表”。
        documents = {
            source: [
                Document(
                    page_content=_clean_content(contents[source]), metadata={"source": str(source)}
                )
            ]
            for source in sources
        }
        logger.info("文档批量加载完成：文件数=%d", len(documents))
        return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse an explicitly selected source document")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    configure_server_logging()
    try:
        DocumentLoader(args.source).load()
    except Exception:
        logger.exception("Document loader check failed")
        raise SystemExit(1) from None
