"""通过 MinerU 官方 SDK 解析 PDF、DOCX 等复杂本地文档。

模块将文件上传、轮询任务、下载 Markdown/图片结果和本地清理封装为同步加载器，并把解析内容
转为带页码/来源信息的 Document。调用前需要有效 MinerU token；普通文本格式不会走此远程流程。
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from mineru import MinerU  # pyright: ignore[reportMissingTypeStubs]
from pypdf import PdfReader, PdfWriter

from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _PreparedFile:
    """记录临时上传文件与原始文档、分段顺序的对应关系。"""

    source: Path  # 用户真正提供的原文件（一般情况下，普通文件的 source 和 upload 相同）
    upload: Path  # 实际提交给 MinerU 的文件
    part_number: int  # 如果文件过大被拆分，这是第几段


class MinerUClient:
    """把官方 MinerU SDK 的批量结果整理为“源文件 → Markdown”映射。"""

    # 单次任务的最大页数限制
    MAX_PDF_PAGES = 200
    # SDK 单次批量任务最长等待时间
    BATCH_TIMEOUT_SECONDS = 1800
    HTML_SUFFIXES = {".html", ".htm"}

    def __init__(self, token: str) -> None:
        """保存精准解析 API 所需的 Token。"""
        if not token:
            raise RuntimeError("MINERU_TOKEN 未配置")
        self._token = token

    def parse(self, file_path: str | Path) -> str:
        """1,解析一份本地文件并返回 Markdown。"""
        source = Path(file_path).expanduser().resolve()
        return self.parse_many([source])[source]

    def parse_many(self, file_paths: Sequence[str | Path]) -> dict[Path, str]:
        """2,批量解析本地文件；超长 PDF 自动拆成最多 200 页的临时文件。"""
        # 统一使用绝对路径，并在提交前一次性检查全部文件。
        sources = [Path(path).expanduser().resolve() for path in file_paths]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"文档不存在：{source}")

        # 创建临时目录，只存放超长 PDF 分段，流程结束后自动清理。
        with TemporaryDirectory(prefix="med_evidence_mineru_") as temporary_directory:
            temporary_root = Path(temporary_directory)
            # HTML 与其他文档使用不同模型，因此分成两个批次。
            grouped: dict[str, list[_PreparedFile]] = defaultdict(list)
            for source in sources:
                model = "html" if source.suffix.lower() in self.HTML_SUFFIXES else "vlm"
                grouped[model].extend(self._prepare_file(source, temporary_root))

            # 官方 SDK 负责上传、轮询和结果下载，本层只整理文件映射。
            markdown_parts: dict[Path, dict[int, str]] = defaultdict(dict)
            client = MinerU(token=self._token)
            client.set_source("med-evidence-agentic-rag")
            try:
                for model, prepared_files in grouped.items():
                    self._parse_batch(client, model, prepared_files, markdown_parts)
            finally:
                client.close()

        # 将同一 PDF 的多个解析分段按原始页序合并。
        markdown_by_path: dict[Path, str] = {}
        for source in sources:
            parts = markdown_parts.get(source, {})
            if not parts:
                raise RuntimeError(f"MinerU 未返回解析结果：{source.name}")
            markdown_by_path[source] = "\n\n".join(parts[number] for number in sorted(parts))

        return markdown_by_path

    def _prepare_file(self, source: Path, temporary_root: Path) -> list[_PreparedFile]:
        """普通文件直接返回，超长 PDF 则拆成最多 200 页的临时分段。"""
        if source.suffix.lower() != ".pdf":
            return [_PreparedFile(source=source, upload=source, part_number=1)]

        reader = PdfReader(str(source))
        page_count = len(reader.pages)

        if page_count <= self.MAX_PDF_PAGES:
            return [_PreparedFile(source=source, upload=source, part_number=1)]

        logger.info("Splitting PDF for MinerU: pages=%d", page_count)
        # 只拆分超限文件，避免 322 页原文被 MinerU 直接拒绝。
        prepared_files: list[_PreparedFile] = []
        for start in range(0, page_count, self.MAX_PDF_PAGES):
            part_number = len(prepared_files) + 1
            writer = PdfWriter()
            for page in reader.pages[start: start + self.MAX_PDF_PAGES]:
                writer.add_page(page)

            upload = temporary_root / f"{source.stem}_part_{part_number:03d}.pdf"
            with upload.open("wb") as file:
                writer.write(file)
            prepared_files.append(
                _PreparedFile(source=source, upload=upload, part_number=part_number)
            )

        logger.info("PDF split for MinerU: parts=%d", len(prepared_files))
        return prepared_files

    def _parse_batch(
        self,
        client: MinerU,
        model: str,
        prepared_files: list[_PreparedFile],
        markdown_parts: dict[Path, dict[int, str]],
    ) -> None:
        """提交一个模型批次，并把乱序结果写回对应原文及分段。"""
        # 以返回文件名定位源文档，不能依赖异步结果的返回顺序。
        by_filename = {prepared.upload.name: prepared for prepared in prepared_files}
        if len(by_filename) != len(prepared_files):
            raise ValueError("_parse_batch: 同一批次存在同名文件，无法可靠对应 MinerU 结果")

        upload_paths = [str(prepared.upload) for prepared in prepared_files]
        logger.info("MinerU batch started: files=%d", len(upload_paths))

        received: set[str] = set()
        # SDK 会在各文件完成时依次产出结果。
        results = client.extract_batch(
            upload_paths,
            model=model,
            timeout=self.BATCH_TIMEOUT_SECONDS,
        )
        for result in results:
            filename = Path(result.filename or "").name
            prepared = by_filename.get(filename)

            # （下面三个if语段要求）每个结果需要满足：
            # 返回文件名是已提交文件
            # state == "done"
            # markdown 非空
            if prepared is None:
                raise RuntimeError(
                    f"_parse_batch: MinerU 返回了未知文件：{filename or '未提供文件名'}"
                )
            received.add(filename)

            # 同时，对服务端明确提示“稍后重试”的临时错误单独重试一次。
            if result.state != "done" and "try again later" in (result.error or "").lower():
                logger.warning("MinerU temporary failure; retrying one part")
                result = client.extract(
                    str(prepared.upload),
                    model=model,
                    timeout=self.BATCH_TIMEOUT_SECONDS,
                )

            if result.state != "done":
                raise RuntimeError(
                    f"_parse_batch: MinerU 解析失败：文件={prepared.source.name}，"
                    f"原因={result.error or '未知原因'}"
                )
            if not result.markdown:
                raise RuntimeError(f"MinerU 解析结果为空：{prepared.source.name}")

            # 按原始文件和分段序号保存，稍后统一拼接。
            markdown_parts[prepared.source][prepared.part_number] = result.markdown
            logger.info("MinerU part parsed")

        missing = set(by_filename) - received
        # 只要有文件没有结果，整个批次就不算完整成功
        if missing:
            raise RuntimeError(f"_parse_batch: MinerU 未返回全部结果：{', '.join(sorted(missing))}")
