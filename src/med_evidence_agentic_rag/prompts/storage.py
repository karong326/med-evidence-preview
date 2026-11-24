"""读取、更新并归档管理员可调整的三段 Agent 提示词。

当前版本保存在 ``prompt_logs/agent_prompts.json``；发生内容变更时，只把对应提示词的旧版本
按原始换行追加到独立 Markdown 历史文件。保存接口携带 ``updated_at`` 时会拒绝过期页面覆盖，
文件损坏或首次运行则安全退回为空覆盖项，由调用方采用源码默认提示词。
"""

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from med_evidence_agentic_rag.utils.config import DATA_ROOT

PROMPT_LOG_DIRECTORY = DATA_ROOT / "prompts"
PROMPT_FILE = PROMPT_LOG_DIRECTORY / "agent_prompts.json"
_LEGACY_PROMPT_FILE = Path(__file__).resolve().parent / "prompt_logs" / "agent_prompts.json"
PROMPT_HISTORY_FILES = {
    "system_prompt": PROMPT_LOG_DIRECTORY / "system_prompt_history.md",
    "query_rewrite_prompt": PROMPT_LOG_DIRECTORY / "query_rewrite_prompt_history.md",
    "todo_prompt": PROMPT_LOG_DIRECTORY / "todo_prompt_history.md",
}


class PromptVersionConflictError(RuntimeError):
    """管理员提交的提示词版本已过期。"""


def load_prompts() -> dict[str, str]:
    """读取已保存提示词；首次运行时返回空覆盖项。"""
    path = PROMPT_FILE
    if not path.is_file() and "DATA_ROOT" not in os.environ:
        path = _LEGACY_PROMPT_FILE
    if not path.is_file():
        return {
            "system_prompt": "",
            "query_rewrite_prompt": "",
            "todo_prompt": "",
            "updated_at": "",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "system_prompt": "",
            "query_rewrite_prompt": "",
            "todo_prompt": "",
            "updated_at": "",
        }
    payload = cast(dict[str, object], value) if isinstance(value, dict) else {}
    return {
        key: text if isinstance(text := payload.get(key), str) else ""
        for key in ("system_prompt", "query_rewrite_prompt", "todo_prompt", "updated_at")
    }


def _archive_prompt_changes(
    previous: Mapping[str, str],
    current: Mapping[str, str],
    changed_at: str,
) -> None:
    """只归档发生变化的旧提示词，并保留原始多行格式。"""
    PROMPT_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for prompt_name, history_path in PROMPT_HISTORY_FILES.items():
        previous_text = previous[prompt_name]
        if previous_text == current[prompt_name]:
            continue
        with history_path.open("a", encoding="utf-8") as history:
            history.write(
                f"## {changed_at}\n\n"
                "### 修改前版本\n\n"
                f"{previous_text.rstrip()}\n\n"
                "---\n\n"
            )


def save_prompts(
    system_prompt: str,
    query_rewrite_prompt: str,
    todo_prompt: str,
    *,
    previous_prompts: Mapping[str, str] | None = None,
    expected_updated_at: str | None = None,
) -> bool:
    """保存提示词，并拒绝覆盖管理员已过期页面中的版本。"""
    current = load_prompts()
    if expected_updated_at is not None and current["updated_at"] != expected_updated_at:
        raise PromptVersionConflictError("提示词已被其他管理员更新")
    previous = previous_prompts or current
    if (
        previous["system_prompt"] == system_prompt
        and previous["query_rewrite_prompt"] == query_rewrite_prompt
        and previous["todo_prompt"] == todo_prompt
    ):
        return False
    changed_at = datetime.now(UTC).isoformat()
    current = {
        "system_prompt": system_prompt,
        "query_rewrite_prompt": query_rewrite_prompt,
        "todo_prompt": todo_prompt,
    }
    _archive_prompt_changes(previous, current, changed_at)

    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(
        json.dumps(
            {
                "system_prompt": system_prompt,
                "query_rewrite_prompt": query_rewrite_prompt,
                "todo_prompt": todo_prompt,
                "updated_at": changed_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True
