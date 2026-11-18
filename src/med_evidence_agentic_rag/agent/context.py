"""为 Agent 构造受控的会话与用户上下文。

本模块不执行模型或数据库操作。它判断“重试”等追问意图，将最近已完成轮次的证据
按原始引用编号附入汇总/邮件请求，并补充用户明确保存的称呼、联系人和长期偏好。
同时生成不泄露正文的上下文加载摘要，供执行记录面板显示。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

HISTORY_MARKERS = ("整理", "汇总", "总结", "发送", "邮件")
RETRY_MARKERS = ("重试", "再试", "重新回答", "继续刚才", "刚才那个", "retry")


class UserContext(TypedDict):
    """向 Tool 提供当前用户和会话轮次，不暴露给模型填写。"""

    user_id: str
    turn_number: int
    conversation_id: NotRequired[str]


def build_user_context_trace(
    preference: str | None,
    long_term_memory: dict[str, object] | None,
) -> str:
    """生成执行记录可展示的用户上下文摘要，不回显个人资料正文。"""
    memory = long_term_memory or {}
    has_personal_info = isinstance(memory.get("display_name"), str) and bool(memory["display_name"])
    contacts = memory.get("contacts")
    aliases = list(cast(dict[str, object], contacts)) if isinstance(contacts, dict) else []
    personal_status = "已加载" if has_personal_info else "未加载"
    memory_status = "已加载" if preference else "未加载"
    contact_status = (
        f"{len(aliases)} 个（{'、'.join(str(alias) for alias in aliases)}）"
        if aliases
        else "未加载"
    )
    return f"个人信息：{personal_status}；长期记忆：{memory_status}；联系人：{contact_status}"


def is_retry_request(question: str) -> bool:
    """判断用户是否明确要求重新执行刚刚失败的问题。"""
    return any(marker in question.lower() for marker in RETRY_MARKERS)


def build_agent_input(
    question: str,
    turns: Sequence[object],
    preference: str | None = None,
    long_term_memory: dict[str, object] | None = None,
    failed_turn: object | None = None,
) -> str:
    """保留用户原意，并在需要时附上按轮次区分的历史原文证据。"""
    retry_failed_turn = failed_turn is not None and is_retry_request(question)
    evidence_lines: list[str] = []

    # 1. 跨轮证据整理：
    if any(marker in question for marker in HISTORY_MARKERS):
        for turn in turns:
            raw_snapshot = getattr(turn, "evidence_snapshot", None)
            if not isinstance(raw_snapshot, list):
                continue
            for raw_evidence in cast(list[object], raw_snapshot):
                if not isinstance(raw_evidence, dict):
                    continue
                evidence = cast(dict[str, object], raw_evidence)
                citation_id = str(evidence.get("citation_id", ""))
                title = str(evidence.get("title", "未知资料"))
                source = Path(str(evidence.get("source", "未知来源"))).name
                content = str(evidence.get("content", ""))
                evidence_lines.append(f"[{citation_id}] {title}｜{source}\n{content}")

    # 2. “重试”恢复：
    if retry_failed_turn:
        failed_question = str(getattr(failed_turn, "question", ""))
        failure_message = str(getattr(failed_turn, "failure_message", ""))
        agent_input = (
            f"用户当前请求：{question}\n"
            f"上一轮执行未完成。刚刚失败的原问题：{failed_question}\n"
            f"失败原因：{failure_message}\n"
            "请重新回答刚刚失败的原问题；不要把失败原因当作医学事实。"
        )
    elif evidence_lines:
        agent_input = (
            f"用户当前请求：{question}\n"
            "以下是最近会话中实际保存的历史证据。引用时必须原样保留"
            " [轮次-子问题.序号] 编号，例如 [01-A.1]：\n\n" + "\n\n".join(evidence_lines)
        )
    else:
        agent_input = question

    additions: list[str] = []
    # 3. 长期资料和偏好：
    if long_term_memory:
        display_name = long_term_memory.get("display_name")
        contacts = long_term_memory.get("contacts")
        memory_lines: list[str] = []
        if isinstance(display_name, str) and display_name:
            memory_lines.append(f"用户称呼：{display_name}")
        if isinstance(contacts, dict) and contacts:
            aliases = "、".join(str(alias) for alias in cast(dict[object, object], contacts))
            memory_lines.append(f"已保存联系人别名：{aliases}")
        if memory_lines:
            additions.append(
                "用户明确保存的长期资料：\n"
                + "\n".join(memory_lines)
                + "\n仅用于称呼和收件人别名理解；联系人邮箱由邮件 Tool 在服务端解析。"
            )

    if preference:
        additions.append(
            f"用户长期偏好与稳定资料：{preference}\n"
            "这些信息可用于个性化称呼、表达与建议；不得改变医学事实、证据要求或工具调用规则。"
        )
    if not additions:
        return agent_input
    if agent_input == question:
        agent_input = f"用户当前问题：{agent_input}"
    return f"{agent_input}\n\n" + "\n\n".join(additions)
