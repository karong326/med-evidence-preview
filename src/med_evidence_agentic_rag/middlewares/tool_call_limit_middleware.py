"""限制 Agent 单轮工具调用次数，并本地化框架的阻止结果。

继承 LangChain ``ToolCallLimitMiddleware`` 的计数行为，在工具超限时保留 ``continue`` 流程，让
模型利用已有结果继续回答。当前规则固定为查询规划一次、知识库检索最多六次；子类仅把框架
错误 ToolMessage 改为中文，供 Agent 与执行记录面板共同展示。
"""

from typing import Any, cast

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import ToolMessage


# 硬性限制：
# rewrite_queries    每轮最多 1 次
# search_knowledge   每轮最多 5 次
class LocalizedToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """沿用框架限流计数，只把交给 Agent 的拦截结果改为中文。"""

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        update = super().after_model(state, runtime)
        if not isinstance(update, dict):
            return update
        messages = update.get("messages")
        if not isinstance(messages, list):
            return update
        result_messages = cast(list[object], messages)
        for message in result_messages:
            if isinstance(message, ToolMessage) and message.status == "error":
                message.content = (
                    f"本轮工具调用次数已达到上限：{self.tool_name}。请基于现有结果继续。"
                )
        return update


def build_tool_call_limit_middlewares() -> list[Any]:
    """限制查询改写一次、知识检索最多六次，避免 Agent 无意义循环。"""
    return [
        LocalizedToolCallLimitMiddleware(
            tool_name="rewrite_queries",
            run_limit=1,
            exit_behavior="continue",
        ),
        LocalizedToolCallLimitMiddleware(
            tool_name="search_knowledge",
            run_limit=6,
            exit_behavior="continue",
        ),
    ]
