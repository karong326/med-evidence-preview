"""限制知识库检索只能执行本轮查询规划已返回的条目。

``allowed_plan_queries`` 从当前 human 消息之后的最近 ``rewrite_queries`` ToolMessage 读取 JSON
计划，避免旧轮次计划影响当前问题。``QueryPlanGuardMiddleware`` 在调用 ``search_knowledge``
前核对 query；没有计划或检索词不匹配时返回带 ``status=error`` 的 ToolMessage，使 Agent 和
执行记录都能如实看到“已阻止”而不是静默跳过。
"""

import json
from collections.abc import Sequence
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage


def allowed_plan_queries(messages: Sequence[object]) -> set[str]:
    """读取当前用户回合中最近一次有效查询计划的原始查询列表。"""
    current_turn = list(messages)
    for index in range(len(current_turn) - 1, -1, -1):
        if getattr(current_turn[index], "type", "") == "human":
            current_turn = current_turn[index + 1 :]
            break
    for message in reversed(current_turn):
        if getattr(message, "name", None) != "rewrite_queries":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        typed_payload = cast(dict[str, object], payload) if isinstance(payload, dict) else {}
        queries = typed_payload.get("queries")
        if isinstance(queries, list):
            query_items = cast(list[object], queries)
            return {item.strip() for item in query_items if isinstance(item, str) and item.strip()}
    return set()


class QueryPlanGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """阻止 Agent 绕过本轮 QueryPlan 自行创造新的检索词。"""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Any:
        tool_call = request.tool_call
        if tool_call.get("name") != "search_knowledge":
            return await handler(request)
        args = tool_call.get("args")
        query = args.get("query")
        allowed = allowed_plan_queries(request.state.get("messages", []))
        if not allowed:
            message = "本轮没有可执行的查询计划，无法检索。"
        elif not isinstance(query, str) or query not in allowed:
            message = "该检索词不在本轮查询规划中，已阻止执行。"
        else:
            return await handler(request)
        return ToolMessage(
            content=message,
            name="search_knowledge",
            tool_call_id=tool_call["id"],
            status="error",
        )
