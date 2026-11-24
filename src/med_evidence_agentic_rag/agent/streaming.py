"""将 LangGraph Agent 的原始事件转换为前端可消费的结构化流。

``stream_chat`` 接收会话、当前用户和已补充的 Agent 输入，按同一会话 thread 运行 Agent。
它只呈现可核对的事实：问题、工具调用及结果、查询计划、检索证据、回答片段和完成状态，
不输出模型隐式推理。代码同时处理工具参数分片、重复 ToolMessage、缺失工具名和中间件
阻止结果，最终产出 ``trace / evidence / answer / done`` 事件给服务层编码为 NDJSON。
"""

import ast
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, cast
from uuid import UUID

from langchain_core.messages import ToolMessage

from med_evidence_agentic_rag.rag_retrieval.models import RetrievalResult
from med_evidence_agentic_rag.tools.rewrite_queries import QueryPlan
from med_evidence_agentic_rag.utils.config import agent_recursion_limit
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger("med_evidence_agentic_rag.agent.agent")
_CITATION_PATTERN = re.compile(r"\s*\[\d{2}-[A-Z]\.\d+\]")
_TOOL_ACTION_LABELS = {
    "save_name": "保存用户称呼",
    "save_contact": "保存联系人",
    "save": "保存长期偏好",
    "clear": "清空长期偏好",
}


def _tool_result_text(content: object) -> str | None:
    """提取 ToolMessage 中承载本项目 JSON 结果的唯一文本内容。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    blocks = cast(list[object], content)

    if len(blocks) != 1:
        return None
    block = blocks[0]

    if not isinstance(block, dict):
        return None
    typed_block = cast(dict[str, object], block)

    if typed_block.get("type") != "text":
        return None
    text = typed_block.get("text")

    return text if isinstance(text, str) else None


def _todo_details(todos: object) -> str:
    """将 Todo 输入压缩为可展示的任务与状态，不保留模型隐式内容。"""
    if not isinstance(todos, list):
        return "任务清单已更新"
    status_labels = {"pending": "待处理", "in_progress": "进行中", "completed": "已完成"}
    details: list[str] = []
    todo_items = cast(list[object], todos)
    for item in todo_items:
        if not isinstance(item, dict):
            continue
        typed_item = cast(dict[str, object], item)
        content = typed_item.get("content")
        status = typed_item.get("status")
        if isinstance(content, str) and content:
            label = status_labels.get(status if isinstance(status, str) else "", "状态未知")
            details.append(f"{label}：{content}")
    return "；".join(details) or "任务清单已更新"


def _todo_details_from_result(content: str) -> str:
    """从 LangChain 标准 write_todos 结果恢复流中缺失的任务参数。"""
    prefix = "Updated todo list to "
    if not content.startswith(prefix):
        return "任务清单已更新"
    try:
        todos = ast.literal_eval(content.removeprefix(prefix))
    except (SyntaxError, ValueError):
        return "任务清单已更新"
    return _todo_details(todos)


def _safe_tool_input(tool_name: str, args: object) -> str:
    """返回可供用户核对的最小输入摘要，不泄露邮件正文或联系人邮箱。"""
    if not isinstance(args, dict):
        return f"原生名称：{tool_name}"
    typed_args = cast(dict[str, object], args)
    if tool_name == "write_todos":
        return f"原生名称：write_todos；输入：{_todo_details(typed_args.get('todos'))}"
    if tool_name == "send_email":
        recipient = typed_args.get("recipient")
        subject = typed_args.get("email_subject")
        parts = ["原生名称：send_email"]
        if isinstance(recipient, str) and recipient:
            recipient_label = "邮箱（已隐藏）" if "@" in recipient else recipient
            parts.append(f"收件人标识：{recipient_label}")
        if isinstance(subject, str) and subject:
            parts.append(f"主题：{subject}")
        return "；".join(parts)
    if tool_name in {"manage_user_preference", "manage_user_memory"}:
        action = typed_args.get("action")
        if isinstance(action, str) and action:
            label = _TOOL_ACTION_LABELS.get(action, action)
            return f"原生名称：{tool_name}；动作：{label}（{action}）"
        return f"原生名称：{tool_name}；动作：未提供"
    return f"原生名称：{tool_name}"


def _safe_tool_output(tool_name: str, content: str) -> str:
    """展示业务工具的成功状态，不回显敏感原始结果。"""
    if tool_name == "write_todos":
        return "任务清单已写入"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "已完成"
    if not isinstance(payload, dict):
        return "已完成"
    typed_payload = cast(dict[str, object], payload)
    success = typed_payload.get("success")
    action = typed_payload.get("action")
    message = typed_payload.get("message")
    parts = [f"成功={success}" if isinstance(success, bool) else "已完成"]
    if isinstance(action, str) and action:
        parts.append(f"动作={action}")
    if isinstance(message, str) and message:
        parts.append(message)
    return "；".join(parts)


async def stream_chat(
    question: str,
    conversation_id: UUID,
    *,
    user_id: UUID | None = None,
    turn_number: int = 1,
    agent_input: str | None = None,
    chat_agent: Any | None = None,
) -> AsyncIterator[dict[str, object]]:
    """恢复指定会话，先产出本轮原文证据，再流式产出最终回答。"""
    if not question.strip():
        raise ValueError("question 不能为空")

    if chat_agent is None:
        from med_evidence_agentic_rag.agent.agent import agent as default_agent

        active_agent = default_agent
    else:
        active_agent = chat_agent

    started_at = perf_counter()
    knowledge_called = False
    evidence_found = False
    called_tools: list[str] = []
    observed_tool_calls: set[str] = set()  # 同一 Tool 调用只记录一次
    traced_search_queries: set[str] = set()
    processed_tool_result_ids: set[str] = set()  # 同一 Tool 结果只解析一次
    tool_names_by_call_id: dict[str, str] = {}
    tool_args_by_call_id: dict[
        str, dict[str, object]
    ] = {}  # Tool 返回消息没有名称时，根据 tool_call_id 找回名称
    tool_chunk_ids_by_index: dict[int, str] = {}
    tool_chunk_names_by_index: dict[int, str] = {}
    tool_chunk_args_by_index: dict[int, str] = {}
    observed_rewrite_call_ids: set[str] = set()
    blocked_tools: list[str] = []
    answer_started = False
    answer_characters = 0
    trace_step = 0

    # 仅将已经收到的运行事件转换为前端可核对的轨迹，不生成模型隐式思维。
    def trace(kind: str, title: str, detail: str) -> dict[str, object]:
        """生成连续编号的可展示执行事实，不暴露模型内部推理。"""
        nonlocal trace_step
        trace_step += 1
        return {
            "type": "trace",
            "trace": {
                "step": trace_step,
                "kind": kind,
                "title": title,
                "detail": detail,
                "created_at": datetime.now(UTC).isoformat(),
            },
        }

    # 服务端日志只保留运行元数据；问题正文仍通过本轮受控执行轨迹返回给当前用户。
    logger.info("Agent turn started")
    yield trace("question_received", "接收问题", question)
    input_question = agent_input or question

    # ==========================================================================================
    async for event in active_agent.astream(
        {"messages": [{"role": "user", "content": input_question}]},
        config={
            "configurable": {"thread_id": str(conversation_id)},
            "recursion_limit": agent_recursion_limit,
        },
        context={
            "user_id": str(user_id) if user_id is not None else "",
            "turn_number": turn_number,
            "conversation_id": str(conversation_id),
        },
        stream_mode="messages",
        version="v2",
    ):
        if event.get("type") != "messages":
            continue
        raw_data: object = event.get("data")
        if not isinstance(raw_data, tuple):
            continue
        message_data = cast(tuple[object, ...], raw_data)
        if len(message_data) < 2:
            continue
        metadata: object = message_data[1]
        if not isinstance(metadata, dict):
            continue
        typed_metadata = cast(dict[str, object], metadata)
        message = message_data[0]
        content = getattr(message, "content", "")
        tool_name = getattr(message, "name", None)

        # 模型节点的 tool_calls 是 Agent 实际请求调用的工具和参数。
        if typed_metadata.get("langgraph_node") == "model":
            raw_tool_calls = getattr(message, "tool_calls", [])
            if isinstance(raw_tool_calls, list):
                tool_calls = list(cast(list[object], raw_tool_calls))
                raw_tool_call_chunks = getattr(message, "tool_call_chunks", [])
                if isinstance(raw_tool_call_chunks, list):
                    for raw_chunk in cast(list[object], raw_tool_call_chunks):
                        if not isinstance(raw_chunk, dict):
                            continue
                        chunk = cast(dict[str, object], raw_chunk)
                        index = chunk.get("index")
                        if not isinstance(index, int):
                            continue
                        chunk_id = chunk.get("id")
                        if isinstance(chunk_id, str) and chunk_id:
                            tool_chunk_ids_by_index[index] = chunk_id
                        chunk_name = chunk.get("name")
                        if isinstance(chunk_name, str) and chunk_name:
                            tool_chunk_names_by_index[index] = chunk_name
                        chunk_args = chunk.get("args")
                        if isinstance(chunk_args, str):
                            tool_chunk_args_by_index[index] = (
                                tool_chunk_args_by_index.get(index, "") + chunk_args
                            )
                        try:
                            parsed_args = json.loads(tool_chunk_args_by_index.get(index, ""))
                        except json.JSONDecodeError:
                            continue
                        name = tool_chunk_names_by_index.get(index)
                        call_id = tool_chunk_ids_by_index.get(index)
                        if name and call_id and isinstance(parsed_args, dict):
                            typed_parsed_args = cast(dict[str, object], parsed_args)
                            tool_calls.append(
                                {"id": call_id, "name": name, "args": typed_parsed_args}
                            )
                for raw_call in tool_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    typed_call = cast(dict[str, object], raw_call)
                    name = typed_call.get("name")
                    args = typed_call.get("args")
                    if not isinstance(name, str) or not name:
                        continue
                    key = typed_call.get("id")
                    if isinstance(key, str) and key:
                        tool_names_by_call_id[key] = name
                        if isinstance(args, dict):
                            tool_args_by_call_id[key] = cast(dict[str, object], args)
                    else:
                        key = json.dumps(
                            {"name": name, "args": args},
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )
                    if key in observed_tool_calls:
                        continue
                    if name in {"manage_user_preference", "manage_user_memory"} and (
                        not isinstance(args, dict)
                        or not isinstance(cast(dict[str, object], args).get("action"), str)
                    ):
                        # 工具参数仍在流式拼接中；等 action 完整后再生成真实调用记录。
                        continue
                    if name == "rewrite_queries" and isinstance(args, dict):
                        typed_args = cast(dict[str, object], args)
                        rewrite_question = typed_args.get("question")
                        if isinstance(rewrite_question, str) and rewrite_question:
                            observed_tool_calls.add(key)
                            observed_rewrite_call_ids.add(key)
                            yield trace(
                                "tool_call",
                                "调用查询规划工具",
                                f"原生名称：rewrite_queries；待规划问题：{rewrite_question}",
                            )
                        continue
                    if name == "search_knowledge" and isinstance(args, dict):
                        typed_args = cast(dict[str, object], args)
                        query = typed_args.get("query")
                        if isinstance(query, str) and query:
                            observed_tool_calls.add(key)
                            traced_search_queries.add(query)
                            yield trace(
                                "tool_call",
                                "调用知识库工具",
                                f"原生名称：search_knowledge；检索问题：{query}",
                            )
                        continue
                    if name == "write_todos":
                        if not isinstance(args, dict):
                            continue
                        typed_args = cast(dict[str, object], args)
                        if not isinstance(typed_args.get("todos"), list):
                            continue
                    observed_tool_calls.add(key)
                    yield trace("tool_call", "调用工具", _safe_tool_input(name, cast(object, args)))

        # ToolMessage 既可能来自 tools 节点，也可能来自中间件的限流拦截。
        # 工具内部 LLM 的 AIMessage 不进入这里，因此不会混淆为工具结果。
        is_tool_result = isinstance(message, ToolMessage) or (
            typed_metadata.get("langgraph_node") == "tools"
            and not getattr(message, "tool_calls", [])
        )

        if is_tool_result:
            content = _tool_result_text(content)
            if content is None:
                logger.warning("[Agent 进度] 工具返回内容不是可解析的单一文本结果")
                continue
            tool_call_id = getattr(message, "tool_call_id", None)
            if isinstance(tool_call_id, str) and tool_call_id:
                if tool_call_id in processed_tool_result_ids:
                    continue
                processed_tool_result_ids.add(tool_call_id)
            if not isinstance(tool_name, str):
                if isinstance(tool_call_id, str):
                    tool_name = tool_names_by_call_id.get(tool_call_id)
            if not isinstance(tool_name, str):
                logger.info("[Agent 工具] 未识别工具执行完成")
                yield trace("tool_complete", "工具执行完成", "工具身份未识别；结果未解析")
                continue
            if tool_name in {"manage_user_preference", "manage_user_memory"}:
                call_key = tool_call_id if isinstance(tool_call_id, str) else None
                if call_key and call_key not in observed_tool_calls:
                    args = dict(tool_args_by_call_id.get(call_key, {}))
                    try:
                        payload = json.loads(content)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        typed_payload = cast(dict[str, object], payload)
                        if isinstance(typed_payload.get("action"), str):
                            args["action"] = typed_payload["action"]
                    observed_tool_calls.add(call_key)
                    yield trace("tool_call", "调用工具", _safe_tool_input(tool_name, args))
            status = getattr(message, "status", "success")
            if status == "error":
                blocked_tools.append(tool_name)
                args = (
                    tool_args_by_call_id.get(tool_call_id, {})
                    if isinstance(tool_call_id, str)
                    else {}
                )
                detail = _safe_tool_input(tool_name, args)
                if tool_name == "search_knowledge" and isinstance(args.get("query"), str):
                    detail += f"；检索问题：{args['query']}"
                yield trace("tool_blocked", "工具调用已阻止", f"{detail}；原因：{content}")
                continue
            called_tools.append(tool_name)
            if tool_name == "rewrite_queries":
                try:
                    plan = QueryPlan.model_validate_json(content)
                except ValueError:
                    logger.warning("[Agent 进度] 查询规划工具返回内容无法解析")
                    yield trace(
                        "tool_complete", "工具执行完成", "工具名称：rewrite_queries；结果无法解析"
                    )
                    continue
                if not observed_rewrite_call_ids:
                    yield trace(
                        "tool_call",
                        "调用查询规划工具",
                        f"原生名称：rewrite_queries；待规划问题：{question}",
                    )
                query_details = "；".join(
                    f"{chr(ord('A') + index)}：{query}" for index, query in enumerate(plan.queries)
                )
                logger.info(
                    "Query planning completed: queries=%d fallback=%s",
                    len(plan.queries),
                    plan.fallback,
                )
                yield trace(
                    "query_plan_complete",
                    "查询规划完成",
                    f"原生名称：rewrite_queries；模式={plan.mode}；查询数={len(plan.queries)}；"
                    f"是否回退={plan.fallback}；{query_details}",
                )
                continue
            if tool_name == "write_todos":
                args = (
                    tool_args_by_call_id.get(tool_call_id, {})
                    if isinstance(tool_call_id, str)
                    else {}
                )
                logger.info("[Agent 工具] write_todos 执行完成")
                todo_detail = _todo_details(args.get("todos"))
                if todo_detail == "任务清单已更新":
                    todo_detail = _todo_details_from_result(content)
                yield trace(
                    "tool_complete",
                    "任务清单已更新",
                    f"原生名称：write_todos；输出：{todo_detail}",
                )
                continue
            if tool_name and tool_name != "search_knowledge":
                logger.info("Agent tool completed")
                yield trace(
                    "tool_complete",
                    "工具执行完成",
                    f"原生名称：{tool_name}；输出：{_safe_tool_output(tool_name, content)}",
                )
                continue
            try:
                result = RetrievalResult.model_validate_json(content)
            except ValueError:
                logger.warning("[Agent 进度] 知识库工具返回内容无法解析")
                continue
            knowledge_called = True
            evidence_found = result.found
            if result.query not in traced_search_queries:
                traced_search_queries.add(result.query)
                yield trace(
                    "tool_call",
                    "调用知识库工具",
                    f"原生名称：search_knowledge；检索问题：{result.query}",
                )
            logger.info("Knowledge tool completed: evidence=%d", len(result.evidences))
            citations = "、".join(item.citation_id for item in result.evidences) or "无"
            yield trace(
                "knowledge_complete",
                "知识库检索完成",
                f"原生名称：search_knowledge；检索问题：{result.query}；"
                f"命中 {len(result.evidences)} 条证据（{citations}）",
            )
            evidence_event: dict[str, object] = {
                "type": "evidence",
                "query": result.query,
                "evidences": [item.model_dump(exclude_none=True) for item in result.evidences],
            }
            yield evidence_event
            continue

        # 只监听模型节点的消息事件，过滤 Tool 原始结果和 Agent 内部状态。
        if typed_metadata.get("langgraph_node") != "model":
            continue
        if isinstance(content, str) and content:
            if not knowledge_called and any(
                name in called_tools
                for name in ("send_email", "manage_user_preference", "manage_user_memory")
            ):
                content = _CITATION_PATTERN.sub("", content)
            if not answer_started:
                answer_started = True
                logger.info("[回答生成] 开始流式生成模型回答")
                yield trace("answer_started", "模型开始输出", "已收到首个模型文本片段")
            answer_characters += len(content)
            yield {"type": "answer", "content": content}

    if not knowledge_called:
        other_tools = [name for name in dict.fromkeys(called_tools) if name != "search_knowledge"]
        if other_tools:
            logger.info("Agent completed without knowledge retrieval; other tools ran")
        else:
            logger.info("[Agent 状态] 本轮未调用工具，由模型直接回答")
    elapsed_seconds = perf_counter() - started_at
    tool_summary = "、".join(dict.fromkeys(called_tools)) or "无"
    blocked_summary = "、".join(dict.fromkeys(blocked_tools))
    blocked_detail = f"；已阻止：{blocked_summary}" if blocked_summary else ""
    yield trace(
        "turn_completed",
        "本轮问答完成",
        f"实际工具：{tool_summary}{blocked_detail}；回答字符数：{answer_characters}；"
        f"耗时：{elapsed_seconds:.2f} 秒",
    )
    yield {"type": "done"}
    logger.info(
        "[问答完成] 本轮调用知识库=%s，命中证据=%s，回答字符数=%d，耗时=%.2f秒",
        knowledge_called,
        evidence_found,
        answer_characters,
        elapsed_seconds,
    )
