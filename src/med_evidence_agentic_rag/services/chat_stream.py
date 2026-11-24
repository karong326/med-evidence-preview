"""Encode NDJSON and persist completed or interrupted turns with safe diagnostics."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from langgraph.errors import GraphRecursionError

from med_evidence_agentic_rag.services.telemetry import telemetry
from med_evidence_agentic_rag.utils.config import agent_turn_deadline_seconds
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
_CLEANUP_TIMEOUT_SECONDS = 2.0


@dataclass
class StreamDeadline:
    expires_at: float
    transport_expired: bool = False


def _failure_details(error: Exception) -> tuple[str, str]:
    """Map external failures to safe localized messages."""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return "llm_rate_limit", "大语言模型服务额度或请求频率受限，请稍后重试。"
    if status_code in {401, 403}:
        return "llm_auth", "大语言模型服务认证或配置异常，请联系管理员。"
    if isinstance(status_code, int) and status_code >= 500:
        return "llm_service", "大语言模型服务暂时不可用，请稍后重试。"
    if isinstance(error, TimeoutError):
        return "service_timeout", "服务响应超时，请稍后重试。"
    if isinstance(error, GraphRecursionError):
        return "agent_step_limit", "问答已达到执行步骤上限，请缩小问题范围后重试。"
    if isinstance(error, (ConnectionError, OSError)):
        return "service_network", "服务连接异常，请检查服务后重试。"
    return "agent_execution", "Agent、检索或工具执行失败，请稍后重试。"


def _safe_trace(raw: dict[str, object]) -> dict[str, object]:
    """Whitelist fields and replace untrusted tool-result diagnostics."""
    kind = raw.get("kind")
    ordinary = {
        "question_received",
        "context_loaded",
        "tool_call",
        "query_plan_complete",
        "knowledge_complete",
        "answer_started",
        "turn_completed",
    }
    if isinstance(kind, str) and kind in ordinary:
        return {
            key: value
            for key, value in raw.items()
            if key in {"step", "kind", "title", "detail", "created_at"}
            and isinstance(value, (str, int))
        }
    safe_kind = kind if kind in ("tool_complete", "tool_blocked") else "failure"
    title = {"tool_complete": "工具执行完成", "tool_blocked": "工具调用未完成"}.get(
        str(safe_kind), "执行失败"
    )
    return {
        "step": raw.get("step") if isinstance(raw.get("step"), int) else 0,
        "kind": safe_kind,
        "title": title,
        "detail": "工具执行完成" if safe_kind == "tool_complete" else "执行异常，请稍后重试。",
    }


async def _bounded_cleanup(operation: Awaitable[None]) -> None:
    """Own and join cleanup through cancellation, with a deadline.

    Persistence callbacks must cooperate with cancellation, as database drivers do.
    """
    task = asyncio.ensure_future(operation)
    deadline = asyncio.get_running_loop().time() + _CLEANUP_TIMEOUT_SECONDS
    try:
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.error("问答收尾超时")
                break
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.CancelledError:
                continue
            except TimeoutError:
                logger.error("问答收尾超时")
                break
    finally:
        if not task.done():
            task.cancel()
        # Retain ownership until cancellation has been acknowledged.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        if not task.cancelled():
            task.result()


async def encode_chat_stream(
    question: str,
    conversation_id: UUID,
    turn_id: UUID,
    chat_agent: Any,
    agent_input: str,
    user_id: UUID,
    turn_number: int,
    *,
    stream_chat: Callable[..., AsyncIterator[dict[str, object]]],
    complete_chat_turn: Callable[..., Awaitable[object]],
    mark_chat_turn_failed: Callable[..., Awaitable[object]],
    context_trace: str | None = None,
    deadline: StreamDeadline | None = None,
) -> AsyncGenerator[str, None]:
    """Persist before done; retain partial results when the stream is interrupted."""
    answers: list[str] = []
    evidences: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    terminal = False
    source: AsyncIterator[dict[str, object]] | None = None
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = StreamDeadline(loop.time() + agent_turn_deadline_seconds)

    def failure_events(code: str, message: str) -> list[dict[str, object]]:
        safe_message = f"{message}\n关联编号：{turn_id}"
        trace: dict[str, object] = {
            "step": max(
                (cast(int, t["step"]) for t in traces if isinstance(t.get("step"), int)),
                default=0,
            )
            + 1,
            "kind": "failure",
            "title": "执行失败",
            "detail": safe_message,
            "code": code,
            "turn_id": str(turn_id),
        }
        traces.append(trace)
        return [
            {"type": "trace", "trace": trace},
            {"type": "error", "code": code, "message": safe_message, "turn_id": str(turn_id)},
        ]

    async def save_failure(code: str, message: str) -> None:
        nonlocal terminal
        if terminal:
            return
        terminal = True
        outcome = {
            "service_timeout": "timeout",
            "agent_interrupted": "interrupted",
            "persistence_error": "persistence",
            "agent_incomplete": "incomplete",
        }.get(code, "execution")
        telemetry.turn_failures[outcome] += 1
        try:
            await mark_chat_turn_failed(
                turn_id,
                failure_code=code,
                failure_message=f"{message}\n关联编号：{turn_id}",
                answer="".join(answers) or None,
                evidence_snapshot=evidences or None,
                execution_trace=traces or None,
            )
        except Exception:
            logger.exception("Failed turn could not be saved")

    try:
        if context_trace:
            trace: dict[str, object] = {
                "step": 0,
                "kind": "context_loaded",
                "title": "已加载用户上下文",
                "detail": context_trace,
            }
            traces.append(trace)
            yield json.dumps({"type": "trace", "trace": trace}, ensure_ascii=False) + "\n"
        source = stream_chat(
            question,
            conversation_id,
            user_id=user_id,
            turn_number=turn_number,
            agent_input=agent_input,
            chat_agent=chat_agent,
        )
        while True:
            if loop.time() >= deadline.expires_at:
                raise TimeoutError
            async with asyncio.timeout_at(deadline.expires_at):
                event = await anext(source, None)
            if event is None:
                break
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError("Agent emitted an error event")
            if event_type == "answer":
                answers.append(str(event["content"]))
            if event_type == "evidence" and isinstance(event.get("evidences"), list):
                evidences.extend(
                    cast(dict[str, object], item)
                    for item in cast(list[object], event["evidences"])
                    if isinstance(item, dict)
                )
            if event_type == "trace":
                raw_trace = event.get("trace")
                if not isinstance(raw_trace, dict):
                    continue
                trace = _safe_trace(cast(dict[str, object], raw_trace))
                traces.append(trace)
                event = {"type": "trace", "trace": trace}
            if event_type == "done":
                try:
                    async with asyncio.timeout_at(deadline.expires_at):
                        await complete_chat_turn(
                            turn_id,
                            "".join(answers),
                            evidence_snapshot=evidences or None,
                            execution_trace=traces or None,
                        )
                except TimeoutError:
                    raise
                except Exception:
                    logger.exception("Completed turn could not be saved")
                    code, message = "persistence_error", "问答结果保存失败，请稍后重试。"
                    events = failure_events(code, message)
                    await _bounded_cleanup(save_failure(code, message))
                    for failure_event in events:
                        yield json.dumps(failure_event, ensure_ascii=False) + "\n"
                    return
                terminal = True
                yield json.dumps(event, ensure_ascii=False) + "\n"
                return
            yield json.dumps(event, ensure_ascii=False) + "\n"
        code, message = "agent_incomplete", "Agent 未返回完整结果，请稍后重试。"
        events = failure_events(code, message)
        await _bounded_cleanup(save_failure(code, message))
        for failure_event in events:
            yield json.dumps(failure_event, ensure_ascii=False) + "\n"
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as error:
        code, message = _failure_details(error)
        logger.exception("Agent turn failed")
        events = failure_events(code, message)
        await _bounded_cleanup(save_failure(code, message))
        for failure_event in events:
            yield json.dumps(failure_event, ensure_ascii=False) + "\n"
    finally:
        try:
            if not terminal:
                if deadline.transport_expired:
                    code, message = _failure_details(TimeoutError())
                else:
                    code, message = "agent_interrupted", "问答连接已中断，请重试。"
                failure_events(code, message)
                await _bounded_cleanup(save_failure(code, message))
        finally:

            async def close_source() -> None:
                close = getattr(source, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        logger.exception("Agent stream close failed")

            await _bounded_cleanup(close_source())
