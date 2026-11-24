"""Offline regression checks for prompt activation and stream cleanup."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from med_evidence_agentic_rag.prompts import storage
from med_evidence_agentic_rag.services.chat_stream import encode_chat_stream


def factory_namespace():
    path = Path(__file__).parents[1] / "src/med_evidence_agentic_rag/agent/agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_chat_agent"
    )
    namespace = {
        "Any": object,
        "load_prompts": storage.load_prompts,
        "create_agent": lambda *args, **kwargs: kwargs,
        "llm": object(),
        "UserContext": object,
        "SYSTEM_PROMPT": "default system",
        "TODO_PLANNING_PROMPT": "default todo",
        "build_todo_list_middleware": lambda **kwargs: kwargs,
        "build_tool_call_limit_middlewares": lambda: [],
        "QueryPlanGuardMiddleware": object,
    }
    for name in (
        "rewrite_queries",
        "search_knowledge",
        "send_email_tool",
        "manage_user_preference",
        "manage_user_memory",
    ):
        namespace[name] = object()
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["create_chat_agent"]


def test_saved_prompts_activate_after_recreation_and_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "PROMPT_FILE", tmp_path / "prompts.json")
    monkeypatch.setattr(storage, "PROMPT_LOG_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage, "PROMPT_HISTORY_FILES", {})
    factory = factory_namespace()
    assert factory()["system_prompt"] == "default system"
    assert factory()["middleware"][0]["todo_prompt"] == "default todo"
    storage.save_prompts("saved system", "saved query", "saved todo")
    for create in (factory, factory_namespace()):
        agent = create()
        assert agent["system_prompt"] == "saved system"
        assert agent["middleware"][0]["todo_prompt"] == "saved todo"
    storage.save_prompts("new system", "new query", "new todo")
    assert factory()["system_prompt"] == "new system"


@pytest.mark.parametrize("payload", [None, [], {"system_prompt": None, "todo_prompt": 12}])
def test_corrupted_prompt_shape_uses_defaults(tmp_path, monkeypatch, payload):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(storage, "PROMPT_FILE", prompt_file)
    agent = factory_namespace()()
    assert agent["system_prompt"] == "default system"
    assert agent["middleware"][0]["todo_prompt"] == "default todo"


def build_stream(source, failed=None, complete=None):
    failed = failed or AsyncMock()
    complete = complete or AsyncMock()
    stream = encode_chat_stream(
        "question",
        uuid4(),
        uuid4(),
        object(),
        "question",
        uuid4(),
        1,
        stream_chat=source,
        complete_chat_turn=complete,
        mark_chat_turn_failed=failed,
    )
    return stream, failed, complete


@pytest.mark.parametrize("mode", ["close", "cancel", "incomplete"])
def test_interrupted_stream_preserves_partial_answer_and_evidence(mode):
    async def run():
        entered = asyncio.Event()

        async def source(*args, **kwargs):
            yield {"type": "answer", "content": "partial"}
            yield {"type": "evidence", "evidences": [{"content": "synthetic"}]}
            if mode == "cancel":
                entered.set()
                await asyncio.Event().wait()

        stream, failed, complete = build_stream(source)
        await anext(stream)
        await anext(stream)
        if mode == "close":
            await stream.aclose()
        elif mode == "cancel":
            task = asyncio.create_task(anext(stream))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            output = [json.loads(line) async for line in stream]
            assert output[-1]["code"] == "agent_incomplete"
            await stream.aclose()
        failed.assert_awaited_once()
        assert failed.await_args.kwargs["answer"] == "partial"
        assert failed.await_args.kwargs["evidence_snapshot"] == [{"content": "synthetic"}]
        complete.assert_not_awaited()

    asyncio.run(run())


def test_success_then_consumer_close_does_not_write_failure():
    async def run():
        closed = asyncio.Event()

        async def source(*args, **kwargs):
            try:
                yield {"type": "done"}
            finally:
                closed.set()

        stream, failed, complete = build_stream(source)
        assert json.loads(await anext(stream))["type"] == "done"
        await stream.aclose()
        assert closed.is_set()
        complete.assert_awaited_once()
        failed.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("event", [None, "error", "tool_blocked", "tool_complete"])
def test_provider_and_tool_diagnostics_are_never_returned_or_persisted(event):
    marker = "SYNTHETIC_SECRET_provider_payload"

    async def run():
        async def source(*args, **kwargs):
            if event == "error":
                yield {"type": "error", "code": marker, "message": marker}
            elif event:
                yield {
                    "type": "trace",
                    "trace": {
                        "kind": event,
                        "title": marker,
                        "detail": marker,
                        "step": 1,
                        "exception": {"raw": marker},
                    },
                    "debug": {"raw": marker},
                }
            raise RuntimeError(marker)

        stream, failed, _ = build_stream(source)
        output = [line async for line in stream]
        assert marker not in "".join(output)
        assert marker not in repr(failed.await_args)
        failed.assert_awaited_once()

    asyncio.run(run())


def test_cleanup_survives_repeated_consumer_cancellation():
    async def run():
        started = asyncio.Event()
        saved = asyncio.Event()
        release = asyncio.Event()

        async def persist(*args, **kwargs):
            started.set()
            await release.wait()
            saved.set()

        async def source(*args, **kwargs):
            yield {"type": "answer", "content": "partial"}

        failed = AsyncMock(side_effect=persist)
        stream, _, complete = build_stream(source, failed=failed)
        await anext(stream)
        task = asyncio.create_task(stream.aclose())
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert saved.is_set()
        failed.assert_awaited_once()
        complete.assert_not_awaited()

    asyncio.run(run())


def test_cleanup_timeout_cancels_and_joins_persistence(monkeypatch):
    from med_evidence_agentic_rag.services import chat_stream

    monkeypatch.setattr(chat_stream, "_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def run():
        stopped = asyncio.Event()

        async def persist(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def source(*args, **kwargs):
            yield {"type": "answer", "content": "partial"}

        stream, failed, _ = build_stream(source, failed=AsyncMock(side_effect=persist))
        await anext(stream)
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        assert stopped.is_set()
        failed.assert_awaited_once()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_cleanup_timeout_still_closes_provider(monkeypatch):
    from med_evidence_agentic_rag.services import chat_stream

    monkeypatch.setattr(chat_stream, "_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def run():
        closed = asyncio.Event()

        async def persist(*args, **kwargs):
            await asyncio.Event().wait()

        async def source(*args, **kwargs):
            try:
                yield {"type": "answer", "content": "partial"}
                await asyncio.Event().wait()
            finally:
                closed.set()

        stream, _, _ = build_stream(source, failed=AsyncMock(side_effect=persist))
        await anext(stream)
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        assert closed.is_set()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_turn_deadline_persists_partial_output_and_closes_provider(monkeypatch):
    from med_evidence_agentic_rag.services import chat_stream

    monkeypatch.setattr(chat_stream, "agent_turn_deadline_seconds", 0.02, raising=False)

    async def run():
        closed = asyncio.Event()

        async def source(*args, **kwargs):
            try:
                yield {"type": "answer", "content": "partial"}
                yield {"type": "evidence", "evidences": [{"key": "synthetic"}]}
                await asyncio.Event().wait()
            finally:
                closed.set()

        stream, failed, complete = build_stream(source)
        async with asyncio.timeout(0.5):
            events = [json.loads(line) async for line in stream]
        assert events[-1]["code"] == "service_timeout"
        assert failed.await_args.kwargs["answer"] == "partial"
        assert failed.await_args.kwargs["evidence_snapshot"] == [{"key": "synthetic"}]
        complete.assert_not_awaited()
        assert closed.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("blocked_event", ["answer", "done"])
def test_turn_deadline_cancels_blocked_asgi_send(monkeypatch, blocked_event):
    from threading import Event
    from types import SimpleNamespace

    from starlette.requests import ClientDisconnect, Request

    from med_evidence_agentic_rag.api import chat
    from med_evidence_agentic_rag.services.coordination import run_stream_worker

    async def run():
        owner, conversation_id, turn_id = uuid4(), uuid4(), uuid4()
        key = ("conversation", str(owner), str(conversation_id))
        send_entered, send_cancelled = asyncio.Event(), asyncio.Event()
        release_send, provider_closed = asyncio.Event(), asyncio.Event()
        failure_saved = asyncio.Event()
        worker_entered, release_worker = Event(), Event()
        worker = None

        def lease_is_active():
            return any(lease.key == key for lease in chat.resources.active)

        def sync_tool():
            worker_entered.set()
            assert release_worker.wait(3)

        async def source(*args, **kwargs):
            nonlocal worker
            try:
                if blocked_event == "answer":
                    worker = asyncio.create_task(run_stream_worker(sync_tool))
                    assert await asyncio.to_thread(worker_entered.wait, 1)
                yield {"type": "trace", "trace": {"kind": "question_received", "step": 1}}
                yield {"type": "evidence", "evidences": [{"id": "synthetic"}]}
                yield {"type": "answer", "content": "partial"}
                yield {"type": "done"}
            finally:
                assert lease_is_active()
                provider_closed.set()

        async def persist_failure(*args, **kwargs):
            assert lease_is_active()
            assert not provider_closed.is_set()
            failure_saved.set()

        failed = AsyncMock(side_effect=persist_failure)
        complete = AsyncMock()
        monkeypatch.setattr(chat, "agent_turn_deadline_seconds", 0.3, raising=False)
        monkeypatch.setattr(chat, "stream_chat", source)
        monkeypatch.setattr(chat, "complete_chat_turn", complete)
        monkeypatch.setattr(chat, "mark_chat_turn_failed", failed)
        monkeypatch.setattr(chat, "get_owned_conversation", AsyncMock(return_value=object()))
        monkeypatch.setattr(chat.journal, "conversation_deleting", AsyncMock(return_value=False))
        monkeypatch.setattr(chat, "list_recent_completed_turns", AsyncMock(return_value=[]))
        monkeypatch.setattr(chat, "get_user_personalization", AsyncMock(return_value=(None, None)))
        monkeypatch.setattr(chat, "create_chat_turn", AsyncMock(
            return_value=SimpleNamespace(id=turn_id, turn_number=1),
        ))
        scope = {
            "type": "http", "asgi": {"spec_version": "2.4"},
            "app": SimpleNamespace(state=SimpleNamespace()),
        }
        response = await chat.ask(
            chat.AskRequest(question="synthetic", conversation_id=conversation_id),
            Request(scope), chat.AuthenticatedUser(owner, "owner@example.org"),
        )

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                if json.loads(message["body"])["type"] == blocked_event:
                    send_entered.set()
                    try:
                        await release_send.wait()
                    except asyncio.CancelledError:
                        send_cancelled.set()
                        raise

        request = asyncio.create_task(response(scope, AsyncMock(), send))
        try:
            await asyncio.wait_for(send_entered.wait(), 1.0)
            await asyncio.wait_for(send_cancelled.wait(), 1.0)
            await asyncio.wait_for(provider_closed.wait(), 1.0)
            if blocked_event == "answer":
                assert failure_saved.is_set()
                failed.assert_awaited_once()
                saved = failed.await_args.kwargs
                assert saved["failure_code"] == "service_timeout"
                assert saved["answer"] == "partial"
                assert saved["evidence_snapshot"] == [{"id": "synthetic"}]
                assert [event["kind"] for event in saved["execution_trace"]] == [
                    "context_loaded", "question_received", "failure",
                ]
                assert lease_is_active()
                assert not request.done()
                complete.assert_not_awaited()
            else:
                complete.assert_awaited_once()
                assert complete.await_args.args == (turn_id, "partial")
                failed.assert_not_awaited()
            release_worker.set()
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(asyncio.shield(request), 1.0)
            assert not lease_is_active()
        finally:
            release_worker.set()
            release_send.set()
            await asyncio.gather(request, return_exceptions=True)
            if worker is not None:
                await worker

    asyncio.run(run())
