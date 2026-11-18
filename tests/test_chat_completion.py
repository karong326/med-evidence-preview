"""Exercise the actual NDJSON finalization with injected Agent and persistence functions."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from med_evidence_agentic_rag.services.chat_stream import encode_chat_stream


def run_stream(events, *, error=None, save_error=None):
    complete = AsyncMock(side_effect=save_error)
    failed = AsyncMock()

    async def source(*args, **kwargs):
        for event in events:
            yield event
        if error:
            raise error

    async def collect():
        return [
            json.loads(line)
            async for line in encode_chat_stream(
                "test question",
                uuid4(),
                uuid4(),
                object(),
                "test question",
                uuid4(),
                1,
                stream_chat=source,
                complete_chat_turn=complete,
                mark_chat_turn_failed=failed,
            )
        ]

    return asyncio.run(collect()), complete, failed


def test_completed_answer_persists_evidence_before_done():
    evidence = {"content": "synthetic evidence", "source": "demo"}
    out, complete, failed = run_stream(
        [
            {"type": "answer", "content": "第一段"},
            {"type": "answer", "content": "第二段"},
            {"type": "evidence", "evidences": [evidence]},
            {"type": "done"},
        ]
    )
    assert out[-1]["type"] == "done"
    assert complete.await_args.args[1] == "第一段第二段"
    assert complete.await_args.kwargs["evidence_snapshot"] == [evidence]
    failed.assert_not_awaited()


@pytest.mark.parametrize(
    "error,code",
    [
        (TimeoutError("synthetic timeout"), "service_timeout"),
        (ConnectionError("synthetic network"), "service_network"),
    ],
)
def test_model_error_preserves_partial_answer(error, code):
    out, complete, failed = run_stream([{"type": "answer", "content": "partial"}], error=error)
    assert out[-1]["type"] == "error"
    assert out[-1]["code"] == code
    assert not any(e["type"] == "done" for e in out)
    assert failed.await_args.kwargs["answer"] == "partial"
    complete.assert_not_awaited()


def test_incomplete_stream_is_not_reported_as_success():
    out, complete, failed = run_stream([{"type": "answer", "content": "partial"}])
    assert out[-1]["code"] == "agent_incomplete"
    assert failed.await_args.kwargs["answer"] == "partial"
    complete.assert_not_awaited()


def test_failed_persistence_suppresses_success_event():
    out, complete, failed = run_stream([{"type": "done"}], save_error=RuntimeError("test storage"))
    assert out[-1]["code"] == "persistence_error"
    assert not any(e["type"] == "done" for e in out)
    failed.assert_awaited_once()
