import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

from med_evidence_agentic_rag.evaluation.adapters import RetrievalAdapter
from med_evidence_agentic_rag.evaluation.metrics import score_case
from med_evidence_agentic_rag.evaluation.schemas import Case, Observation
from med_evidence_agentic_rag.services.coordination import local_guards


def test_effect_receipts_are_independent_from_attempts():
    case = Case(
        id="case",
        query="private",
        forbidden_tools=["send_email"],
        expected_effects={"smtp_acceptances": 0},
    )
    for fields, attempt, effect in [
        ({}, None, None),
        ({"tools": ["send_email"]}, 0, None),
        ({"effect_counts": {"smtp_acceptances": None}}, None, None),
        ({"tools": [], "effect_counts": {"smtp_acceptances": 1}}, 1, 0),
        ({"effect_counts": {"smtp_acceptances": 0}}, None, 1),
    ]:
        scores = score_case(case, Observation(id=case.id, **fields), [1])
        assert scores["forbidden_tool_compliance"] == attempt
        assert scores["side_effect_accuracy"] == effect


def test_contention_precedes_backend_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(local_guards, "root", tmp_path)
    adapter = RetrievalAdapter(allow_live=True, user_id=str(uuid4()))
    monkeypatch.setattr(adapter, "_retrieve", lambda *_: pytest.fail("backend entered"))

    async def scenario():
        lease = await local_guards.obtain(("index-owner", "maintenance"))
        try:
            with pytest.raises(HTTPException, match="409"):
                await adapter.execute(Case(id="case", query="probe"))
        finally:
            await lease.close()

    asyncio.run(scenario())


def test_run_holds_one_index_lease_until_cancelled_workers_exit(monkeypatch, tmp_path):
    from threading import Event

    from med_evidence_agentic_rag.evaluation.runner import run
    from med_evidence_agentic_rag.evaluation.schemas import RunConfig
    from med_evidence_agentic_rag.rag_retrieval.models import RetrievalResult

    monkeypatch.setattr(local_guards, "root", tmp_path)
    adapter = RetrievalAdapter(allow_live=True, user_id=str(uuid4()))
    entered, finish = Event(), Event()

    def retrieve(query):
        entered.set()
        assert finish.wait(10)
        return RetrievalResult(query=query, found=False, evidences=[]), 1

    monkeypatch.setattr(adapter, "_retrieve", retrieve)
    cases = [Case(id=str(i), query="probe") for i in range(2)]

    async def scenario():
        config = RunConfig(concurrency=2)
        task = asyncio.create_task(run(cases, adapter, config))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            with pytest.raises(HTTPException, match="409"):
                await local_guards.obtain(("index-owner", "maintenance"))
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        report = await run(cases, adapter, config)
        assert report["aggregate"]["error_rate"] == 0

    asyncio.run(scenario())


def test_load_target_and_opt_in_checked_before_network():
    from med_evidence_agentic_rag.evaluation.load import LoadConfig

    for target, live in [
        ("https://example.org", False),
        ("http://example.org", True),
        ("https://name:token@example.org", True),
        ("", True),
    ]:
        with pytest.raises(ValueError):
            LoadConfig(target=target, allow_live=live)


@pytest.mark.parametrize(
    "traces,terminal,compliance,error",
    [
        ([], "done", None, None),
        ([{"kind": "question_received"}], "done", 1, None),
        ([{"kind": "tool_call", "detail": "调用工具"}], "done", None, None),
        ([{"kind": "tool_call", "detail": "原生名称：send_email"}], "done", 0, None),
        ([{"kind": "question_received"}], None, None, "incomplete"),
        ([{"kind": "question_received"}], "error", None, "agent"),
    ],
)
def test_stream_requires_observable_tool_attempts_to_score_compliance(
    traces, terminal, compliance, error
):
    import json

    import httpx

    from med_evidence_agentic_rag.evaluation.adapters import HTTPAgentAdapter
    from med_evidence_agentic_rag.evaluation.runner import gate, run
    from med_evidence_agentic_rag.evaluation.schemas import RunConfig

    def respond(request):
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path == "/api/conversations":
            return httpx.Response(201, json={"id": str(uuid4())})
        events = [{"type": "trace", "trace": trace} for trace in traces]
        if terminal is not None:
            events.append({"type": terminal})
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = HTTPAgentAdapter("https://example.org", "test", allow_live=True,
                                       client=client)
            case = Case(id="one", query="probe", forbidden_tools=["send_email"])
            report = await run([case], adapter, RunConfig())
            assert report["cases"][0]["error"] == error
            assert report["cases"][0]["metrics"]["forbidden_tool_compliance"] == compliance
            assert report["aggregate"]["forbidden_tool_compliance"] == compliance
            expected_failures = [] if compliance == 1 else ["forbidden_tool_compliance"]
            assert gate(report, {"forbidden_tool_compliance": 1}, {"error_rate": 1}) == (
                expected_failures
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_ask", [False, True])
@pytest.mark.parametrize("conflict_resolves", [False, True])
def test_load_reports_unresolved_cleanup_after_retry(monkeypatch, cancel_ask, conflict_resolves):
    import httpx

    from med_evidence_agentic_rag.evaluation import adapters
    from med_evidence_agentic_rag.evaluation.load import LoadConfig, measure

    monkeypatch.setattr(adapters, "_CLEANUP_TIMEOUT_SECONDS", 0.35)
    conversation_id = str(uuid4())
    remaining = set()
    attempts = 0

    async def handler(request):
        nonlocal attempts
        if request.method == "DELETE":
            assert request.url.path == "/api/conversations/" + conversation_id
            attempts += 1
            if attempts == 1 or not conflict_resolves:
                return httpx.Response(409)
            remaining.remove(conversation_id)
            return httpx.Response(204)
        if request.url.path == "/api/conversations":
            remaining.add(conversation_id)
            return httpx.Response(201, json={"id": conversation_id})
        if cancel_ask:
            await asyncio.sleep(5)
            pytest.fail("ask was not cancelled")
        return httpx.Response(
            200, text='{"type":"answer","content":"probe"}\n{"type":"done"}\n'
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await measure(
                LoadConfig(target="https://example.org", allow_live=True, requests=1,
                           duration=0.1 if cancel_ask else 5, max_error_rate=1),
                "test-token", "probe", client,
            )

    report = asyncio.run(scenario())
    errors = ({"timeout": 1} if cancel_ask else {}) if conflict_resolves else {"cleanup": 1}
    assert report["errors"] == errors
    assert report["passed"] is conflict_resolves
    assert attempts >= 2
    assert remaining == (set() if conflict_resolves else {conversation_id})


def test_load_report_omits_payload_and_credential():
    import json

    import httpx

    from med_evidence_agentic_rag.evaluation.load import LoadConfig, measure

    seen = []

    def handler(request):
        seen.append(request.method)
        if request.url.path.endswith("conversations"):
            return httpx.Response(200, json={"id": str(uuid4())})
        return httpx.Response(
            200, text='{"type":"answer","content":"private-answer"}\n{"type":"done"}\n'
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await measure(
                LoadConfig(target="https://example.org", allow_live=True, requests=2),
                "private-token",
                "private-prompt",
                client,
            )

    report = asyncio.run(scenario())
    assert report["passed"] and report["completed"] == 2
    assert seen.count("DELETE") == 2
    assert "private-" not in json.dumps(report)


@pytest.mark.parametrize("service", ["model", "parser", "smtp"])
def test_live_probe_needs_opt_in_and_target(service):
    import runpy
    from pathlib import Path

    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "verify_live.py"))["main"]
    for args in ([service, "--allow-live"], [service, "--target", "example.org"]):
        with pytest.raises(SystemExit) as stopped:
            main(args)
        assert stopped.value.code == 2
    if service == "smtp":
        with pytest.raises(SystemExit) as stopped:
            main([service, "--target", "example.org", "--allow-live"])
        assert stopped.value.code == 2


def test_cancel_during_conversation_creation_joins_and_deletes():
    import httpx

    from med_evidence_agentic_rag.evaluation.adapters import HTTPAgentAdapter

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        chat = str(uuid4())
        deleted = []

        async def handler(request):
            if request.method == "POST":
                started.set()
                await release.wait()
                return httpx.Response(201, json={"id": chat})
            deleted.append(request.url.path)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = HTTPAgentAdapter(
                "https://example.org", "test", allow_live=True, client=client
            )
            task = asyncio.create_task(adapter.execute(Case(id="test", query="private")))
            await started.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert deleted == ["/api/conversations/" + chat]

    asyncio.run(scenario())
