import asyncio
import json
import logging

import httpx
from fastapi import FastAPI


def test_readiness_expires_and_does_not_repeat_a_stalled_probe():
    from med_evidence_agentic_rag.services.health import Readiness

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def probe():
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()

        health = Readiness(interval=0.05, timeout=0.25)

        async def wait_for_status(status):
            while health.snapshot()["status"] != status:
                await asyncio.sleep(0.005)

        assert health.snapshot()["status"] == "starting"
        health.start({"postgresql": probe})
        try:
            await wait_for_status("ready")
            await entered.wait()
            await wait_for_status("unavailable")
            await asyncio.sleep(health.interval * 2)
            assert calls == 2
            release.set()
            await wait_for_status("ready")
        finally:
            release.set()
            await health.stop()
        assert health.snapshot()["status"] == "draining"

    asyncio.run(asyncio.wait_for(run(), 5))


def test_readiness_refuses_a_previous_success_when_samples_age(monkeypatch):
    from med_evidence_agentic_rag.services import health

    now = 100.0
    monkeypatch.setattr(health, "monotonic", lambda: now)

    async def run():
        nonlocal now
        monitor = health.Readiness(interval=30, max_age=15)

        async def probe():
            return None

        monitor.start({"redis": probe})
        try:
            async with asyncio.timeout(1):
                while monitor.snapshot()["status"] != "ready":
                    await asyncio.sleep(0)
            now += 16
            assert monitor.snapshot() == {
                "status": "unavailable", "services": {"redis": "unavailable"}
            }
        finally:
            await monitor.stop()

    asyncio.run(run())


def test_http_telemetry_uses_only_route_templates_and_safe_ids():
    from med_evidence_agentic_rag.services.telemetry import RequestTelemetry, Telemetry

    async def run():
        app = FastAPI()
        telemetry = Telemetry()
        app.add_middleware(RequestTelemetry, telemetry=telemetry)

        @app.get("/probe/{identity}")
        async def probe(identity: str):
            if identity == "private-error":
                raise ValueError("private SQL parameters")
            return {"ok": True}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            first = await client.get(
                "/probe/private-user?recipient=private-email",
                headers={"X-Request-ID": "unsafe@id.example"},
            )
            second = await client.get(
                "/probe/private-error", headers={"X-Request-ID": "synthetic-request"}
            )
            assert first.headers["x-request-id"] != "unsafe@id.example"
            assert second.headers["x-request-id"] == "synthetic-request"
            assert second.status_code == 500
            assert second.headers["x-frame-options"] == "DENY"
        metrics = telemetry.render()
        assert 'route="/probe/{identity}"' in metrics
        assert "private-" not in metrics
        assert "synthetic-request" not in metrics

    asyncio.run(run())


def test_json_logs_strip_untrusted_arguments_and_exception_payload():
    from med_evidence_agentic_rag.utils.logger import SafeJsonFormatter

    try:
        raise ValueError("private SQL or provider payload")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "httpx", logging.ERROR, __file__, 1, "request %s",
            ("http://private-url.example",), sys.exc_info(),
        )
    output = SafeJsonFormatter().format(record)
    entry = json.loads(output)
    assert "private" not in output
    assert entry["exception_type"] == "ValueError"
    assert entry["frames"][-1]["function"] == (
        "test_json_logs_strip_untrusted_arguments_and_exception_payload"
    )


def test_application_liveness_readiness_and_metrics_authentication(monkeypatch):
    from med_evidence_agentic_rag.api import app

    monkeypatch.setattr(app, "diagnostics_token", "isolated-diagnostics")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app.app), base_url="http://test"
        ) as client:
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503
            assert (await client.get("/api/health")).status_code == 200
            assert (await client.get("/metrics")).status_code == 401
            response = await client.get(
                "/metrics", headers={"Authorization": "Bearer isolated-diagnostics"}
            )
            assert response.status_code == 200
            assert "medevidence_dependency_ready" in response.text

    asyncio.run(run())


def test_compatibility_health_reports_successful_retrieval_prewarm(tmp_path, monkeypatch):
    import sys
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from starlette.datastructures import State

    from med_evidence_agentic_rag.api import app

    @asynccontextmanager
    async def open_checkpointer():
        yield SimpleNamespace(
            adelete_thread=AsyncMock(),
            conn=AsyncMock(spec=app.AsyncConnection),
            lock=asyncio.Lock(),
        )

    monkeypatch.setattr(app.app, "state", State())
    monkeypatch.setattr(app.local_guards, "root", tmp_path / "index-fence")
    monkeypatch.setattr(app, "create_application_tables", AsyncMock())
    monkeypatch.setattr(app, "open_checkpointer", open_checkpointer)
    monkeypatch.setattr(app, "redis_client", SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(app, "probe_postgresql", AsyncMock())
    monkeypatch.setattr(app, "probe_redis", AsyncMock())
    monkeypatch.setattr(app, "verify_llm_model", AsyncMock())
    monkeypatch.setattr(app, "prewarm_retrieval", lambda: 23)
    monkeypatch.setattr(app, "create_chat_agent", lambda _: object())
    monkeypatch.setattr(app, "recover", AsyncMock(return_value=dict.fromkeys(
        ("completed", "busy", "failed"), 0,
    )))
    monkeypatch.setattr(app.email_approvals, "recover_abandoned", AsyncMock(return_value=0))
    monkeypatch.setitem(
        sys.modules, "med_evidence_agentic_rag.models_and_database.chroma",
        SimpleNamespace(vectorstore=SimpleNamespace(get=lambda **_: None)),
    )

    async def run():
        async with app.lifespan(app.app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app.app), base_url="http://test"
        ) as client:
            compatibility = (await client.get("/api/health")).json()
            readiness = (await client.get("/readyz")).json()
            assert compatibility["services"]["retrieval_prewarm"] == "ready (23 chunks)"
            assert "retrieval_prewarm" not in readiness["services"]

    asyncio.run(run())


def test_question_length_limit():
    import pytest
    from pydantic import ValidationError

    from med_evidence_agentic_rag.api.chat import AskRequest

    identity = "c08ba761-cf39-4f6f-8251-a98a692791fd"
    assert AskRequest(question="问" * 16000, conversation_id=identity).question
    with pytest.raises(ValidationError):
        AskRequest(question="问" * 16001, conversation_id=identity)


def test_plaintext_smtp_capture_never_sends_login_credentials(monkeypatch):
    from unittest.mock import MagicMock

    from med_evidence_agentic_rag.tools import send_email as mail

    smtp = MagicMock()
    monkeypatch.setenv("MY_SENDER_EMAIL", "sender@example.org")
    monkeypatch.setenv("MY_SENDER_PASSWORD", "dummy-secret")
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_TRANSPORT", "plain")
    monkeypatch.setattr(mail.smtplib, "SMTP", smtp)
    monkeypatch.setattr(mail.smtplib, "SMTP_SSL", MagicMock(side_effect=AssertionError))
    assert mail.send_email("recipient@example.org", "summary", "synthetic") is True
    smtp.assert_called_once_with("127.0.0.1", 2525, timeout=15)
    smtp.return_value.login.assert_not_called()


def test_timed_out_retrieval_retains_the_stream_lease_until_worker_exits(tmp_path):
    from threading import Event

    import pytest
    from fastapi import HTTPException

    from med_evidence_agentic_rag.services.coordination import (
        LeasedStreamingResponse,
        ResourceRegistry,
        run_stream_worker,
    )

    entered, release = Event(), Event()

    def retrieve():
        entered.set()
        assert release.wait(5)

    async def run():
        locks = ResourceRegistry(tmp_path, None)
        key = ("conversation", "synthetic-owner", "synthetic-thread")
        lease = await locks.obtain(key)

        async def source():
            async with asyncio.timeout(0.02):
                await run_stream_worker(retrieve)
            yield "unexpected"

        response = LeasedStreamingResponse(source(), lease)
        task = asyncio.create_task(anext(response._iterate()))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(0.04)
            with pytest.raises(HTTPException) as busy:
                await locks.obtain(key)
            assert busy.value.status_code == 409
        finally:
            release.set()
            with pytest.raises(TimeoutError):
                await task
        await (await locks.obtain(key)).close()

    asyncio.run(asyncio.wait_for(run(), 2))
