"""Child process of test_synthetic_deployment; all service imports follow fixture setup."""

import asyncio
import json
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import uvicorn
from sqlalchemy import text
from sqlalchemy.engine import make_url
from verification_providers import MAIL, PASSAGE, QUERY, Capture, install_inference, model_endpoint


@asynccontextmanager
async def serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(60):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("fixture startup failed")
                    await asyncio.sleep(0.05)
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            server.should_exit = True
            await task


async def exercise(client, capture, target, overlap_checkpointer_probe):
    from med_evidence_agentic_rag.models_and_database.postgres import engine
    from med_evidence_agentic_rag.models_and_database.redis import (
        HUMAN_CHALLENGE_PREFIX,
        redis_client,
    )
    from med_evidence_agentic_rag.persistence.operations import journal

    challenge = (await client.get("/api/auth/captcha")).raise_for_status().json()["challenge_id"]
    answer = await redis_client.get(HUMAN_CHALLENGE_PREFIX + challenge)
    human = (
        await client.post(
            "/api/auth/human-check", json={"challenge_id": challenge, "answer": answer}
        )
    ).raise_for_status()
    login = (
        (
            await client.post(
                "/api/auth/login/access-token",
                json={
                    "human_token": human.json()["human_token"],
                    "visitor_id": uuid4().hex,
                    "access_code": os.environ["ACCESS_TOKEN_CODE"],
                },
            )
        )
        .raise_for_status()
        .json()
    )
    token = login["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    document = (
        (
            await client.post(
                "/api/documents/upload", files={"file": ("fixture.txt", PASSAGE, "text/plain")}
            )
        )
        .raise_for_status()
        .json()
    )
    assert document["status"] == "ready" and document["chunks"] > 0

    async def ask(question):
        chat = (await client.post("/api/conversations")).raise_for_status().json()["id"]
        stream = (
            await client.post("/api/ask", json={"question": question, "conversation_id": chat})
        ).raise_for_status()
        events = [json.loads(line) for line in stream.text.splitlines()]
        assert events[-1]["type"] == "done", [e["type"] for e in events]
        return chat, events

    chat, events = await ask(QUERY)
    evidence = next(event["evidences"][0] for event in events if event["type"] == "evidence")
    assert PASSAGE in evidence["content"]
    turns = (await client.get(f"/api/conversations/{chat}/turns")).raise_for_status().json()
    assert evidence["citation_id"] in turns[0]["answer"] and PASSAGE in turns[0]["answer"]
    async with engine.connect() as conn:
        assert await conn.scalar(
            text("SELECT count(*) FROM checkpoints WHERE thread_id=:chat"), {"chat": chat}
        )
        owner = await conn.scalar(
            text("SELECT user_id FROM conversations WHERE id=:chat"), {"chat": UUID(chat)}
        )

    await overlap_checkpointer_probe()

    # Execute the public load CLI with private credentials only in its process environment.
    process = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "med_evidence_agentic_rag.evaluation.load",
            "--target",
            target,
            "--allow-live",
            "--concurrency",
            "2",
            "--requests",
            "4",
            "--duration",
            "60",
        ],
        env=dict(os.environ, EVAL_AGENT_TOKEN=token, EVAL_LOAD_QUERY=QUERY),
        stdout=subprocess.PIPE,
        timeout=90,
    )
    report = json.loads(process.stdout)
    assert (
        process.returncode == 0 and report["completed"] == 4 and report["peak_client_inflight"] == 2
    )
    assert token not in str(report) and QUERY not in str(report)
    print("Synthetic load", json.dumps(report), flush=True)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM conversations")) == 1

    for i in range(2):
        mail_chat, _ = await ask(MAIL)
        drafts = (
            (await client.get("/api/email-actions", params={"conversation_id": mail_chat}))
            .raise_for_status()
            .json()
        )
        assert len(drafts) == 1 and drafts[0]["status"] == "pending"
        assert len(capture.accepted) == i
        confirm = f"/api/email-actions/{drafts[0]['id']}/confirm"
        if i == 1:
            # Simulate lost SQL acknowledgement after a real loopback SMTP acceptance.
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "ALTER TABLE email_actions ADD CONSTRAINT fixture_ack "
                        "CHECK (status <> 'sent') NOT VALID"
                    )
                )
            assert (await client.post(confirm)).status_code == 500
            assert (await client.post(confirm)).raise_for_status().json()["status"] == "executing"
        else:
            for _ in range(2):
                assert (await client.post(confirm)).raise_for_status().json()["status"] == "sent"
        assert len(capture.accepted) == i + 1
        assert b"capture@example.org" in capture.accepted[-1]

    await journal.begin("upload", owner, uuid4(), "interrupted.txt")
    (Path(os.environ["DATA_ROOT"]) / "uploads" / str(owner) / "interrupted.txt").write_text(PASSAGE)


async def scenario():
    target = make_url(os.environ["POSTGRESQL_URL"])
    template = make_url(os.environ["MEDEVIDENCE_TEST_POSTGRES_URL"])
    assert (
        target.database.startswith("verification_")
        and target.set(database=template.database) == template
    )
    assert os.environ["REDIS_URL"] == os.environ["MEDEVIDENCE_TEST_REDIS_URL"]
    install_inference()
    capture = Capture()
    smtp = await asyncio.start_server(capture.session, "127.0.0.1", 0)
    async with smtp:
        os.environ.update(
            SMTP_HOST="127.0.0.1",
            SMTP_PORT=str(smtp.sockets[0].getsockname()[1]),
            SMTP_TRANSPORT="plain",
            MY_SENDER_EMAIL="fixture@example.org",
        )
        from med_evidence_agentic_rag.api import app as application
        from med_evidence_agentic_rag.models_and_database.postgres import engine
        from med_evidence_agentic_rag.persistence.migrations import migrate

        app = application.app

        class ObservedReadiness(application.Readiness):
            def __init__(self):
                super().__init__(interval=0.2)
                self.watch_checkpoint = False
                self.overlap_started = asyncio.Event()
                self.overlap_succeeded = asyncio.Event()

            def start(self, probes):
                probe = probes["checkpointer"]

                async def observe():
                    overlap = self.watch_checkpoint and app.state.checkpointer.lock.locked()
                    if overlap:
                        self.overlap_started.set()
                    await probe()
                    if overlap:
                        self.overlap_succeeded.set()

                super().start({**probes, "checkpointer": observe})

        async def overlap_checkpointer_probe():
            readiness = app.state.readiness
            async with asyncio.timeout(5), asyncio.TaskGroup() as group:
                async with engine.begin() as blocker:
                    await blocker.execute(text("LOCK TABLE checkpoints IN ACCESS EXCLUSIVE MODE"))
                    readiness.watch_checkpoint = True
                    group.create_task(app.state.checkpointer.adelete_thread(str(uuid4())))
                    await readiness.overlap_started.wait()
                    assert not readiness.overlap_succeeded.is_set()
                await readiness.overlap_succeeded.wait()
            readiness.watch_checkpoint = False

        application.Readiness = ObservedReadiness
        async with engine.begin() as conn:
            await conn.run_sync(migrate, "upgrade")
        async with (
            serve(app) as url,
            httpx.AsyncClient(base_url=url, timeout=60, trust_env=False) as client,
        ):
            await exercise(client, capture, url, overlap_checkpointer_probe)


if __name__ == "__main__":
    with model_endpoint() as endpoint:
        os.environ["LLM_BASE_URL"] = endpoint
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None
        ) as runner:
            runner.run(scenario())
