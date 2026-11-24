import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from med_evidence_agentic_rag.persistence.postgres_tables import Base
from med_evidence_agentic_rag.services.email_approval import ApprovalError, EmailApprovalRegistry


def test_restart_foreign_expired_rejected_and_unknown(tmp_path):
    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        now = [datetime.now(UTC)]
        first, second = [
            EmailApprovalRegistry(factory, ttl_seconds=10, clock=lambda: now[0]) for _ in range(2)
        ]
        owner, conversation = uuid4(), uuid4()

        async def propose():
            return await first.propose(owner, conversation, "draft@example.org", "summary", "body")

        def disconnected(*args):
            raise ConnectionError("lost acknowledgement")

        action = await propose()
        assert (await second.list_pending(owner, conversation))[0] == action
        with pytest.raises(ApprovalError):
            await second.confirm(uuid4(), action.id, disconnected)
        assert (await second.confirm(owner, action.id, disconnected)).status == "unknown"
        assert (
            await first.confirm(owner, action.id, lambda *args: pytest.fail())
        ).status == "unknown"
        rejected = await propose()
        await second.reject(owner, rejected.id)
        with pytest.raises(ApprovalError):
            await first.confirm(owner, rejected.id, disconnected)
        expired = await propose()
        now[0] += timedelta(seconds=11)
        with pytest.raises(ApprovalError):
            await second.confirm(owner, expired.id, disconnected)
        assert [a.id for a in await second.list_pending(owner, conversation)] == [action.id]
        await engine.dispose()

    asyncio.run(scenario())


def test_cancelled_request_keeps_single_claim(tmp_path):
    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        first, second = EmailApprovalRegistry(factory), EmailApprovalRegistry(factory)
        owner = uuid4()
        action = await first.propose(owner, uuid4(), "draft@example.org", "summary", "body")
        entered, release = Event(), Event()
        sends = []

        def sender(*args):
            sends.append(args)
            entered.set()
            return release.wait(10)

        request = asyncio.create_task(first.confirm(owner, action.id, sender))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert (await second.confirm(owner, action.id, sender)).status == "executing"
        finally:
            release.set()
            await first.wait_pending()
        assert (await second.confirm(owner, action.id, sender)).status == "sent"
        assert len(sends) == 1
        await engine.dispose()

    asyncio.run(scenario())


def test_invalid_adoption_refuses_to_stamp(tmp_path):
    async def scenario():
        from med_evidence_agentic_rag.persistence.migrations import migrate

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
            with pytest.raises(RuntimeError, match="legacy"):
                await conn.run_sync(migrate, "adopt-legacy")
            assert not await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
        await engine.dispose()

    asyncio.run(scenario())


def test_upgrade_requires_empty_or_recognized_schema(tmp_path):
    async def scenario():
        from med_evidence_agentic_rag.persistence.migrations import migrate

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unknown.db'}")
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
            with pytest.raises(RuntimeError, match="schema"):
                await conn.run_sync(migrate, "upgrade")
            assert not await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
        await engine.dispose()

    asyncio.run(scenario())


def test_lost_connection_after_acceptance_remains_sent(monkeypatch):
    from unittest.mock import MagicMock

    from med_evidence_agentic_rag.tools import send_email as mail

    monkeypatch.setenv("MY_SENDER_EMAIL", "sender@example.org")
    monkeypatch.setenv("MY_SENDER_PASSWORD", "synthetic")
    smtp = MagicMock()
    smtp.return_value.quit.side_effect = ConnectionError("QUIT lost")
    monkeypatch.setattr(mail.smtplib, "SMTP_SSL", smtp)
    assert mail.send_email("recipient@example.org", "summary", "body") is True
    smtp.return_value.sendmail.side_effect = ConnectionError("DATA lost")
    with pytest.raises(mail.DeliveryUncertain):
        mail.send_email("recipient@example.org", "summary", "body")


@pytest.mark.parametrize("quit_failed", [False, True])
@pytest.mark.parametrize("refused", ["sender", "recipient", "data"])
def test_definitive_smtp_refusal_survives_cleanup(monkeypatch, refused, quit_failed):
    from unittest.mock import MagicMock

    from med_evidence_agentic_rag.tools import send_email as mail

    monkeypatch.setenv("MY_SENDER_EMAIL", "sender@example.org")
    monkeypatch.setenv("MY_SENDER_PASSWORD", "synthetic")
    smtp = MagicMock()
    server = smtp.return_value
    errors = {
        "sender": mail.smtplib.SMTPSenderRefused(553, b"denied", "sender@example.org"),
        "recipient": mail.smtplib.SMTPRecipientsRefused({}),
        "data": mail.smtplib.SMTPDataError(554, b"denied"),
    }
    server.sendmail.side_effect = errors[refused]
    if quit_failed:
        server.quit.side_effect = ConnectionError("QUIT lost")
    monkeypatch.setattr(mail.smtplib, "SMTP_SSL", smtp)
    if refused == "recipient":
        with pytest.raises(mail.EmailRecipientError):
            mail.send_email("recipient@example.org", "summary", "body")
    else:
        assert mail.send_email("recipient@example.org", "summary", "body") is False
