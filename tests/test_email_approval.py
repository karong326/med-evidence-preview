import asyncio
from uuid import uuid4

import pytest

from med_evidence_agentic_rag.services.email_approval import ApprovalError, EmailApprovalRegistry


def test_email_cannot_execute_without_owner_confirmation_and_cannot_send_twice():
    async def scenario():
        registry = EmailApprovalRegistry()
        owner, other, conversation = uuid4(), uuid4(), uuid4()
        action = registry.propose(owner, conversation, "test@example.org", "subject", "body")
        sent = []

        def sender(*args):
            sent.append(args)
            return True

        assert registry.list_pending(owner, conversation)[0].id == action.id
        assert registry.list_pending(other, conversation) == []
        with pytest.raises(ApprovalError):
            await registry.confirm(other, action.id, sender)
        assert sent == []
        assert (await registry.confirm(owner, action.id, sender)).status == "sent"
        assert (await registry.confirm(owner, action.id, sender)).status == "sent"
        assert len(sent) == 1
        assert sent[0][:3] == ("test@example.org", "subject", "body")

    asyncio.run(scenario())


def test_expired_and_rejected_actions_never_execute():
    async def scenario():
        now = [0.0]
        registry = EmailApprovalRegistry(ttl_seconds=10, clock=lambda: now[0])
        owner, conversation = uuid4(), uuid4()
        action = registry.propose(owner, conversation, "test@example.org", "subject", "body")
        registry.reject(owner, action.id)
        with pytest.raises(ApprovalError):
            await registry.confirm(owner, action.id, lambda *args: True)
        expired = registry.propose(owner, conversation, "test@example.org", "subject", "body")
        now[0] = 11
        with pytest.raises(ApprovalError):
            await registry.confirm(owner, expired.id, lambda *args: True)

    asyncio.run(scenario())


def test_request_cancellation_does_not_allow_duplicate_send():
    async def scenario():
        registry = EmailApprovalRegistry()
        owner = uuid4()
        action = registry.propose(owner, uuid4(), "test@example.org", "subject", "body")
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def execute(record, sender):
            calls.append(record.id)
            entered.set()
            await release.wait()
            record.status = "sent"

        registry._execute = execute
        task = asyncio.create_task(registry.confirm(owner, action.id, lambda *args: True))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await registry.confirm(owner, action.id, lambda *args: True)).status == "executing"
        release.set()
        await registry.wait_pending()
        assert len(calls) == 1
        assert (await registry.confirm(owner, action.id, lambda *args: True)).status == "sent"

    asyncio.run(scenario())
