import asyncio
from uuid import uuid4

import pytest

from med_evidence_agentic_rag.services.email_approval import ApprovalError, EmailApprovalRegistry


def test_email_cannot_execute_without_owner_confirmation_and_cannot_send_twice(email_sessions):
    async def scenario():
        registry = EmailApprovalRegistry(email_sessions)
        owner, other, conversation = uuid4(), uuid4(), uuid4()
        action = await registry.propose(owner, conversation, "test@example.org", "subject", "body")
        sent = []

        def sender(*args):
            sent.append(args)
            return True

        assert (await registry.list_pending(owner, conversation))[0].id == action.id
        assert await registry.list_pending(other, conversation) == []
        with pytest.raises(ApprovalError):
            await registry.confirm(other, action.id, sender)
        assert sent == []
        assert (await registry.confirm(owner, action.id, sender)).status == "sent"
        assert (await registry.confirm(owner, action.id, sender)).status == "sent"
        assert len(sent) == 1
        assert sent[0][:3] == ("test@example.org", "subject", "body")

    asyncio.run(scenario())
