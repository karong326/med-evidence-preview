"""Single-worker, owner-bound email approval and at-most-once execution per action."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
Sender = Callable[[str, str, str, bool], bool]


class ApprovalError(ValueError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class EmailAction:
    id: UUID
    owner_id: UUID
    conversation_id: UUID
    recipient: str
    subject: str
    body: str
    is_html: bool
    deadline: float
    status: Literal["pending", "executing", "sent", "failed", "rejected"] = "pending"


class EmailApprovalRegistry:
    def __init__(self, ttl_seconds: float = 300, clock: Callable[[], float] = monotonic) -> None:
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._actions: dict[UUID, EmailAction] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def _purge(self) -> None:
        now = self.clock()
        self._actions = {
            key: action
            for key, action in self._actions.items()
            if action.status == "executing" or action.deadline > now
        }

    def propose(
        self,
        owner: UUID,
        conversation: UUID,
        recipient: str,
        subject: str,
        body: str,
        is_html: bool = False,
    ) -> EmailAction:
        self._purge()
        if not recipient or not subject.strip() or not body.strip():
            raise ApprovalError("收件人、主题和正文不能为空", 400)
        if len(subject) > 300 or len(body) > 100_000:
            raise ApprovalError("邮件内容过长", 400)
        if (
            len(self._actions) >= 200
            or sum(a.owner_id == owner for a in self._actions.values()) >= 10
        ):
            raise ApprovalError("待处理邮件过多，请先确认或取消已有邮件", 429)
        action = EmailAction(
            uuid4(),
            owner,
            conversation,
            recipient,
            subject,
            body,
            is_html,
            self.clock() + self.ttl_seconds,
        )
        self._actions[action.id] = action
        return action

    def _owned(self, owner: UUID, action_id: UUID) -> EmailAction:
        self._purge()
        action = self._actions.get(action_id)
        if action is None or action.owner_id != owner:
            raise ApprovalError("确认事项不存在或已过期", 404)
        return action

    def list_pending(self, owner: UUID, conversation: UUID) -> list[EmailAction]:
        self._purge()
        return [
            a
            for a in self._actions.values()
            if a.owner_id == owner
            and a.conversation_id == conversation
            and a.status in {"pending", "executing"}
        ]

    def reject(self, owner: UUID, action_id: UUID) -> EmailAction:
        action = self._owned(owner, action_id)
        if action.status != "pending":
            raise ApprovalError("此邮件已处理或正在执行")
        action.status = "rejected"
        return action

    async def confirm(self, owner: UUID, action_id: UUID, sender: Sender) -> EmailAction:
        action = self._owned(owner, action_id)
        if action.status in {"executing", "sent", "failed"}:
            return action  # Retrying the HTTP request never repeats the external send.
        if action.status != "pending":
            raise ApprovalError("此邮件已取消")
        # No await between ownership/status check and claim in this single event-loop worker.
        action.status = "executing"
        task = asyncio.create_task(self._execute(action, sender))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        await asyncio.shield(task)
        return action

    async def _execute(self, action: EmailAction, sender: Sender) -> None:
        try:
            accepted = await asyncio.to_thread(
                sender, action.recipient, action.subject, action.body, action.is_html
            )
            action.status = "sent" if accepted else "failed"
        except Exception:
            action.status = "failed"
            logger.exception("Confirmed email failed; action=%s", action.id)
        action.deadline = self.clock() + self.ttl_seconds

    async def wait_pending(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks))


email_approvals = EmailApprovalRegistry()
