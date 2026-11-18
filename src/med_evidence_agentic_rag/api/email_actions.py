"""Explicit user confirmation endpoints; these endpoints are not Agent tools."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.services.email_approval import (
    ApprovalError,
    EmailAction,
    email_approvals,
)

email_actions_router = APIRouter(prefix="/api/email-actions", tags=["邮件确认"])


class EmailActionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    conversation_id: UUID
    recipient: str
    subject: str
    body: str
    status: str


def response(action: EmailAction) -> EmailActionResponse:
    return EmailActionResponse.model_validate(action)


@email_actions_router.get("")
async def pending_email_actions(
    conversation_id: UUID, user: AuthenticatedUser = Depends(require_user)
) -> list[EmailActionResponse]:
    return [response(action) for action in email_approvals.list_pending(user.id, conversation_id)]


@email_actions_router.post("/{action_id}/confirm")
async def confirm_email(
    action_id: UUID, user: AuthenticatedUser = Depends(require_user)
) -> EmailActionResponse:
    from med_evidence_agentic_rag.tools.send_email import send_email

    try:
        return response(await email_approvals.confirm(user.id, action_id, send_email))
    except ApprovalError as error:
        raise HTTPException(error.status_code, str(error)) from error


@email_actions_router.post("/{action_id}/reject")
async def reject_email(
    action_id: UUID, user: AuthenticatedUser = Depends(require_user)
) -> EmailActionResponse:
    try:
        return response(email_approvals.reject(user.id, action_id))
    except ApprovalError as error:
        raise HTTPException(error.status_code, str(error)) from error
