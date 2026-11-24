"""提供 SMTP 发送实现与 Agent 可调用的受控邮件工具。

底层 ``send_email`` 从当前进程环境读取发件人凭证并映射常见 SMTP 异常；Tool 层从 runtime
获取登录用户，将“我的邮箱”或联系人别名在服务器端解析为真实地址，再在线程中执行阻塞发送。
工具结果为 JSON，执行记录仅展示安全摘要，不回显正文或联系人邮箱。
"""

import json
import os
import smtplib
import ssl
from contextlib import suppress
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from uuid import UUID

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from med_evidence_agentic_rag.agent.context import UserContext
from med_evidence_agentic_rag.persistence.user_preferences import (
    resolve_email_recipient,
)
from med_evidence_agentic_rag.services.email_approval import DeliveryRejected, email_approvals
from med_evidence_agentic_rag.utils.config import integer_setting
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


def smtp_configured() -> bool:
    return bool(os.getenv("MY_SENDER_EMAIL")) and (
        os.getenv("SMTP_TRANSPORT") == "plain" or bool(os.getenv("MY_SENDER_PASSWORD"))
    )


class EmailRecipientError(DeliveryRejected):
    """SMTP 已确认收件邮箱不存在或不可投递。"""


class DeliveryUncertain(RuntimeError):
    """SMTP acceptance could not be established."""


def send_email(
    recipient: str,
    email_subject: str,
    email_content: str,
    is_html: bool = False,
) -> bool:
    """通过配置的 SMTP 服务发送一封纯文本或 HTML 邮件。

    Args:
        recipient: 收件人邮箱地址。
        email_subject: 邮件主题。
        email_content: 邮件正文。
        is_html: 是否按 HTML 格式发送正文。
    Returns:
        SMTP 服务接受时为 True；配置、认证或明确拒绝时为 False。
    Raises:
        DeliveryUncertain: 提交过程中连接中断，结果未知。
        EmailRecipientError: SMTP 明确返回收件邮箱不存在或无法投递。
    Side effects:
        连接配置的 SMTP 服务并真实发送邮件。
    """
    # 凭证在调用时读取，缺失配置不会导致模块导入失败。
    sender_email = os.getenv("MY_SENDER_EMAIL", "")
    sender_password = os.getenv("MY_SENDER_PASSWORD")
    transport = os.getenv("SMTP_TRANSPORT", "ssl")
    if not smtp_configured() or transport not in {"ssl", "starttls", "plain"}:
        logger.error("SMTP configuration is missing or invalid")
        return False

    # 根据 is_html 选择正文类型，其余 SMTP 参数保持统一。
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = email_subject
    message.attach(MIMEText(email_content, "html" if is_html else "plain", "utf-8"))

    submitted = False
    try:
        port = integer_setting("SMTP_PORT", 465 if transport == "ssl" else 587, 1, 65535)
        host = os.getenv("SMTP_HOST", "smtp.qq.com")
        if not host or (transport == "plain" and not os.getenv("SMTP_HOST")):
            return False
        server = (
            smtplib.SMTP_SSL(host, port, timeout=15, context=ssl.create_default_context())
            if transport == "ssl" else smtplib.SMTP(host, port, timeout=15)
        )
    except Exception:
        return False
    try:
        if transport == "starttls":
            server.starttls(context=ssl.create_default_context())
        if transport != "plain" and sender_password is not None:
            server.login(sender_email, sender_password)
        payload = message.as_string()
        submitted = True
        server.sendmail(sender_email, [recipient], payload)
    except smtplib.SMTPSenderRefused:
        return False
    except smtplib.SMTPDataError as exc:
        if exc.smtp_code == 550:
            logger.warning("邮件发送失败：收件邮箱不存在或无法投递")
            raise EmailRecipientError("收件邮箱不存在或无法投递，请检查邮箱地址") from exc
        logger.error("邮件发送失败：SMTP 数据错误，代码=%s", exc.smtp_code)
        return False
    except smtplib.SMTPRecipientsRefused as exc:
        logger.warning("邮件发送失败：收件邮箱被服务器拒绝")
        raise EmailRecipientError("收件邮箱不存在或无法投递，请检查邮箱地址") from exc
    except smtplib.SMTPAuthenticationError:
        logger.error("邮件发送失败：SMTP 认证未通过")
        return False
    except Exception:
        if not submitted:
            return False
        raise DeliveryUncertain("无法确认 SMTP 服务是否接受了邮件") from None
    finally:
        # QUIT can fail after any DATA or envelope response. Preserve that outcome.
        with suppress(Exception):
            server.quit()
        with suppress(Exception):
            server.close()

    logger.info("SMTP 服务已接受邮件")
    return True


@tool("send_email")
async def send_email_tool(
    recipient: str,
    email_subject: str,
    email_content: str,
    runtime: ToolRuntime[UserContext, dict[str, Any]],
    is_html: bool = False,
) -> str:
    """生成一封等待当前用户在页面确认的邮件草稿。

    recipient 可为原始邮箱、“我的邮箱”或已保存联系人别名；实际邮箱只在服务端解析。
    工具不会发送邮件。只有当前用户确认绑定的收件人、主题和正文后才执行外发。
    """
    raw_user_id = runtime.context.get("user_id", "")
    if not raw_user_id:
        raise ValueError("当前 Tool 缺少登录用户")
    resolved_recipient = await resolve_email_recipient(UUID(raw_user_id), recipient)
    if not resolved_recipient:
        return json.dumps(
            {
                "success": False,
                "recipient": recipient,
                "message": "未找到收件人邮箱，请提供邮箱或先明确要求记住联系人。",
            },
            ensure_ascii=False,
        )
    raw_conversation = runtime.context.get("conversation_id")
    if not raw_conversation:
        raise ValueError("邮件草稿需要有效会话")
    action = await email_approvals.propose(
        UUID(raw_user_id),
        UUID(raw_conversation),
        resolved_recipient,
        email_subject,
        email_content,
        is_html,
    )
    return json.dumps(
        {
            "success": False,
            "requires_confirmation": True,
            "action_id": str(action.id),
            "message": "邮件草稿已生成，等待用户在页面核对并确认，尚未发送。",
        },
        ensure_ascii=False,
    )
