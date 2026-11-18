"""提供 SMTP 发送实现与 Agent 可调用的受控邮件工具。

底层 ``send_email`` 从当前进程环境读取发件人凭证并映射常见 SMTP 异常；Tool 层从 runtime
获取登录用户，将“我的邮箱”或联系人别名在服务器端解析为真实地址，再在线程中执行阻塞发送。
工具结果为 JSON，执行记录仅展示安全摘要，不回显正文或联系人邮箱。
"""

import json
import os
import smtplib
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
from med_evidence_agentic_rag.services.email_approval import email_approvals
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465


class EmailRecipientError(ValueError):
    """SMTP 已确认收件邮箱不存在或不可投递。"""


def send_email(
    recipient: str,
    email_subject: str,
    email_content: str,
    is_html: bool = False,
) -> bool:
    """通过 QQ SMTP 发送一封纯文本或 HTML 邮件。

    Args:
        recipient: 收件人邮箱地址。
        email_subject: 邮件主题。
        email_content: 邮件正文。
        is_html: 是否按 HTML 格式发送正文。
    Returns:
        邮件已被 SMTP 服务接受时为 True；配置、认证或其他 SMTP 错误时为 False。
    Raises:
        EmailRecipientError: SMTP 明确返回收件邮箱不存在或无法投递。
    Side effects:
        连接 QQ SMTP 并真实发送邮件。
    """
    # 凭证在调用时读取，缺失配置不会导致模块导入失败。
    sender_email = os.getenv("MY_SENDER_EMAIL")
    sender_password = os.getenv("MY_SENDER_PASSWORD")
    if not sender_email or not sender_password:
        logger.error("邮件发送失败：未配置 MY_SENDER_EMAIL 或 MY_SENDER_PASSWORD")
        return False

    # 根据 is_html 选择正文类型，其余 SMTP 参数保持统一。
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = email_subject
    message.attach(MIMEText(email_content, "html" if is_html else "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [recipient], message.as_string())
    except smtplib.SMTPDataError as exc:
        if exc.smtp_code == 550:
            logger.warning("邮件发送失败：收件邮箱不存在或无法投递：%s", recipient)
            raise EmailRecipientError("收件邮箱不存在或无法投递，请检查邮箱地址") from exc
        logger.error("邮件发送失败：SMTP 数据错误，代码=%s", exc.smtp_code)
        return False
    except smtplib.SMTPRecipientsRefused as exc:
        logger.warning("邮件发送失败：收件邮箱被服务器拒绝：%s", recipient)
        raise EmailRecipientError("收件邮箱不存在或无法投递，请检查邮箱地址") from exc
    except smtplib.SMTPAuthenticationError:
        logger.error("邮件发送失败：SMTP 认证未通过")
        return False
    except Exception:
        logger.exception("邮件发送失败")
        return False

    logger.info("邮件发送成功：收件人=%s，主题=%s", recipient, email_subject)
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
    action = email_approvals.propose(
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
