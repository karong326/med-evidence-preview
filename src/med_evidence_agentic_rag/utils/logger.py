"""Application JSON logs with correlation and bounded exception diagnostics."""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_owned_loggers: set[str] = set()
_safe_fields = ("http_method", "http_route", "status_class", "duration_seconds", "dependency")


class SafeJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": (
                record.msg if record.name in _owned_loggers and isinstance(record.msg, str)
                else "Dependency log"
            ),
        }
        if identity := request_id.get():
            entry["request_id"] = identity
        if record.name in _owned_loggers:
            if isinstance(record.args, tuple) and record.args and all(
                type(value) in (int, float, bool) for value in record.args
            ):
                entry["event"] = record.getMessage()
            for field in _safe_fields:
                if (value := getattr(record, field, None)) is not None:
                    entry[field] = value
        if record.exc_info is not None:
            error_type, _, stack = record.exc_info
            if error_type is not None:
                entry["exception_type"] = error_type.__name__
                entry["frames"] = [
                    {"function": frame.name, "line": frame.lineno}
                    for frame in traceback.extract_tb(stack, limit=-8)
                ]
        return json.dumps(entry, ensure_ascii=True, separators=(",", ":"))


def configure_server_logging() -> None:
    """Route server/dependency warnings through the same safe formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeJsonFormatter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def setup_logger(name: str, level: str = "INFO", console: bool = True) -> logging.Logger:
    """Only project call sites supply event names. Never format record arguments."""
    logger = logging.getLogger(name)
    _owned_loggers.add(name)
    logger.setLevel(level)
    logger.propagate = False
    if console and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SafeJsonFormatter())
        logger.addHandler(handler)
    return logger
