"""Request correlation and bounded-cardinality Prometheus exposition."""

import re
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from med_evidence_agentic_rag.utils.logger import request_id, setup_logger

logger = setup_logger(__name__)
_ID = re.compile(r"[A-Za-z0-9._-]{1,64}", flags=re.ASCII)
_BUCKETS = (0.1, 0.5, 1, 3, 10, 30, 60, 120, 300)
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"})


@dataclass
class RequestStats:
    count: int = 0
    seconds: float = 0
    buckets: list[int] = field(default_factory=lambda: [0] * len(_BUCKETS))


class Telemetry:
    def __init__(self) -> None:
        self.requests: dict[tuple[str, str, str], RequestStats] = {}
        self.active = 0
        self.recovery = dict.fromkeys(("completed", "busy", "failed", "emails_unknown"), 0)
        self.turn_failures: dict[str, int] = dict.fromkeys(
            ("timeout", "interrupted", "persistence", "execution", "incomplete"), 0
        )

    def observe(self, method: str, route: str, code: str, seconds: float) -> None:
        stats = self.requests.setdefault((method, route, code), RequestStats())
        stats.count += 1
        stats.seconds += seconds
        for index, limit in enumerate(_BUCKETS):
            if seconds <= limit:
                stats.buckets[index] += 1

    def render(
        self,
        dependencies: dict[str, str] | None = None,
        *,
        ready: bool = False,
        protected: int = 0,
        deliveries: int = 0,
    ) -> str:
        lines = [
            "# TYPE medevidence_active_requests gauge",
            f"medevidence_active_requests {self.active}",
            "# TYPE medevidence_active_resource_leases gauge",
            f"medevidence_active_resource_leases {protected}",
            "# TYPE medevidence_active_email_deliveries gauge",
            f"medevidence_active_email_deliveries {deliveries}",
            "# TYPE medevidence_ready gauge",
            f"medevidence_ready {int(ready)}",
            "# TYPE medevidence_dependency_ready gauge",
        ]
        for name, state in sorted((dependencies or {}).items()):
            sample = int(state == "ready")
            lines.append(f'medevidence_dependency_ready{{dependency="{name}"}} {sample}')
        for group, counts in (
            ("recovery", self.recovery),
            ("turn_failures", self.turn_failures),
        ):
            lines.append(f"# TYPE medevidence_{group}_total counter")
            for outcome, count in counts.items():
                lines.append(f'medevidence_{group}_total{{outcome="{outcome}"}} {count}')
        basename = "medevidence_http_request_duration_seconds"
        lines.append(f"# TYPE {basename} histogram")
        for (method, route, code), stats in sorted(self.requests.items()):
            labels = f'method="{method}",route="{route}",status_class="{code}"'
            for limit, count in zip(_BUCKETS, stats.buckets, strict=True):
                lines.append(f'{basename}_bucket{{{labels},le="{limit}"}} {count}')
            lines.extend((
                f'{basename}_bucket{{{labels},le="+Inf"}} {stats.count}',
                f"{basename}_count{{{labels}}} {stats.count}",
                f"{basename}_sum{{{labels}}} {stats.seconds}",
            ))
        return "\n".join(lines) + "\n"


telemetry = Telemetry()


class RequestTelemetry:
    def __init__(self, app: ASGIApp, telemetry: Telemetry) -> None:
        self.app = app
        self.telemetry = telemetry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).getlist("x-request-id")
        identity = supplied[0] if len(supplied) == 1 and _ID.fullmatch(supplied[0]) else uuid4().hex
        token = request_id.set(identity)
        started = monotonic()
        status = 0
        failed = False
        self.telemetry.active += 1

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = identity
                headers["X-Frame-Options"] = "DENY"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Content-Security-Policy"] = "frame-ancestors 'none'; object-src 'none'"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            failed = True
            logger.exception("HTTP request failed")
            if status:
                raise RuntimeError("HTTP response interrupted") from None
            await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                scope, receive, send_with_headers
            )
        finally:
            self.telemetry.active -= 1
            template = getattr(scope.get("route"), "path", None)
            route = template if isinstance(template, str) else "unmatched"
            method = scope["method"] if scope["method"] in _METHODS else "OTHER"
            code = "5xx" if failed else f"{status // 100}xx" if 100 <= status <= 599 else "none"
            seconds = monotonic() - started
            self.telemetry.observe(method, route, code, seconds)
            logger.info(
                "HTTP request",
                extra={"http_method": method, "http_route": route, "status_class": code,
                       "duration_seconds": round(seconds, 6)},
            )
            request_id.reset(token)
