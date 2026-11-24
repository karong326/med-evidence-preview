"""Monitor one in-flight probe per dependency and serve bounded cached snapshots."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from time import monotonic

from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
Probe = Callable[[], Awaitable[None]]


class Readiness:
    def __init__(
        self, *, interval: float = 5, timeout: float = 3, max_age: float = 15
    ) -> None:
        self.interval = interval
        self.timeout = timeout
        self.max_age = max_age
        self.phase = "starting"
        self.samples: dict[str, tuple[str, float]] = {}
        self.tasks: list[asyncio.Task[None]] = []

    def start(self, probes: Mapping[str, Probe]) -> None:
        self.samples = dict.fromkeys(probes, ("starting", monotonic()))
        self.phase = "running"
        self.tasks = [
            asyncio.create_task(self._monitor(name, probe)) for name, probe in probes.items()
        ]

    async def _monitor(self, name: str, probe: Probe) -> None:
        async def perform() -> None:
            await probe()

        while True:
            task = asyncio.create_task(perform())
            try:
                done, _ = await asyncio.wait({task}, timeout=self.timeout)
                if not done:
                    self.samples[name] = ("unavailable", monotonic())
                try:
                    await asyncio.shield(task)
                except Exception:
                    if self.samples[name][0] != "unavailable":
                        logger.exception("Dependency check failed", extra={"dependency": name})
                    self.samples[name] = ("unavailable", monotonic())
                else:
                    if done:
                        self.samples[name] = ("ready", monotonic())
            finally:
                # A Chroma metadata read can run in a thread. Keep the index
                # owner fence until it returns, even after the probe deadline.
                while not task.done():
                    await asyncio.wait({task})
                if not task.cancelled():
                    task.exception()
            await asyncio.sleep(self.interval)

    def snapshot(self) -> dict[str, str | dict[str, str]]:
        now = monotonic()
        services = {
            name: status if now - sampled_at <= self.max_age else "unavailable"
            for name, (status, sampled_at) in self.samples.items()
        }
        phase = self.phase
        if phase == "running":
            values = set(services.values())
            if "unavailable" in values:
                phase = "unavailable"
            elif "starting" in values:
                phase = "starting"
            else:
                phase = "ready"
        return {"status": phase, "services": services}

    async def stop(self) -> None:
        self.phase = "draining"
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            drain = asyncio.gather(*self.tasks, return_exceptions=True)
            cancelled: asyncio.CancelledError | None = None
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError as error:
                    cancelled = error
            self.tasks.clear()
            if cancelled is not None:
                raise cancelled
