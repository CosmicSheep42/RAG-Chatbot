from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
import threading
import time
from typing import AsyncIterator


@dataclass(frozen=True)
class RuntimeControlConfig:
    max_input_chars: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    inference_timeout_seconds: float
    max_concurrent_inferences: int = 2
    max_pending_inferences: int = 4
    request_timeout_seconds: float = 60.0
    rate_limit_max_buckets: int = 10000


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int, *, max_buckets: int = 10000) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_buckets = max(1, max_buckets)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            self._cleanup(now)
            if client_id not in self._requests and len(self._requests) >= self.max_buckets:
                return False, float(self.window_seconds)

            timestamps = self._requests[client_id]
            while timestamps and now - timestamps[0] > self.window_seconds:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = self.window_seconds - (now - timestamps[0])
                return False, max(retry_after, 0.0)

            timestamps.append(now)
            return True, 0.0

    def _cleanup(self, now: float) -> None:
        expired = [
            client_id
            for client_id, timestamps in self._requests.items()
            if not timestamps or now - timestamps[-1] > self.window_seconds
        ]
        for client_id in expired:
            self._requests.pop(client_id, None)


class RuntimeControls:
    def __init__(self, config: RuntimeControlConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window_seconds,
            max_buckets=config.rate_limit_max_buckets,
        )

    def validate_input(self, text: str) -> tuple[bool, str | None]:
        if not text or not text.strip():
            return False, "Message cannot be empty."
        if len(text) > self.config.max_input_chars:
            return False, f"Message exceeds the maximum size of {self.config.max_input_chars} characters."
        return True, None

    def allow_request(self, client_id: str | None = None) -> tuple[bool, float]:
        return self.rate_limiter.allow(client_id or "global")


class CapacityLimitExceeded(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        super().__init__("Inference capacity is saturated.")
        self.retry_after = retry_after


class AsyncInferenceCapacityLimiter:
    def __init__(self, *, max_active: int, max_pending: int, retry_after_seconds: float = 1.0) -> None:
        self.max_active = max(1, max_active)
        self.max_pending = max(0, max_pending)
        self.retry_after_seconds = retry_after_seconds
        self._semaphore = asyncio.Semaphore(self.max_active)
        self._lock = asyncio.Lock()
        self._active = 0
        self._pending = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> int:
        return self._pending

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._semaphore.locked() and self._pending >= self.max_pending:
                raise CapacityLimitExceeded(self.retry_after_seconds)
            self._pending += 1

        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._pending -= 1
                self._active += 1
            yield
        finally:
            if acquired:
                async with self._lock:
                    self._active -= 1
                self._semaphore.release()
            else:
                async with self._lock:
                    self._pending = max(0, self._pending - 1)
