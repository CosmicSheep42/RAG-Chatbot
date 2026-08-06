from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Iterator


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def normalize_request_id(value: str | None) -> str:
    if value:
        candidate = value.strip()
        if REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
    return uuid.uuid4().hex


@dataclass(frozen=True)
class LatencySummary:
    count: int
    average_ms: float
    p95_ms: float


class MetricsRegistry:
    def __init__(self, *, max_samples: int = 1000) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._latencies: dict[tuple[str, tuple[tuple[str, str], ...]], deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def set_gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe_latency(self, name: str, seconds: float, **labels: Any) -> None:
        with self._lock:
            self._latencies[self._key(name, labels)].append(max(0.0, seconds))

    @contextmanager
    def timer(self, name: str, **labels: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe_latency(name, time.perf_counter() - started, **labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = {
                self._format_key(key): self._summary(values).__dict__
                for key, values in self._latencies.items()
            }
            return {
                "counters": {self._format_key(key): value for key, value in self._counters.items()},
                "gauges": {self._format_key(key): value for key, value in self._gauges.items()},
                "latencies": latencies,
            }

    def prometheus_text(self) -> str:
        lines: list[str] = []
        snapshot = self.snapshot()
        for name, value in snapshot["counters"].items():
            lines.append(f"{self._metric_name(name)} {value}")
        for name, value in snapshot["gauges"].items():
            lines.append(f"{self._metric_name(name)} {value}")
        for name, summary in snapshot["latencies"].items():
            metric = self._metric_name(name)
            lines.append(f"{metric}_count {summary['count']}")
            lines.append(f"{metric}_avg_ms {summary['average_ms']:.3f}")
            lines.append(f"{metric}_p95_ms {summary['p95_ms']:.3f}")
        return "\n".join(lines) + "\n"

    def _key(self, name: str, labels: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((str(key), str(value)) for key, value in labels.items()))

    def _format_key(self, key: tuple[str, tuple[tuple[str, str], ...]]) -> str:
        name, labels = key
        if not labels:
            return name
        label_text = ",".join(f"{k}={v}" for k, v in labels)
        return f"{name}{{{label_text}}}"

    def _metric_name(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_:]", "_", name)

    def _summary(self, values: deque[float]) -> LatencySummary:
        if not values:
            return LatencySummary(0, 0.0, 0.0)
        samples = sorted(values)
        index = min(len(samples) - 1, int(len(samples) * 0.95))
        average = sum(samples) / len(samples)
        return LatencySummary(len(samples), average * 1000, samples[index] * 1000)


metrics = MetricsRegistry()
