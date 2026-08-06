from __future__ import annotations

from dataclasses import dataclass
import errno
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Any


class FileLockTimeoutError(TimeoutError):
    pass


@dataclass
class FileLock:
    path: Path
    timeout_seconds: float = 30.0
    poll_seconds: float = 0.1

    def __post_init__(self) -> None:
        self._handle = None
        self._backend: _LockBackend | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        self._backend = _backend_for_platform()
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._backend.acquire(self._handle)
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(f"pid {os.getpid()} lock acquired at {time.time()}\n")
                self._handle.flush()
                return self
            except OSError as exc:
                if not self._backend.is_lock_contention(exc):
                    self._handle.close()
                    self._handle = None
                    raise
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise FileLockTimeoutError(
                        f"Timed out waiting for lock {self.path}. Another process may be indexing."
                    ) from exc
                time.sleep(self.poll_seconds)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            if self._backend is not None:
                self._backend.release(self._handle)
        finally:
            self._handle.close()
            self._handle = None
            self._backend = None


class _LockBackend:
    def acquire(self, handle: Any) -> None:
        raise NotImplementedError

    def release(self, handle: Any) -> None:
        raise NotImplementedError

    def is_lock_contention(self, exc: OSError) -> bool:
        return exc.errno in {errno.EACCES, errno.EAGAIN}


class _UnixLockBackend(_LockBackend):
    def __init__(self) -> None:
        import fcntl

        self._fcntl = fcntl

    def acquire(self, handle: Any) -> None:
        self._fcntl.flock(handle.fileno(), self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)

    def release(self, handle: Any) -> None:
        self._fcntl.flock(handle.fileno(), self._fcntl.LOCK_UN)


class _WindowsLockBackend(_LockBackend):
    _LOCK_BYTES = 1

    def __init__(self) -> None:
        import msvcrt

        self._msvcrt = msvcrt

    def acquire(self, handle: Any) -> None:
        handle.seek(0)
        self._msvcrt.locking(handle.fileno(), self._msvcrt.LK_NBLCK, self._LOCK_BYTES)

    def release(self, handle: Any) -> None:
        handle.seek(0)
        self._msvcrt.locking(handle.fileno(), self._msvcrt.LK_UNLCK, self._LOCK_BYTES)

    def is_lock_contention(self, exc: OSError) -> bool:
        return exc.errno in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}


def _backend_for_platform() -> _LockBackend:
    if os.name == "nt":
        return _WindowsLockBackend()
    return _UnixLockBackend()
