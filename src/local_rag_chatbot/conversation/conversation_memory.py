from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import time
from typing import Callable


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    timestamp: float


@dataclass
class _ConversationSession:
    messages: list[ConversationMessage]
    updated_at: float


class InMemoryConversationStore:
    def __init__(
        self,
        *,
        max_messages: int = 8,
        ttl_seconds: int = 1800,
        max_sessions: int = 1000,
        max_message_chars: int = 2000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_messages = max(0, int(max_messages))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_sessions = max(1, int(max_sessions))
        self.max_message_chars = max(1, int(max_message_chars))
        self._clock = clock or time.time
        self._sessions: dict[str, _ConversationSession] = {}
        self._lock = RLock()

    def get_recent_messages(self, conversation_id: str) -> list[ConversationMessage]:
        return self.get_history(conversation_id)

    def get_history(self, conversation_id: str) -> list[ConversationMessage]:
        if self.max_messages <= 0:
            return []

        normalized_id = self._normalize_conversation_id(conversation_id)
        if not normalized_id:
            return []

        now = self._clock()
        with self._lock:
            self._clear_expired_locked(now)
            session = self._sessions.get(normalized_id)
            if session is None:
                return []
            if self._is_expired(session, now):
                self._sessions.pop(normalized_id, None)
                return []
            session.updated_at = now
            return list(session.messages[-self.max_messages :])

    def add_user_message(self, conversation_id: str, content: str) -> None:
        self.append(conversation_id, "user", content)

    def add_assistant_message(self, conversation_id: str, content: str) -> None:
        self.append(conversation_id, "assistant", content)

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self.append(conversation_id, role, content)

    def append(self, conversation_id: str, role: str, content: str) -> None:
        if self.max_messages <= 0:
            return

        normalized_id = self._normalize_conversation_id(conversation_id)
        normalized_role = role.strip().casefold()
        if normalized_id is None or normalized_role not in {"user", "assistant"}:
            return

        text = self._truncate(content)
        if not text:
            return

        now = self._clock()
        message = ConversationMessage(role=normalized_role, content=text, timestamp=now)
        with self._lock:
            self._clear_expired_locked(now)
            session = self._sessions.setdefault(
                normalized_id,
                _ConversationSession(messages=[], updated_at=now),
            )
            session.messages.append(message)
            session.messages = session.messages[-self.max_messages :]
            session.updated_at = now
            self._enforce_session_limit_locked()

    def remove_session(self, conversation_id: str) -> None:
        normalized_id = self._normalize_conversation_id(conversation_id)
        if not normalized_id:
            return
        with self._lock:
            self._sessions.pop(normalized_id, None)

    def clear_expired_sessions(self) -> int:
        now = self._clock()
        with self._lock:
            before = len(self._sessions)
            self._clear_expired_locked(now)
            return before - len(self._sessions)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _clear_expired_locked(self, now: float) -> None:
        expired_ids = [
            conversation_id
            for conversation_id, session in self._sessions.items()
            if self._is_expired(session, now)
        ]
        for conversation_id in expired_ids:
            self._sessions.pop(conversation_id, None)

    def _enforce_session_limit_locked(self) -> None:
        overflow = len(self._sessions) - self.max_sessions
        if overflow <= 0:
            return
        oldest = sorted(self._sessions.items(), key=lambda item: item[1].updated_at)[:overflow]
        for conversation_id, _session in oldest:
            self._sessions.pop(conversation_id, None)

    def _is_expired(self, session: _ConversationSession, now: float) -> bool:
        return now - session.updated_at > self.ttl_seconds

    def _truncate(self, content: str) -> str:
        text = str(content or "").strip()
        if len(text) <= self.max_message_chars:
            return text
        return text[: self.max_message_chars].rstrip()

    @staticmethod
    def _normalize_conversation_id(conversation_id: str) -> str | None:
        text = str(conversation_id or "").strip()
        if not text or len(text) > 128:
            return None
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
        if any(char not in allowed for char in text):
            return None
        return text


ConversationMemory = InMemoryConversationStore
