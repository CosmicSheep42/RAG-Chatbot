from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    conversation_id: str | None = None

    @property
    def session_id(self) -> str | None:
        return self.conversation_id
