from __future__ import annotations

from typing import Any


class RoutingOrderResolver:
    def __init__(self, default_priorities: dict[str, int] | None = None) -> None:
        self.default_priorities = default_priorities or {
            "deny": 1,
            "simple": 2,
            "faq": 3,
        }

    def resolve(
        self,
        configs: dict[str, dict[str, Any]],
        priority_overrides: dict[str, int] | None = None,
    ) -> list[str]:
        overrides = priority_overrides or {}

        def sort_key(category: str) -> tuple[int, str]:
            priority = overrides.get(category, self.default_priorities.get(category, 1000))
            return int(priority), category

        return sorted(configs.keys(), key=sort_key)
