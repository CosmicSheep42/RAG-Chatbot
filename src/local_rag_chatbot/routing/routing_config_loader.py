from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

from local_rag_chatbot.resource_refs import ConfigResource


@dataclass(frozen=True)
class RoutingConfigBundle:
    configs: dict[str, dict[str, Any]]
    priority_overrides: dict[str, int]


class RoutingConfigLoader:
    def __init__(self, routing_dir: ConfigResource | Traversable | Path) -> None:
        self.routing_dir = routing_dir

    def load(self) -> RoutingConfigBundle:
        routing_dir = self.routing_dir.location if isinstance(self.routing_dir, ConfigResource) else self.routing_dir
        if not routing_dir.exists():
            raise FileNotFoundError(f"Routing directory not found: {routing_dir}")

        configs: dict[str, dict[str, Any]] = {}
        priority_overrides = self._load_priority_overrides(routing_dir)

        for category_dir in sorted(routing_dir.iterdir(), key=lambda item: item.name):
            if not category_dir.is_dir():
                continue

            config = self._load_category(category_dir)
            if config["intents"]:
                configs[category_dir.name] = config

        return RoutingConfigBundle(configs=configs, priority_overrides=priority_overrides)

    def _load_priority_overrides(self, routing_dir: Traversable | Path) -> dict[str, int]:
        overrides: dict[str, int] = {}

        root_file = routing_dir / "priorities.json"
        if root_file.exists():
            with root_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            overrides.update(self._extract_priority_map(data))

        return overrides

    def _extract_priority_map(self, data: Any) -> dict[str, int]:
        if not isinstance(data, dict):
            return {}

        if isinstance(data.get("priorities"), dict):
            source = data["priorities"]
        elif "intents" not in data:
            source = data
        else:
            return {}

        overrides: dict[str, int] = {}
        for category, value in source.items():
            try:
                overrides[str(category)] = int(value)
            except (TypeError, ValueError):
                continue
        return overrides

    def _load_category(self, category_dir: Traversable | Path) -> dict[str, Any]:
        intents: list[dict[str, Any]] = []
        thresholds: list[float] = []

        json_files = (item for item in category_dir.iterdir() if item.is_file() and item.name.endswith(".json"))
        for json_file in sorted(json_files, key=lambda item: item.name):
            with json_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            if "intents" not in data or not isinstance(data["intents"], list):
                raise ValueError(f"Invalid routing config format: {json_file}")

            intents.extend(data["intents"])

            if "threshold" in data:
                try:
                    thresholds.append(float(data["threshold"]))
                except Exception:
                    pass

        config: dict[str, Any] = {
            "intents": intents,
            "threshold": max(thresholds) if thresholds else 0.80,
        }

        return config
