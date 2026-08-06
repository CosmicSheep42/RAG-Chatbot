from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

from local_rag_chatbot.resource_refs import ConfigResource


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: str


@dataclass(frozen=True)
class ToolCallingConfig:
    answer_prompt_template: str
    system_prompt_template: str
    tools: list[ToolDefinition]


class ToolCallingConfigRepository:
    def __init__(self, config_file: ConfigResource | Traversable | Path) -> None:
        self.config_file = config_file

    def load(self) -> ToolCallingConfig:
        config_file = self.config_file.location if isinstance(self.config_file, ConfigResource) else self.config_file
        with config_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid tool calling config format: {self.config_file}")

        answer_prompt_template = data.get("answer_prompt_template")
        if not isinstance(answer_prompt_template, str) or not answer_prompt_template.strip():
            raise ValueError(f"Missing answer_prompt_template in {self.config_file}")

        system_prompt_template = data.get("system_prompt_template")
        if not isinstance(system_prompt_template, str) or not system_prompt_template.strip():
            raise ValueError(f"Missing system_prompt_template in {self.config_file}")

        raw_tools = data.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError(f"Missing tools list in {self.config_file}")

        tools: list[ToolDefinition] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                raise ValueError(f"Invalid tool entry in {self.config_file}")

            name = raw_tool.get("name")
            description = raw_tool.get("description")
            parameters = raw_tool.get("parameters")
            executor = raw_tool.get("executor")

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Tool without valid name in {self.config_file}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Tool '{name}' missing description in {self.config_file}")
            if not isinstance(parameters, dict):
                raise ValueError(f"Tool '{name}' missing parameters schema in {self.config_file}")
            if not isinstance(executor, str) or not executor.strip():
                raise ValueError(f"Tool '{name}' missing executor in {self.config_file}")

            tools.append(
                ToolDefinition(
                    name=name.strip(),
                    description=description.strip(),
                    parameters=parameters,
                    executor=executor.strip(),
                )
            )

        return ToolCallingConfig(
            answer_prompt_template=answer_prompt_template.strip(),
            system_prompt_template=system_prompt_template.strip(),
            tools=tools,
        )
