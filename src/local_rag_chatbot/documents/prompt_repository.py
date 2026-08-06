from __future__ import annotations

import json
from importlib.abc import Traversable
from pathlib import Path

from local_rag_chatbot.resource_refs import ConfigResource


class MasterPromptRepository:
    def __init__(self, prompt_file: ConfigResource | Traversable | Path) -> None:
        self.prompt_file = prompt_file

    def load(self) -> str:
        prompt_file = self.prompt_file.location if isinstance(self.prompt_file, ConfigResource) else self.prompt_file
        with prompt_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            prompt = data.get("system_prompt") or data.get("prompt") or data.get("text")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()

        raise ValueError(f"Invalid master prompt format: {self.prompt_file}")
