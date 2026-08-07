from __future__ import annotations

import json
from typing import Any

from local_rag_chatbot.providers.errors import ProviderAuthenticationError, ProviderError
from local_rag_chatbot.providers.llm.types import ChatResult, ToolCall, ToolSpec
from local_rag_chatbot.providers.openai_translation import translate_openai_error


class OpenAIChatModel:
    """OpenAI adapter for the provider-neutral `ChatModel` contract.

    Uses the official `openai` SDK's Responses API. Callers still pass
    Chat-Completions-shaped messages (the format already used by
    `RAGChatbot`/`AnswerService`/`QueryRewriter`): `role` in
    system/user/assistant/tool, assistant messages carry `tool_calls`, tool
    result messages carry `tool_call_id`. This adapter is the only place
    that translates that shape into Responses API input items and back.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_model: str,
        temperature: float,
        max_output_tokens: int = 2048,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ProviderAuthenticationError("OPENAI_API_KEY is required to use the OpenAI provider.")

        from openai import AsyncOpenAI, OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.default_model = default_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        payload = self._build_payload(messages, tools, model=model)
        try:
            response = self._client.responses.create(**payload)
        except Exception as exc:
            raise self._translate_error(exc) from exc
        return self._parse_response(response)

    async def achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        payload = self._build_payload(messages, tools, model=model)
        try:
            response = await self._async_client.responses.create(**payload)
        except Exception as exc:
            raise self._translate_error(exc) from exc
        return self._parse_response(response)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        *,
        model: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "input": self._translate_messages(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if tools:
            payload["tools"] = [self._serialize_tool(tool) for tool in tools]
        return payload

    def _translate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": content if isinstance(content, str) else json.dumps(content),
                    }
                )
                continue

            tool_calls = message.get("tool_calls") if role == "assistant" else None
            if tool_calls:
                if content:
                    items.append({"role": "assistant", "content": content})
                for tool_call in tool_calls:
                    function_payload = tool_call.get("function", {}) or {}
                    arguments = function_payload.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments or {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(tool_call.get("id") or ""),
                            "name": str(function_payload.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
                continue

            items.append({"role": role, "content": content if content is not None else ""})
        return items

    def _serialize_tool(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    def _parse_response(self, response: Any) -> ChatResult:
        content = str(getattr(response, "output_text", "") or "").strip()
        tool_calls: list[ToolCall] = []
        for index, item in enumerate(getattr(response, "output", None) or [], start=1):
            if getattr(item, "type", None) != "function_call":
                continue
            name = str(getattr(item, "name", "") or "").strip()
            if not name:
                continue
            call_id = str(getattr(item, "call_id", "") or f"tool_call_{index}")
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=self._coerce_arguments(getattr(item, "arguments", None)),
                )
            )
        return ChatResult(content=content, tool_calls=tool_calls)

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return {"raw_arguments": stripped}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _translate_error(self, exc: Exception) -> ProviderError:
        return translate_openai_error(exc, context="OpenAI request")
