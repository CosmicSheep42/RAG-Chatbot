from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: list[ToolCall]


class OllamaClientError(RuntimeError):
    pass


class OllamaConnectionError(OllamaClientError):
    pass


class OllamaTimeoutError(OllamaClientError):
    pass


class OllamaHTTPStatusError(OllamaClientError):
    pass


class OllamaInvalidResponseError(OllamaClientError):
    pass


class OllamaToolCallingClient:
    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        temperature: float,
        num_predict: int = 2048,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 60.0,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.temperature = temperature
        self.num_predict = num_predict
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.max_response_bytes = max_response_bytes

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        payload = self._build_payload(messages, tools, model=model)
        timeout = self._timeout()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectTimeout as exc:
            raise OllamaTimeoutError(f"Timed out connecting to Ollama at {self.base_url}.") from exc
        except httpx.ReadTimeout as exc:
            raise OllamaTimeoutError(f"Timed out reading from Ollama at {self.base_url}.") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Could not reach Ollama at {self.base_url}.") from exc
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"Ollama request failed: {exc}") from exc

        return self._parse_response(response)

    async def achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        payload = self._build_payload(messages, tools, model=model)
        timeout = self._timeout()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectTimeout as exc:
            raise OllamaTimeoutError(f"Timed out connecting to Ollama at {self.base_url}.") from exc
        except httpx.ReadTimeout as exc:
            raise OllamaTimeoutError(f"Timed out reading from Ollama at {self.base_url}.") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Could not reach Ollama at {self.base_url}.") from exc
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"Ollama request failed: {exc}") from exc

        return self._parse_response(response)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if tools:
            payload["tools"] = [self._serialize_tool(tool) for tool in tools]
        return payload

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=None,
            connect=self.connect_timeout_seconds,
            read=self.read_timeout_seconds,
            write=self.connect_timeout_seconds,
            pool=self.connect_timeout_seconds,
        )

    def _parse_response(self, response: httpx.Response) -> ChatResult:
        raw_bytes = response.content
        if len(raw_bytes) > self.max_response_bytes:
            raise OllamaInvalidResponseError(
                f"Ollama response exceeded {self.max_response_bytes} bytes."
            )
        if not raw_bytes:
            raise OllamaInvalidResponseError("Ollama returned an empty response body.")
        if response.status_code >= 400:
            detail = raw_bytes[:1024].decode("utf-8", errors="replace")
            raise OllamaHTTPStatusError(
                f"Ollama chat request failed with status {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaInvalidResponseError("Ollama returned malformed JSON.") from exc

        if not isinstance(data, dict):
            raise OllamaInvalidResponseError("Ollama returned a non-object JSON response.")
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise OllamaInvalidResponseError("Ollama response message is invalid.")
        content = str(message.get("content") or "").strip()
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        return ChatResult(content=content, tool_calls=tool_calls)

    def _serialize_tool(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        if not isinstance(raw_tool_calls, list):
            return []

        parsed_calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_tool_calls, start=1):
            if not isinstance(raw_call, dict):
                continue

            function_payload = raw_call.get("function")
            if isinstance(function_payload, dict):
                name = str(function_payload.get("name") or "").strip()
                arguments = function_payload.get("arguments")
            else:
                name = str(raw_call.get("name") or "").strip()
                arguments = raw_call.get("arguments")

            if not name:
                continue

            call_id = str(raw_call.get("id") or f"tool_call_{index}")
            parsed_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=self._coerce_arguments(arguments),
                )
            )
        return parsed_calls

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
