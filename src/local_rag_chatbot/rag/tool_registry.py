from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import logging
import re
from typing import Any, Callable, Mapping

from local_rag_chatbot.documents.tool_config_repository import ToolDefinition
from local_rag_chatbot.request_context import RequestContext


ToolHandler = Callable[
    [ToolDefinition, Mapping[str, Any]],
    tuple[str, list[Any], dict[str, Any]],
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler
    normalized_integer_fields: frozenset[str] = frozenset()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
        *,
        normalized_integer_fields: frozenset[str] = frozenset(),
    ) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered.")
        self._validate_handler_signature(definition.name, handler)
        self._tools[definition.name] = RegisteredTool(
            definition=definition,
            handler=handler,
            normalized_integer_fields=normalized_integer_fields,
        )

    def available(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_context: RequestContext | None = None,
    ) -> tuple[str, list[Any], dict[str, Any]]:
        tool = self.get(name)
        if tool is None:
            payload = {"status": "validation_error", "error": f"Unknown tool '{name}'."}
            return json.dumps(payload, ensure_ascii=True), [], payload
        normalized_arguments = self._normalize_arguments(tool, arguments)
        validation_error = self._validate_arguments(tool.definition, normalized_arguments)
        if validation_error is not None:
            payload = {"status": "validation_error", "error": validation_error}
            return json.dumps(payload, ensure_ascii=True), [], payload
        try:
            content, documents, payload = tool.handler(
                tool.definition,
                normalized_arguments,
                request_context=request_context,
            )
            status = payload.get("status")
            if status not in {"validation_error", "execution_error"}:
                status = "execution_error" if "error" in payload else "success" if documents else "success_empty"
            payload = {**payload, "status": status}
            return json.dumps(payload, ensure_ascii=True), documents, payload
        except Exception:
            logger.exception("Tool execution failed", extra={"tool_name": name})
            payload = {"status": "execution_error", "error": f"Tool '{name}' failed."}
            return json.dumps(payload, ensure_ascii=True), [], payload

    def _normalize_arguments(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return arguments
        normalized = dict(arguments)
        properties = tool.definition.parameters.get("properties", {})
        for field_name in tool.normalized_integer_fields:
            schema = properties.get(field_name) if isinstance(properties, dict) else None
            if not isinstance(schema, dict) or schema.get("type") != "integer":
                continue
            if field_name not in normalized and "default" in schema:
                normalized[field_name] = schema["default"]
            value = normalized.get(field_name)
            if isinstance(value, str) and re.fullmatch(r"-?(?:0|[1-9]\d*)", value.strip()):
                value = int(value.strip())
            if isinstance(value, int) and not isinstance(value, bool):
                maximum = schema.get("maximum")
                if isinstance(maximum, int) and not isinstance(maximum, bool):
                    value = min(value, maximum)
                normalized[field_name] = value
        return normalized

    def _validate_handler_signature(self, name: str, handler: ToolHandler) -> None:
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Tool '{name}' handler has an unsupported signature.") from exc

        parameters = signature.parameters
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_var_positional = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()
        )
        has_var_keyword = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        accepts_request_context = "request_context" in parameters or has_var_keyword
        if len(positional) < 2 and not has_var_positional:
            raise ValueError(f"Tool '{name}' handler must accept definition and arguments parameters.")
        if not accepts_request_context:
            raise ValueError(f"Tool '{name}' handler must accept a request_context keyword argument.")

    def _validate_arguments(self, definition: ToolDefinition, arguments: dict[str, Any]) -> str | None:
        if not isinstance(arguments, dict):
            return f"Arguments for tool '{definition.name}' must be an object."

        schema = definition.parameters or {}
        required = schema.get("required", [])
        if isinstance(required, list):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in arguments:
                    return f"Missing required argument '{field_name}'."

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return None

        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            for field_name in arguments:
                if field_name not in properties:
                    return f"Unknown argument '{field_name}' for tool '{definition.name}'."

        for field_name, value in arguments.items():
            property_schema = properties.get(field_name)
            if not isinstance(property_schema, dict):
                continue
            expected_type = property_schema.get("type")
            if expected_type and not self._matches_type(value, expected_type):
                return f"Argument '{field_name}' must be of type {expected_type}."
            minimum_length = property_schema.get("minLength")
            if isinstance(value, str) and isinstance(minimum_length, int) and len(value) < minimum_length:
                return f"Argument '{field_name}' must contain at least {minimum_length} character(s)."
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                minimum = property_schema.get("minimum")
                maximum = property_schema.get("maximum")
                if isinstance(minimum, (int, float)) and value < minimum:
                    return f"Argument '{field_name}' must be >= {minimum}."
                if isinstance(maximum, (int, float)) and value > maximum:
                    return f"Argument '{field_name}' must be <= {maximum}."
        return None

    def _matches_type(self, value: Any, expected_type: Any) -> bool:
        if isinstance(expected_type, list):
            return any(self._matches_type(value, item) for item in expected_type)
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        return True
