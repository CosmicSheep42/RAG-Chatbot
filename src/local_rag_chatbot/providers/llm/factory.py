from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_chatbot.app_errors import ConfigurationError
from local_rag_chatbot.providers.llm.contracts import ChatModel

if TYPE_CHECKING:
    from local_rag_chatbot.settings import AppSettings


def create_chat_model(settings: AppSettings) -> ChatModel:
    if settings.llm_provider == "ollama":
        from local_rag_chatbot.providers.llm.ollama import OllamaChatModel

        return OllamaChatModel(
            base_url=settings.ollama_base_url,
            default_model=settings.generation_model,
            temperature=settings.generation_temperature,
            num_predict=2048,
            connect_timeout_seconds=settings.ollama_connect_timeout_seconds,
            read_timeout_seconds=settings.ollama_read_timeout_seconds,
            max_response_bytes=settings.ollama_max_response_bytes,
        )
    if settings.llm_provider == "openai":
        from local_rag_chatbot.providers.llm.openai import OpenAIChatModel

        return OpenAIChatModel(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            default_model=settings.generation_model,
            temperature=settings.generation_temperature,
            timeout_seconds=settings.request_timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
