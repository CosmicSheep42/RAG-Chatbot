from __future__ import annotations

from local_rag_chatbot.providers.errors import ProviderAuthenticationError, ProviderError
from local_rag_chatbot.providers.openai_translation import translate_openai_error


class OpenAIEmbeddings:
    """OpenAI adapter for the provider-neutral `Embeddings` contract.

    Uses the official `openai` SDK. Exposes `.model` so
    `IndexManager._embedding_model_identifier()` (duck-typed `getattr`)
    keeps reporting the correct model name in the index fingerprint.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ProviderAuthenticationError("OPENAI_API_KEY is required to use the OpenAI provider.")

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise self._translate_error(exc) from exc
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _translate_error(self, exc: Exception) -> ProviderError:
        return translate_openai_error(exc, context="OpenAI embeddings request")
