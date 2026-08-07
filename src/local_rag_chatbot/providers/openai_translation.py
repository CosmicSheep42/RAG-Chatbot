from __future__ import annotations

from local_rag_chatbot.providers.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)


def translate_openai_error(exc: Exception, *, context: str) -> ProviderError:
    """Translate an OpenAI SDK exception into the provider-neutral error hierarchy.

    Shared by the chat and embeddings adapters so both stay in sync as the SDK's
    exception hierarchy evolves.
    """
    import openai as openai_sdk

    if isinstance(exc, openai_sdk.APITimeoutError):
        return ProviderTimeoutError(f"{context} timed out: {exc}")
    if isinstance(exc, openai_sdk.APIConnectionError):
        return ProviderConnectionError(f"Could not reach the OpenAI API: {exc}")
    if isinstance(exc, openai_sdk.AuthenticationError):
        return ProviderAuthenticationError("OpenAI authentication failed. Check OPENAI_API_KEY.")
    if isinstance(exc, openai_sdk.RateLimitError):
        return ProviderRateLimitError(f"OpenAI rate limit exceeded: {exc}", retry_after=_extract_retry_after(exc))
    if isinstance(exc, openai_sdk.APIStatusError):
        return ProviderResponseError(f"{context} failed with status {exc.status_code}: {exc}")
    if isinstance(exc, openai_sdk.APIError):
        return ProviderError(f"{context} failed: {exc}")
    return ProviderError(f"{context} failed: {exc}")


def _extract_retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
