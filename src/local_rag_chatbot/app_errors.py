from __future__ import annotations

from local_rag_chatbot.providers.errors import (
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class AppError(Exception):
    status_code = 500
    public_message = "Internal server error."

    def __init__(self, message: str | None = None, *, public_message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        if public_message is not None:
            self.public_message = public_message


class ConfigurationError(AppError):
    status_code = 500
    public_message = "Invalid server configuration."


class AuthenticationError(AppError):
    status_code = 401
    public_message = "Authentication required."


class AuthorizationError(AppError):
    status_code = 403
    public_message = "Access denied."


class DocumentIngestionError(AppError):
    status_code = 400
    public_message = "Document could not be ingested."


class PayloadTooLargeError(AppError):
    status_code = 413
    public_message = "Payload too large."


class RetrievalError(AppError):
    status_code = 503
    public_message = "Retrieval is unavailable."


class ModelUnavailableError(AppError):
    status_code = 503
    public_message = "Model service is unavailable."


class ModelTimeoutError(AppError):
    status_code = 504
    public_message = "Model request timed out."


class ModelRateLimitedError(AppError):
    status_code = 429
    public_message = "Model provider rate limit exceeded."

    def __init__(
        self,
        message: str | None = None,
        *,
        public_message: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, public_message=public_message)
        self.retry_after = retry_after


class IndexError(AppError):
    status_code = 503
    public_message = "Document index is unavailable."


class ToolExecutionError(AppError):
    status_code = 503
    public_message = "Tool execution failed."


def map_provider_error(exc: ProviderError) -> AppError:
    """Translate a provider-adapter error into the application's HTTP-facing error hierarchy.

    This is the single boundary where the OpenAI/Ollama distinction disappears: callers
    (the RAG pipeline) only ever need to know about `ProviderError`, and routes only ever
    need to know about `AppError`.
    """
    if isinstance(exc, ProviderTimeoutError):
        return ModelTimeoutError(str(exc))
    if isinstance(exc, ProviderRateLimitError):
        return ModelRateLimitedError(str(exc), retry_after=exc.retry_after)
    if isinstance(exc, ProviderConnectionError):
        return ModelUnavailableError(str(exc))
    return ModelUnavailableError(str(exc))
