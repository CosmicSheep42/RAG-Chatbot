from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for provider-adapter errors.

    Adapters must translate provider-specific exceptions (httpx errors, the
    OpenAI SDK's exception hierarchy, etc.) into this hierarchy so no
    provider-specific exception type ever crosses the adapter boundary.
    """


class ProviderConnectionError(ProviderError):
    """The provider could not be reached (network, DNS, connection refused)."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the configured timeout."""


class ProviderAuthenticationError(ProviderError):
    """The provider rejected the request due to missing/invalid credentials."""


class ProviderRateLimitError(ProviderError):
    """The provider rejected the request due to rate limiting."""

    def __init__(self, message: str | None = None, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderResponseError(ProviderError):
    """The provider returned an HTTP error status or a malformed/invalid response."""
