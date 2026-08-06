from __future__ import annotations


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


class IndexError(AppError):
    status_code = 503
    public_message = "Document index is unavailable."


class ToolExecutionError(AppError):
    status_code = 503
    public_message = "Tool execution failed."
