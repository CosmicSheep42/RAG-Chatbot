# NebulaDesk Platform Overview

## API identity

NebulaDesk is the fictional service covered by this corpus. The public API base URL is `/v2`.

Every authenticated request must include the API key in the `X-Nebula-Key` header.

The API accepts JSON request bodies and returns JSON responses.

## Request controls

The default tenant rate limit is 120 requests per minute.

Each request may include an `X-Request-Id` value, which is echoed in the response and in the audit event.

The service rejects request bodies larger than 2 MiB before authentication is evaluated.

## Accounts

A tenant can have at most 50 active API keys.

API keys are scoped to one tenant and cannot be used to access another tenant's resources.
