# NebulaDesk API Versions

## Version precedence

When multiple documents describe the same behavior, the section naming the requested API version takes precedence over a general section.

If no version is named, use the current version, which is `v2.4`.

## v2.3

In `v2.3`, the default tenant rate limit was 60 requests per minute.

In `v2.3`, the readiness endpoint did not require authentication.

## v2.4

In `v2.4`, the default tenant rate limit is 120 requests per minute.

In `v2.4`, `GET /v2/health/ready` requires the `X-Nebula-Key` header.

In `v2.4`, the request ID is echoed in both the response and the audit event.

The `v2.4` behavior is the current behavior unless a request explicitly targets `v2.3`.
