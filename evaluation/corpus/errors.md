# NebulaDesk Errors and Troubleshooting

## Authentication errors

`NBL-401` means the `X-Nebula-Key` header is missing, malformed, or expired.

`NBL-403` means the API key is valid but lacks permission for the requested resource.

For `NBL-401`, obtain a new key for the same tenant and send it in the `X-Nebula-Key` header.

## Capacity errors

`NBL-429` means the tenant rate limit was exceeded; the response includes a `Retry-After` header in seconds.

`NBL-503` means a temporary service dependency is unavailable and the request may be retried.

Do not retry `NBL-413`, because it means the request body exceeded the 2 MiB limit.

## Incident procedure

When an incident is declared, capture the `X-Request-Id`, UTC timestamp, tenant identifier, and response code before opening a support case.

Support cases for data loss require the affected resource identifier and the last successful request ID.
