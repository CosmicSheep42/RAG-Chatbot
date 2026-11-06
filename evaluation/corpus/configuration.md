# NebulaDesk Configuration Reference

## Service mode

`NEBULA_MODE` selects the service mode and defaults to `balanced` when unset.

The `diagnostic` mode includes additional timing fields in responses but does not change rate limits.

## Client behavior

`NEBULA_TIMEOUT_SECONDS` defaults to 10 seconds.

The client retries a transient `503` response at most three times.

Retries use delays of 1, 2, and 4 seconds and are not used for `401` or `403` responses.

## Cache

`NEBULA_CACHE_TTL_SECONDS` defaults to 300 seconds.

Setting the cache TTL to `0` disables cache reads and writes.

Cache entries are isolated by tenant and API version.
