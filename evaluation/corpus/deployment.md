# NebulaDesk Deployment Guide

## Runtime requirements

The supported runtime is Python 3.11 or newer on a 64-bit Linux host.

A single application worker requires at least 512 MiB of available memory.

The application listens on port 8443 by default.

## Health checks

The liveness endpoint is `GET /v2/health/live` and does not require authentication.

The readiness endpoint is `GET /v2/health/ready` and requires the `X-Nebula-Key` header.

Readiness is false while the primary datastore connection is unavailable.

## Rollouts

A rolling deployment must keep at least two healthy workers before the old worker is stopped.

The deployment controller waits 30 seconds after a worker becomes ready before routing tenant traffic to it.
