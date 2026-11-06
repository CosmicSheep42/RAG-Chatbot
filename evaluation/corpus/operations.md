# NebulaDesk Operations Manual

## Backups

NebulaDesk creates an encrypted backup every six hours.

Backups are retained for 14 days before automatic deletion.

Restore tests run on the first Sunday of every month and use a quarantined tenant.

## Audit exports

An audit export contains request IDs, tenant IDs, response codes, and UTC timestamps.

Audit exports are generated asynchronously and expire 24 hours after completion.

The export endpoint returns a job identifier immediately; clients must poll the job status endpoint.

## Data residency

Tenant data is stored in the region selected at tenant creation time.

Changing a tenant's region requires a support-assisted migration and a new backup verification.

The migration does not change API keys or tenant identifiers.

## Large-account procedure

Tenants with more than 40 active API keys should request a capacity review before adding keys.

The capacity review is advisory and does not raise the 50-key account limit.
