# Security Model

The project is designed for local demonstrations and embedding into trusted Python applications.

The standalone application is intended for local demonstration and trusted environments. Do not expose it directly to the public internet. It does not implement chatbot-managed authentication.

Embedded deployments must use the host application's authentication, authorization, tenant isolation, and user-level rate limiting. The chatbot receives only an optional `conversation_id` for temporary memory after the host has already resolved access. A conversation ID is not trusted identity and must not grant permissions.

Do not expose Ollama, Chroma files, or PostgreSQL/pgvector directly to untrusted networks. The Docker Compose PGVector
service is internal by default and does not publish port `5432` to the host. Do not commit `.env`, secrets, generated
indexes, pickles, local database files, or PostgreSQL DSNs with credentials.

Vector-store fingerprints and logs must not include full DSNs, passwords, tokens, or credential-bearing URLs. The PGVector
adapter validates schema and table identifiers with a conservative SQL identifier pattern and uses parameterized queries
for values.

The document ingestion layer validates size and archive limits, but knowledge-base content should still be reviewed before public demonstrations.
