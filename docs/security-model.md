# Security Model

The project is designed for local demonstrations and embedding into trusted Python applications.

The standalone application is intended for local demonstration and trusted environments. Do not expose it directly to the public internet. It does not implement chatbot-managed authentication.

Embedded deployments must use the host application's authentication, authorization, tenant isolation, and user-level rate limiting. The chatbot receives only an optional `conversation_id` for temporary memory after the host has already resolved access. A conversation ID is not trusted identity and must not grant permissions.

Do not expose Ollama or Chroma directly to untrusted networks. Do not commit `.env`, secrets, generated indexes, pickles, or local database files.

The document ingestion layer validates size and archive limits, but knowledge-base content should still be reviewed before public demonstrations.
