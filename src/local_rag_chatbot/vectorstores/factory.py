from __future__ import annotations

from local_rag_chatbot.app_errors import ConfigurationError
from local_rag_chatbot.settings import AppSettings
from local_rag_chatbot.vectorstores.contracts import VectorStoreBackend


def create_vector_store_backend(settings: AppSettings, embeddings: object) -> VectorStoreBackend:
    if settings.vector_store_provider == "chroma":
        from local_rag_chatbot.vectorstores.chroma_backend import ChromaVectorStoreBackend, ChromaVectorStoreSettings

        return ChromaVectorStoreBackend(
            settings=ChromaVectorStoreSettings(
                rag_persist_directory=settings.chroma_db_dir,
                router_persist_directory=settings.router_store_dir,
            ),
            embeddings=embeddings,
        )
    if settings.vector_store_provider == "pgvector":
        from local_rag_chatbot.vectorstores.pgvector_backend import PgVectorStoreBackend, PgVectorStoreSettings

        pg_settings = settings.pgvector_settings()
        return PgVectorStoreBackend(
            settings=PgVectorStoreSettings(
                dsn=pg_settings.dsn,
                schema_name=pg_settings.schema_name,
                table_name=pg_settings.table_name,
                collection_table_name=pg_settings.collection_table_name,
                distance_strategy=pg_settings.distance_strategy,
                index_type=pg_settings.index_type,
                create_extension=pg_settings.create_extension,
                connect_max_attempts=pg_settings.connect_max_attempts,
                connect_retry_seconds=pg_settings.connect_retry_seconds,
                connect_timeout_seconds=pg_settings.connect_timeout_seconds,
            ),
            embeddings=embeddings,
        )
    raise ConfigurationError(f"Unsupported vector store provider: {settings.vector_store_provider}")
