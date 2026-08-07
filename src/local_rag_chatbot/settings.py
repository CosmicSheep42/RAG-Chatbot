from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import os
import re
import warnings
from urllib.parse import urlparse
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None

from local_rag_chatbot.app_errors import ConfigurationError
from local_rag_chatbot.resource_refs import ConfigResource, resource_from_optional_path

DEFAULT_RUNTIME_ROOT = Path.home() / ".local" / "share" / "local-rag-chatbot"
SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PgVectorStoreSettings:
    dsn: str
    schema_name: str
    table_name: str
    collection_table_name: str
    distance_strategy: str
    index_type: str
    create_extension: bool
    connect_max_attempts: int = 12
    connect_retry_seconds: float = 2.0
    connect_timeout_seconds: int = 5


@dataclass(frozen=True)
class AppSettings:
    project_root: Path | None
    documents_dir: Path
    data_dir: Path
    web_dir: Path | None
    vector_store_provider: str
    chroma_db_dir: Path
    router_store_dir: Path
    pgvector_dsn: str
    pgvector_schema: str
    pgvector_table: str
    pgvector_collection_table: str
    pgvector_distance_strategy: str
    pgvector_index_type: str
    pgvector_create_extension: bool
    pgvector_connect_max_attempts: int
    pgvector_connect_retry_seconds: float
    pgvector_connect_timeout_seconds: int
    routing_dir: ConfigResource
    master_prompt_file: ConfigResource
    tool_calling_config_file: ConfigResource
    llm_provider: str
    embedding_provider: str
    generation_model: str
    embedding_model: str
    ollama_base_url: str
    openai_api_key: str
    openai_base_url: str
    chunk_size: int
    chunk_overlap: int
    max_document_file_size_bytes: int
    server_host: str
    server_port: int
    cleaner_remove_emojis: bool
    cleaner_expand_abbreviations: bool
    cleaner_normalize_unicode: bool
    cleaner_remove_extra_spaces: bool
    cleaner_lowercase: bool
    max_input_chars: int 
    rate_limit_requests: int
    rate_limit_window_seconds: int
    inference_timeout_seconds: float
    hybrid_dense_k: int
    hybrid_dense_fetch_k: int
    hybrid_sparse_k: int
    hybrid_candidate_k: int
    hybrid_final_k: int
    hybrid_dense_lambda_mult: float
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    metadata_boost_weight: float
    reranking_mode: str
    reranker_model: str
    reranker_allow_lexical_fallback: bool
    retrieval_confidence_threshold: float
    generation_temperature: float
    tool_calling_enabled: bool
    tool_calling_model: str
    tool_calling_max_rounds: int
    conversation_memory_enabled: bool
    conversation_max_messages: int
    conversation_ttl_seconds: int
    conversation_max_sessions: int
    conversation_max_message_chars: int
    query_rewriting_enabled: bool
    query_rewriting_model: str
    query_rewriting_use_llm: bool
    query_rewriting_timeout_seconds: float
    ollama_connect_timeout_seconds: float
    ollama_read_timeout_seconds: float
    ollama_max_response_bytes: int
    max_concurrent_inferences: int
    max_pending_inferences: int
    request_timeout_seconds: float
    rate_limit_max_buckets: int
    trusted_proxy_hosts: tuple[str, ...]
    app_env: str
    log_level: str
    query_debug_log_content: bool
    max_zip_entries: int
    max_zip_uncompressed_bytes: int
    max_zip_compression_ratio: float
    max_pdf_pages: int
    max_spreadsheet_sheets: int
    max_spreadsheet_rows: int
    max_presentation_slides: int
    auto_index_on_startup: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        return get_settings()

    def with_overrides(self, **overrides: Any) -> "AppSettings":
        overrides = self._normalize_resource_overrides(overrides)
        updated = replace(self, **overrides)
        updated.validate()
        return updated

    def _normalize_resource_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(overrides)
        resource_specs = {
            "master_prompt_file": {
                "packaged_parts": ("prompts", "master_prompt.json"),
                "description": "master prompt file",
                "kind": "file",
            },
            "tool_calling_config_file": {
                "packaged_parts": ("prompts", "tool_calling.json"),
                "description": "tool calling config file",
                "kind": "file",
            },
            "routing_dir": {
                "packaged_parts": ("routing",),
                "description": "routing directory",
                "kind": "dir",
            },
        }
        for field_name, spec in resource_specs.items():
            if field_name in normalized:
                normalized[field_name] = resource_from_optional_path(
                    normalized[field_name],
                    env_value=None,
                    packaged_parts=spec["packaged_parts"],
                    description=spec["description"],
                    kind=spec["kind"],
                )
        for field_name in ("documents_dir", "data_dir", "web_dir", "chroma_db_dir", "router_store_dir"):
            if field_name in normalized and normalized[field_name] is not None:
                normalized[field_name] = Path(normalized[field_name]).expanduser()
        for field_name in ("vector_store_provider", "llm_provider", "embedding_provider"):
            if field_name in normalized and normalized[field_name] is not None:
                normalized[field_name] = str(normalized[field_name]).strip().lower()
        for field_name in (
            "pgvector_schema",
            "pgvector_table",
            "pgvector_collection_table",
            "pgvector_distance_strategy",
            "pgvector_index_type",
        ):
            if field_name in normalized and normalized[field_name] is not None:
                normalized[field_name] = str(normalized[field_name]).strip().lower()
        return normalized

    def input_cleaner_config(self) -> dict[str, Any]:
        return {
            "preprocessor_config": {
                "remove_emojis": self.cleaner_remove_emojis,
                "expand_abbreviations": self.cleaner_expand_abbreviations,
                "normalize_unicode": self.cleaner_normalize_unicode,
                "remove_extra_spaces": self.cleaner_remove_extra_spaces,
                "lowercase": self.cleaner_lowercase,
            },
        }

    def validate(self) -> None:
        errors: list[str] = []
        if self.app_env not in {"dev", "test", "prod"}:
            errors.append("APP_ENV must be one of dev, test, prod.")
        if not self.vector_store_provider:
            errors.append("VECTOR_STORE_PROVIDER must not be empty.")
        if self.vector_store_provider not in {"chroma", "pgvector"}:
            errors.append(
                f"Unsupported VECTOR_STORE_PROVIDER: {self.vector_store_provider}. Supported values: chroma, pgvector."
            )
        if self.vector_store_provider == "pgvector" and not self.pgvector_dsn.strip():
            errors.append("PGVECTOR_DSN is required when VECTOR_STORE_PROVIDER=pgvector.")
        if self.llm_provider not in {"ollama", "openai"}:
            errors.append(f"Unsupported LLM_PROVIDER: {self.llm_provider}. Supported values: ollama, openai.")
        if self.embedding_provider not in {"ollama", "openai"}:
            errors.append(
                f"Unsupported EMBEDDING_PROVIDER: {self.embedding_provider}. Supported values: ollama, openai."
            )
        if (self.llm_provider == "openai" or self.embedding_provider == "openai") and not self.openai_api_key.strip():
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER or EMBEDDING_PROVIDER is openai.")
        for field_name in ("pgvector_schema", "pgvector_table", "pgvector_collection_table"):
            value = getattr(self, field_name)
            if not SQL_IDENTIFIER_PATTERN.match(value):
                errors.append(f"{field_name.upper()} must match {SQL_IDENTIFIER_PATTERN.pattern}.")
        if self.pgvector_distance_strategy not in {"cosine", "l2", "inner_product"}:
            errors.append("PGVECTOR_DISTANCE_STRATEGY must be one of: cosine, l2, inner_product.")
        if self.pgvector_index_type not in {"none", "hnsw", "ivfflat"}:
            errors.append("PGVECTOR_INDEX_TYPE must be one of: none, hnsw, ivfflat.")
        if self.pgvector_connect_max_attempts <= 0:
            errors.append("PGVECTOR_CONNECT_MAX_ATTEMPTS must be > 0.")
        if self.pgvector_connect_retry_seconds < 0:
            errors.append("PGVECTOR_CONNECT_RETRY_SECONDS must be >= 0.")
        if self.pgvector_connect_timeout_seconds <= 0:
            errors.append("PGVECTOR_CONNECT_TIMEOUT_SECONDS must be > 0.")
        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be > 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE.")
        for name in ("hybrid_dense_k", "hybrid_sparse_k", "hybrid_candidate_k", "hybrid_final_k"):
            if getattr(self, name) <= 0:
                errors.append(f"{name.upper()} must be > 0.")
        if self.hybrid_dense_fetch_k < self.hybrid_dense_k:
            errors.append("HYBRID_DENSE_FETCH_K must be >= HYBRID_DENSE_K.")
        for name in ("hybrid_dense_lambda_mult", "hybrid_dense_weight", "hybrid_sparse_weight", "metadata_boost_weight"):
            value = getattr(self, name)
            if value < 0 or value > 1:
                errors.append(f"{name.upper()} must be between 0 and 1.")
        if self.reranking_mode not in {"auto", "true", "false"}:
            errors.append(
                f"Invalid RERANKING_MODE value: {self.reranking_mode}. "
                "Expected one of: auto, true, false."
            )
        if self.retrieval_confidence_threshold < 0 or self.retrieval_confidence_threshold > 1:
            errors.append("RETRIEVAL_CONFIDENCE_THRESHOLD must be between 0 and 1.")
        for name in (
            "inference_timeout_seconds",
            "request_timeout_seconds",
            "query_rewriting_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                errors.append(f"{name.upper()} must be > 0.")
        if self.query_rewriting_timeout_seconds > self.request_timeout_seconds:
            errors.append("QUERY_REWRITING_TIMEOUT_SECONDS must be <= REPLY_TIMEOUT_SECONDS.")
        uses_ollama = self.llm_provider == "ollama" or self.embedding_provider == "ollama"
        if uses_ollama:
            for name in ("ollama_connect_timeout_seconds", "ollama_read_timeout_seconds"):
                if getattr(self, name) <= 0:
                    errors.append(f"{name.upper()} must be > 0.")
            if self.ollama_connect_timeout_seconds > self.request_timeout_seconds:
                errors.append("OLLAMA_CONNECT_TIMEOUT_SECONDS must be <= REPLY_TIMEOUT_SECONDS.")
            if self.ollama_read_timeout_seconds > self.request_timeout_seconds:
                errors.append("OLLAMA_READ_TIMEOUT_SECONDS must be <= REPLY_TIMEOUT_SECONDS.")
            parsed = urlparse(self.ollama_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append("OLLAMA_BASE_URL must be an http(s) URL.")
        if self.max_document_file_size_bytes <= 0:
            errors.append("MAX_DOCUMENT_FILE_SIZE_BYTES must be > 0.")
        if not 1 <= self.server_port <= 65535:
            errors.append("SERVER_PORT must be a valid TCP port.")
        if self.conversation_max_messages < 0 or self.conversation_max_sessions < 0:
            errors.append("Conversation max values must be >= 0.")
        if self.max_concurrent_inferences <= 0 or self.max_pending_inferences < 0:
            errors.append("Concurrency limits must be positive/coherent.")
        if self.max_zip_entries <= 0 or self.max_zip_uncompressed_bytes <= 0:
            errors.append("ZIP limits must be > 0.")
        if errors:
            raise ConfigurationError("Invalid configuration: " + " ".join(errors))

    def effective_configuration(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        if result.get("pgvector_dsn"):
            result["pgvector_dsn"] = _sanitize_url(str(result["pgvector_dsn"]))
        if result.get("openai_api_key"):
            result["openai_api_key"] = "[redacted]"
        for field_name in ("master_prompt_file", "tool_calling_config_file", "routing_dir"):
            resource = result.get(field_name)
            if isinstance(resource, ConfigResource):
                result[field_name] = {
                    "source": resource.source,
                    "location": str(resource.location),
                }
        return result

    def runtime_control_config(self) -> dict[str, Any]:
        return {
            "max_input_chars": self.max_input_chars,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "max_concurrent_inferences": self.max_concurrent_inferences,
            "max_pending_inferences": self.max_pending_inferences,
            "request_timeout_seconds": self.request_timeout_seconds,
            "rate_limit_max_buckets": self.rate_limit_max_buckets,
        }

    def pgvector_settings(self) -> PgVectorStoreSettings:
        return PgVectorStoreSettings(
            dsn=self.pgvector_dsn,
            schema_name=self.pgvector_schema,
            table_name=self.pgvector_table,
            collection_table_name=self.pgvector_collection_table,
            distance_strategy=self.pgvector_distance_strategy,
            index_type=self.pgvector_index_type,
            create_extension=self.pgvector_create_extension,
            connect_max_attempts=self.pgvector_connect_max_attempts,
            connect_retry_seconds=self.pgvector_connect_retry_seconds,
            connect_timeout_seconds=self.pgvector_connect_timeout_seconds,
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv()
    def _env_bool(name: str, default: str) -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    project_root = None
    documents_dir = Path(os.getenv("DOCUMENTS_DIR", str(DEFAULT_RUNTIME_ROOT / "documents"))).expanduser()
    data_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_RUNTIME_ROOT / "data"))).expanduser()
    web_dir = Path(os.environ["WEB_DIR"]).expanduser() if os.getenv("WEB_DIR") else None
    routing_dir = resource_from_optional_path(
        None,
        env_value=os.getenv("ROUTING_DIR"),
        packaged_parts=("routing",),
        description="routing directory",
        kind="dir",
    )
    chroma_db_dir = Path(os.getenv("CHROMA_DB_DIR", str(data_dir / "rag_data")))
    router_store_dir = Path(os.getenv("ROUTER_STORE_DIR", str(data_dir / "router_data")))
    master_prompt_file = resource_from_optional_path(
        None,
        env_value=os.getenv("MASTER_PROMPT_FILE"),
        packaged_parts=("prompts", "master_prompt.json"),
        description="master prompt file",
        kind="file",
    )
    tool_calling_config_file = resource_from_optional_path(
        None,
        env_value=os.getenv("TOOL_CALLING_CONFIG_FILE"),
        packaged_parts=("prompts", "tool_calling.json"),
        description="tool calling config file",
        kind="file",
    )

    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    deprecated_auth_vars = [name for name in ("AUTH_ENABLED", "AUTH_API_KEY", "AUTH_HEADER_NAME", "AUTH_PROTECT_HEALTH") if name in os.environ]
    if deprecated_auth_vars:
        warnings.warn(
            "Standalone API key authentication has been removed. "
            f"Ignoring deprecated environment variables: {', '.join(deprecated_auth_vars)}. "
            "Protect embedded deployments in the host application.",
            DeprecationWarning,
            stacklevel=2,
        )
    prod_defaults = app_env == "prod"
    request_timeout_seconds = float(
        os.getenv(
            "REPLY_TIMEOUT_SECONDS",
            os.getenv("REQUEST_TIMEOUT_SECONDS", os.getenv("INFERENCE_TIMEOUT_SECONDS", "60")),
        )
    )
    settings = AppSettings(
        project_root=project_root,
        documents_dir=documents_dir,
        data_dir=data_dir,
        web_dir=web_dir,
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "chroma").strip().lower(),
        chroma_db_dir=chroma_db_dir,
        router_store_dir=router_store_dir,
        pgvector_dsn=os.getenv("PGVECTOR_DSN", ""),
        pgvector_schema=os.getenv("PGVECTOR_SCHEMA", "rag").strip().lower(),
        pgvector_table=os.getenv("PGVECTOR_TABLE", "vector_documents").strip().lower(),
        pgvector_collection_table=os.getenv("PGVECTOR_COLLECTION_TABLE", "vector_collections").strip().lower(),
        pgvector_distance_strategy=os.getenv("PGVECTOR_DISTANCE_STRATEGY", "cosine").strip().lower(),
        pgvector_index_type=os.getenv("PGVECTOR_INDEX_TYPE", "hnsw").strip().lower(),
        pgvector_create_extension=_env_bool("PGVECTOR_CREATE_EXTENSION", "true"),
        pgvector_connect_max_attempts=int(os.getenv("PGVECTOR_CONNECT_MAX_ATTEMPTS", "12")),
        pgvector_connect_retry_seconds=float(os.getenv("PGVECTOR_CONNECT_RETRY_SECONDS", "2")),
        pgvector_connect_timeout_seconds=int(os.getenv("PGVECTOR_CONNECT_TIMEOUT_SECONDS", "5")),
        routing_dir=routing_dir,
        master_prompt_file=master_prompt_file,
        tool_calling_config_file=tool_calling_config_file,
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower(),
        generation_model=os.getenv("GENERATION_MODEL", "llama3.2:3b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        max_document_file_size_bytes=int(os.getenv("MAX_DOCUMENT_FILE_SIZE_BYTES", str(25 * 1024 * 1024))),
        server_host=os.getenv("SERVER_HOST", "127.0.0.1"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        cleaner_remove_emojis=_env_bool("CLEANER_REMOVE_EMOJIS", "true"),
        cleaner_expand_abbreviations=_env_bool("CLEANER_EXPAND_ABBREVIATIONS", "true"),
        cleaner_normalize_unicode=_env_bool("CLEANER_NORMALIZE_UNICODE", "true"),
        cleaner_remove_extra_spaces=_env_bool("CLEANER_REMOVE_EXTRA_SPACES", "true"),
        cleaner_lowercase=_env_bool("CLEANER_LOWERCASE", "true"),
        max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "1000")),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "60" if prod_defaults else "20")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        inference_timeout_seconds=request_timeout_seconds,
        hybrid_dense_k=int(os.getenv("HYBRID_DENSE_K", "10")),
        hybrid_dense_fetch_k=int(os.getenv("HYBRID_DENSE_FETCH_K", "20")),
        hybrid_sparse_k=int(os.getenv("HYBRID_SPARSE_K", "10")),
        hybrid_candidate_k=int(os.getenv("HYBRID_CANDIDATE_K", "10")),
        hybrid_final_k=int(os.getenv("HYBRID_FINAL_K", "5")),
        hybrid_dense_lambda_mult=float(os.getenv("HYBRID_DENSE_LAMBDA_MULT", "0.5")),
        hybrid_dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5")),
        hybrid_sparse_weight=float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.5")),
        metadata_boost_weight=float(os.getenv("METADATA_BOOST_WEIGHT", "0.10")),
        reranking_mode=os.getenv("RERANKING_MODE", "auto").strip().lower(),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        reranker_allow_lexical_fallback=_env_bool("RERANKER_ALLOW_LEXICAL_FALLBACK", "false"),
        retrieval_confidence_threshold=float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.35")),
        generation_temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.7")),
        tool_calling_enabled=_env_bool("TOOL_CALLING_ENABLED", "true"),
        tool_calling_model=os.getenv(
            "TOOL_CALLING_MODEL",
            os.getenv("GENERATION_MODEL", "llama3.2:3b"),
        ),
        tool_calling_max_rounds=int(os.getenv("TOOL_CALLING_MAX_ROUNDS", "2")),
        conversation_memory_enabled=_env_bool("CONVERSATION_MEMORY_ENABLED", "true"),
        conversation_max_messages=int(os.getenv("CONVERSATION_MAX_MESSAGES", "8")),
        conversation_ttl_seconds=int(os.getenv("CONVERSATION_TTL_SECONDS", "1800")),
        conversation_max_sessions=int(os.getenv("CONVERSATION_MAX_SESSIONS", "1000")),
        conversation_max_message_chars=int(os.getenv("CONVERSATION_MAX_MESSAGE_CHARS", "2000")),
        query_rewriting_enabled=_env_bool("QUERY_REWRITING_ENABLED", "true"),
        query_rewriting_model=os.getenv(
            "QUERY_REWRITING_MODEL",
            os.getenv("GENERATION_MODEL", "llama3.2:3b"),
        ),
        query_rewriting_use_llm=_env_bool("QUERY_REWRITING_USE_LLM", "true"),
        query_rewriting_timeout_seconds=float(os.getenv("QUERY_REWRITING_TIMEOUT_SECONDS", "5")),
        ollama_connect_timeout_seconds=float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5")),
        ollama_read_timeout_seconds=float(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS", str(request_timeout_seconds))),
        ollama_max_response_bytes=int(os.getenv("OLLAMA_MAX_RESPONSE_BYTES", str(4 * 1024 * 1024))),
        max_concurrent_inferences=int(os.getenv("MAX_CONCURRENT_INFERENCES", "2")),
        max_pending_inferences=int(os.getenv("MAX_PENDING_INFERENCES", "8" if prod_defaults else "4")),
        request_timeout_seconds=request_timeout_seconds,
        rate_limit_max_buckets=int(os.getenv("RATE_LIMIT_MAX_BUCKETS", "10000")),
        trusted_proxy_hosts=tuple(
            host.strip()
            for host in os.getenv("TRUSTED_PROXY_HOSTS", "").split(",")
            if host.strip()
        ),
        app_env=app_env,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        query_debug_log_content=_env_bool("QUERY_DEBUG_LOG_CONTENT", "false"),
        max_zip_entries=int(os.getenv("MAX_ZIP_ENTRIES", "1000")),
        max_zip_uncompressed_bytes=int(os.getenv("MAX_ZIP_UNCOMPRESSED_BYTES", str(100 * 1024 * 1024))),
        max_zip_compression_ratio=float(os.getenv("MAX_ZIP_COMPRESSION_RATIO", "100")),
        max_pdf_pages=int(os.getenv("MAX_PDF_PAGES", "1000")),
        max_spreadsheet_sheets=int(os.getenv("MAX_SPREADSHEET_SHEETS", "100")),
        max_spreadsheet_rows=int(os.getenv("MAX_SPREADSHEET_ROWS", "100000")),
        max_presentation_slides=int(os.getenv("MAX_PRESENTATION_SLIDES", "500")),
        auto_index_on_startup=_env_bool("AUTO_INDEX_ON_STARTUP", "true"),
    )
    settings.validate()
    return settings


def _sanitize_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.password:
        return value
    netloc = parsed.netloc.replace(f":{parsed.password}@", ":[redacted]@")
    return parsed._replace(netloc=netloc).geturl()
