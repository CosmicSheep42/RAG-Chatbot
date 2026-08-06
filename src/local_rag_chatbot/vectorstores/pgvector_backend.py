from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import math
import re
import time
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from local_rag_chatbot.app_errors import ConfigurationError
from local_rag_chatbot.vectorstores.contracts import ScoredDocument, VectorSearchStrategy

logger = logging.getLogger(__name__)
PGVECTOR_DEPENDENCY_ERROR = (
    "PGVector backend dependencies are not installed.\n"
    'Install them with: pip install -e ".[pgvector]"'
)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LAYOUT_VERSION = "pgvector-generations-v1"

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover - depends on selected optional extra
    psycopg = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PgVectorStoreSettings:
    dsn: str
    schema_name: str
    table_name: str
    collection_table_name: str
    distance_strategy: str = "cosine"
    index_type: str = "hnsw"
    create_extension: bool = True
    connect_max_attempts: int = 12
    connect_retry_seconds: float = 2.0
    connect_timeout_seconds: int = 5


class PgVectorStoreBackend:
    provider_id = "pgvector"
    schema_version = LAYOUT_VERSION

    def __init__(self, *, settings: PgVectorStoreSettings, embeddings: object) -> None:
        if psycopg is None:
            raise ConfigurationError(PGVECTOR_DEPENDENCY_ERROR)
        self.settings = settings
        self.embeddings = embeddings
        self._validate_settings()

    def fingerprint_payload(self) -> dict[str, str | int | float | bool | None]:
        return {
            "provider": self.provider_id,
            "schema_version": self.schema_version,
            "distance_strategy": self.settings.distance_strategy,
            "index_type": self.settings.index_type,
            "schema": self.settings.schema_name,
            "table_layout_version": LAYOUT_VERSION,
            "embedding_dimensions": self._embedding_dimensions_if_known(),
        }

    def index_exists(self, namespace: str) -> bool:
        self.healthcheck()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT active_generation FROM {}.{} WHERE namespace = %s").format(
                        self._id(self.settings.schema_name),
                        self._id(self.settings.collection_table_name),
                    ),
                    (namespace,),
                )
                row = cur.fetchone()
                return bool(row and row[0])

    def replace_documents(self, namespace: str, documents: list[Document]) -> None:
        if not documents:
            raise ValueError(f"No documents supplied for vector namespace '{namespace}'.")
        embeddings = self._embed_documents([document.page_content for document in documents])
        dimension = len(embeddings[0]) if embeddings else self._embedding_dimensions()
        generation_id = uuid4().hex
        rows = [
            (
                namespace,
                generation_id,
                self._document_key(document, index),
                document.page_content,
                json.dumps(getattr(document, "metadata", {}) or {}),
                self._vector_literal(vector),
            )
            for index, (document, vector) in enumerate(zip(documents, embeddings))
        ]

        with self._connect() as conn:
            with conn.transaction():
                self._advisory_lock(conn, namespace)
                self._ensure_schema(conn, dimension=dimension)
                with conn.cursor() as cur:
                    cur.executemany(
                        sql.SQL(
                            """
                            INSERT INTO {}.{}
                                (namespace, generation_id, document_key, content, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                            """
                        ).format(self._id(self.settings.schema_name), self._id(self.settings.table_name)),
                        rows,
                    )
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {}.{} WHERE namespace = %s AND generation_id = %s").format(
                            self._id(self.settings.schema_name),
                            self._id(self.settings.table_name),
                        ),
                        (namespace, generation_id),
                    )
                    indexed_count = int(cur.fetchone()[0])
                    if indexed_count != len(documents):
                        raise RuntimeError(
                            f"PGVector replacement failed for namespace '{namespace}': expected "
                            f"{len(documents)} documents, indexed {indexed_count}."
                        )
                    self._before_activate(namespace, generation_id)
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.{} (namespace, active_generation)
                            VALUES (%s, %s)
                            ON CONFLICT (namespace) DO UPDATE
                            SET active_generation = EXCLUDED.active_generation,
                                updated_at = now()
                            """
                        ).format(self._id(self.settings.schema_name), self._id(self.settings.collection_table_name)),
                        (namespace, generation_id),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.{} WHERE namespace = %s AND generation_id <> %s").format(
                            self._id(self.settings.schema_name),
                            self._id(self.settings.table_name),
                        ),
                        (namespace, generation_id),
                    )

    def search(
        self,
        namespace: str,
        query: str,
        *,
        k: int,
        fetch_k: int | None = None,
        strategy: VectorSearchStrategy = "similarity",
        lambda_mult: float | None = None,
    ) -> list[ScoredDocument]:
        query_embedding = self._embed_query(query)
        limit = fetch_k if strategy == "mmr" and fetch_k is not None else k
        candidates = self._similarity_candidates(namespace, query_embedding, limit=max(limit, k))
        if strategy == "mmr":
            candidates = self._mmr_candidates(candidates, query_embedding, k=k, lambda_mult=lambda_mult or 0.5)
        else:
            candidates = candidates[:k]
        return [ScoredDocument(document=item["document"], score=item["score"]) for item in candidates]

    def count(self, namespace: str) -> int:
        self.healthcheck()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT count(*)
                        FROM {}.{} d
                        JOIN {}.{} c
                          ON c.namespace = d.namespace
                         AND c.active_generation = d.generation_id
                        WHERE d.namespace = %s
                        """
                    ).format(
                        self._id(self.settings.schema_name),
                        self._id(self.settings.table_name),
                        self._id(self.settings.schema_name),
                        self._id(self.settings.collection_table_name),
                    ),
                    (namespace,),
                )
                return int(cur.fetchone()[0])

    def delete_index(self, namespace: str) -> None:
        self.healthcheck()
        with self._connect() as conn:
            with conn.transaction():
                self._advisory_lock(conn, namespace)
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {}.{} WHERE namespace = %s").format(
                            self._id(self.settings.schema_name),
                            self._id(self.settings.table_name),
                        ),
                        (namespace,),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.{} WHERE namespace = %s").format(
                            self._id(self.settings.schema_name),
                            self._id(self.settings.collection_table_name),
                        ),
                        (namespace,),
                    )

    def cleanup_namespace_artifacts(self, namespace: str) -> None:
        self.healthcheck()
        with self._connect() as conn:
            with conn.transaction():
                self._advisory_lock(conn, namespace)
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            DELETE FROM {}.{} d
                            USING {}.{} c
                            WHERE d.namespace = %s
                              AND c.namespace = d.namespace
                              AND d.generation_id <> c.active_generation
                            """
                        ).format(
                            self._id(self.settings.schema_name),
                            self._id(self.settings.table_name),
                            self._id(self.settings.schema_name),
                            self._id(self.settings.collection_table_name),
                        ),
                        (namespace,),
                    )

    def validate_index(self, namespace: str) -> None:
        if self.count(namespace) <= 0:
            raise RuntimeError(f"PGVector validation failed for namespace '{namespace}': no active documents indexed.")

    def healthcheck(self) -> None:
        try:
            with self._connect() as conn:
                with conn.transaction():
                    self._ensure_schema(conn, dimension=self._embedding_dimensions())
        except psycopg.errors.InsufficientPrivilege as exc:
            raise ConfigurationError(
                "The PostgreSQL vector extension is unavailable or cannot be created by this user. "
                "Ask a database administrator to run: CREATE EXTENSION IF NOT EXISTS vector;"
            ) from exc
        except psycopg.errors.UndefinedObject as exc:
            raise ConfigurationError(
                "The PostgreSQL vector extension is unavailable. Ask a database administrator to run: "
                "CREATE EXTENSION IF NOT EXISTS vector;"
            ) from exc
        except psycopg.OperationalError as exc:
            raise ConfigurationError(f"Could not connect to PGVector database: {self._sanitize_error(str(exc))}") from exc

    def _similarity_candidates(self, namespace: str, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
        self.healthcheck()
        operator = self._distance_operator()
        query_vector = self._vector_literal(query_embedding)
        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT d.content,
                               d.metadata,
                               d.embedding::text AS embedding_text,
                               d.embedding {} %s::vector AS distance
                        FROM {}.{} d
                        JOIN {}.{} c
                          ON c.namespace = d.namespace
                         AND c.active_generation = d.generation_id
                        WHERE d.namespace = %s
                        ORDER BY d.embedding {} %s::vector
                        LIMIT %s
                        """
                    ).format(
                        sql.SQL(operator),
                        self._id(self.settings.schema_name),
                        self._id(self.settings.table_name),
                        self._id(self.settings.schema_name),
                        self._id(self.settings.collection_table_name),
                        sql.SQL(operator),
                    ),
                    (query_vector, namespace, query_vector, limit),
                )
                rows = list(cur.fetchall())

        raw_scores = [self._raw_relevance(float(row["distance"])) for row in rows]
        normalized_scores = self._normalize_batch(raw_scores)
        return [
            {
                "document": self._document_from_row(row, score),
                "score": score,
                "embedding": self._parse_vector(str(row["embedding_text"])),
            }
            for row, score in zip(rows, normalized_scores)
        ]

    def _mmr_candidates(
        self,
        candidates: list[dict[str, Any]],
        query_embedding: list[float],
        *,
        k: int,
        lambda_mult: float,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        remaining = list(candidates)
        while remaining and len(selected) < k:
            best_index = 0
            best_score = -math.inf
            for index, item in enumerate(remaining):
                query_score = self._cosine_similarity(query_embedding, item["embedding"])
                diversity_penalty = max(
                    (self._cosine_similarity(item["embedding"], selected_item["embedding"]) for selected_item in selected),
                    default=0.0,
                )
                score = lambda_mult * query_score - (1.0 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))
        for rank, item in enumerate(selected):
            score = 1.0 - (rank / max(len(selected), 1))
            item["score"] = score
            item["document"].metadata["retrieval_dense_score"] = round(score, 6)
        return selected

    def _ensure_schema(self, conn: Any, *, dimension: int | None) -> None:
        with conn.cursor() as cur:
            if self.settings.create_extension:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(self._id(self.settings.schema_name)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        namespace text PRIMARY KEY,
                        active_generation text NOT NULL,
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                ).format(self._id(self.settings.schema_name), self._id(self.settings.collection_table_name))
            )
            if dimension is not None:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            id bigserial PRIMARY KEY,
                            namespace text NOT NULL,
                            generation_id text NOT NULL,
                            document_key text NOT NULL,
                            content text NOT NULL,
                            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                            embedding vector({}),
                            created_at timestamptz NOT NULL DEFAULT now(),
                            UNIQUE(namespace, generation_id, document_key)
                        )
                        """
                    ).format(
                        self._id(self.settings.schema_name),
                        self._id(self.settings.table_name),
                        sql.SQL(str(dimension)),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (namespace, generation_id)").format(
                        self._id(f"{self.settings.table_name}_namespace_generation_idx"),
                        self._id(self.settings.schema_name),
                        self._id(self.settings.table_name),
                    )
                )
                self._create_vector_index(cur)

    def _create_vector_index(self, cur: Any) -> None:
        if self.settings.index_type == "none":
            return
        operator_class = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }[self.settings.distance_strategy]
        index_name = f"{self.settings.table_name}_embedding_{self.settings.index_type}_idx"
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING {} (embedding {})").format(
                self._id(index_name),
                self._id(self.settings.schema_name),
                self._id(self.settings.table_name),
                sql.SQL(self.settings.index_type),
                sql.SQL(operator_class),
            )
        )

    def _connect(self) -> Any:
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, self.settings.connect_max_attempts + 1):
            attempts_made = attempt
            try:
                return psycopg.connect(
                    self._psycopg_dsn(),
                    connect_timeout=self.settings.connect_timeout_seconds,
                )
            except psycopg.OperationalError as exc:
                last_error = exc
                if not self._is_transient_connection_error(exc) or attempt >= self.settings.connect_max_attempts:
                    break
                logger.warning(
                    "PGVector connection attempt %s of %s failed; retrying.",
                    attempt,
                    self.settings.connect_max_attempts,
                )
                if self.settings.connect_retry_seconds:
                    time.sleep(self.settings.connect_retry_seconds)

        raise ConfigurationError(
            f"Could not connect to PGVector database after {attempts_made} attempts. "
            "Check whether PostgreSQL is running and whether PGVECTOR_DSN is correct."
        ) from last_error

    def _before_activate(self, namespace: str, generation_id: str) -> None:
        """Test seam for failures between generation validation and promotion."""
        return None

    @staticmethod
    def _is_transient_connection_error(exc: Exception) -> bool:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate is not None:
            return str(sqlstate).startswith("08")
        message = str(exc).lower()
        transient_markers = (
            "connection refused",
            "connection timed out",
            "timeout expired",
            "could not translate host name",
            "temporary failure in name resolution",
            "server is starting",
            "the database system is starting",
        )
        return any(marker in message for marker in transient_markers)

    def _psycopg_dsn(self) -> str:
        if self.settings.dsn.startswith("postgresql+psycopg://"):
            return "postgresql://" + self.settings.dsn.removeprefix("postgresql+psycopg://")
        return self.settings.dsn

    def _advisory_lock(self, conn: Any, namespace: str) -> None:
        key = int.from_bytes(sha256(f"local-rag-chatbot:{namespace}".encode("utf-8")).digest()[:8], "big", signed=True)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (key,))

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embeddings.embed_documents(texts)
        return [[float(value) for value in vector] for vector in vectors]

    def _embed_query(self, text: str) -> list[float]:
        vector = self.embeddings.embed_query(text)
        return [float(value) for value in vector]

    def _embedding_dimensions(self) -> int:
        known = self._embedding_dimensions_if_known()
        if known is not None:
            return known
        return len(self._embed_query("dimension probe"))

    def _embedding_dimensions_if_known(self) -> int | None:
        for attribute in ("dimensions", "embedding_dimensions"):
            value = getattr(self.embeddings, attribute, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _distance_operator(self) -> str:
        return {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>",
        }[self.settings.distance_strategy]

    def _raw_relevance(self, distance: float) -> float:
        if self.settings.distance_strategy == "cosine":
            return 1.0 - distance
        if self.settings.distance_strategy == "l2":
            return 1.0 / (1.0 + max(distance, 0.0))
        return -distance

    def _normalize_batch(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        if self.settings.distance_strategy in {"cosine", "l2"}:
            return [max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0 for score in scores]
        finite_scores = [score for score in scores if math.isfinite(score)]
        if not finite_scores:
            return [0.0 for _ in scores]
        minimum = min(finite_scores)
        maximum = max(finite_scores)
        if maximum <= minimum:
            return [max(0.0, min(1.0, score)) if self.settings.distance_strategy != "inner_product" else 1.0 for score in scores]
        return [max(0.0, min(1.0, (score - minimum) / (maximum - minimum))) for score in scores]

    def _document_from_row(self, row: dict[str, Any], score: float) -> Document:
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        document = Document(page_content=str(row["content"]), metadata=dict(metadata))
        document.metadata["retrieval_dense_score"] = round(score, 6)
        return document

    def _document_key(self, document: Document, index: int) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        key_bits = [
            str(metadata.get("chunk_id", "")),
            str(metadata.get("source", "")),
            str(metadata.get("chunk_index", "")),
            str(index),
            document.page_content,
        ]
        return sha256("\x1f".join(key_bits).encode("utf-8")).hexdigest()

    def _vector_literal(self, vector: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def _parse_vector(self, value: str) -> list[float]:
        return [float(part) for part in value.strip("[]").split(",") if part]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)

    def _id(self, value: str) -> Any:
        if not IDENTIFIER_PATTERN.match(value):
            raise ConfigurationError(f"Unsafe SQL identifier for PGVector configuration: {value!r}")
        return sql.Identifier(value)

    def _validate_settings(self) -> None:
        if not self.settings.dsn.strip():
            raise ConfigurationError("PGVECTOR_DSN is required when VECTOR_STORE_PROVIDER=pgvector.")
        for value in (self.settings.schema_name, self.settings.table_name, self.settings.collection_table_name):
            if not IDENTIFIER_PATTERN.match(value):
                raise ConfigurationError(f"Unsafe SQL identifier for PGVector configuration: {value!r}")
        if self.settings.distance_strategy not in {"cosine", "l2", "inner_product"}:
            raise ConfigurationError("PGVECTOR_DISTANCE_STRATEGY must be one of: cosine, l2, inner_product.")
        if self.settings.index_type not in {"none", "hnsw", "ivfflat"}:
            raise ConfigurationError("PGVECTOR_INDEX_TYPE must be one of: none, hnsw, ivfflat.")

    def _sanitize_error(self, message: str) -> str:
        sanitized = message
        if self.settings.dsn:
            sanitized = sanitized.replace(self.settings.dsn, "[redacted]")
        return re.sub(r"://([^:/@\s]+):([^@\s]+)@", r"://\1:[redacted]@", sanitized)
