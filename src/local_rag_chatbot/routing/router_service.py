from __future__ import annotations

import hashlib
import json
import logging
import shutil
from hashlib import sha1
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma

from local_rag_chatbot.documents.hash_manager import HashManager
from local_rag_chatbot.file_lock import FileLock, FileLockTimeoutError
from local_rag_chatbot.input_cleaner import TextPreprocessorPipeline
from local_rag_chatbot.routing.routing_config_loader import RoutingConfigLoader
from local_rag_chatbot.routing.routing_order_resolver import RoutingOrderResolver
from local_rag_chatbot.routing.routing_store_factory import RoutingStoreFactory
from local_rag_chatbot.resource_refs import ConfigResource
from local_rag_chatbot.settings import AppSettings


logger = logging.getLogger(__name__)
ROUTER_INDEX_SCHEMA_VERSION = "router-index-v2"
ROUTER_INDEX_STRATEGY_VERSION = "static-examples-chroma-v1"
ROUTER_LOCK_TIMEOUT_SECONDS = 60.0


class EmbeddingRouter:
    def __init__(
        self,
        *,
        settings: AppSettings,
        embeddings: object,
        input_pipeline: TextPreprocessorPipeline,
        hash_manager: HashManager | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.configs: dict[str, dict[str, Any]] = {}
        self.stores: dict[str, Chroma] = {}
        self.ordered_categories: list[str] = []
        self.ready = False
        self.config_loader = RoutingConfigLoader(routing_dir=self.settings.routing_dir)
        self.order_resolver = RoutingOrderResolver()
        self.store_factory = RoutingStoreFactory(self.embeddings)
        self.input_pipeline = input_pipeline
        self.hash_manager = hash_manager or HashManager(self.settings.data_dir / "router_hash.json")
        self.routing_hash_key = "routing_documents"
        self.routing_fingerprint_key = "routing_index_fingerprint"

    def setup(self) -> bool:
        bundle = self.config_loader.load()
        self.configs = bundle.configs
        self.ordered_categories = self.order_resolver.resolve(
            self.configs,
            bundle.priority_overrides,
        )
        self.stores = {}

        routing_files_hash = self._compute_routing_hash()
        current_fingerprint = self._compute_router_fingerprint(routing_files_hash)

        try:
            with FileLock(self.settings.data_dir / "router_index.lock", timeout_seconds=ROUTER_LOCK_TIMEOUT_SECONDS):
                stored_fingerprint = self.hash_manager.load_stored_hash(self.routing_fingerprint_key)
                force_rebuild = current_fingerprint != stored_fingerprint
                if force_rebuild:
                    logger.info("Router fingerprint changed; rebuilding routing stores")
                    self._replace_routing_stores_atomically()
                    self.hash_manager.save_hash(self.routing_hash_key, routing_files_hash)
                    self.hash_manager.save_hash(self.routing_fingerprint_key, current_fingerprint)

                for category in self.ordered_categories:
                    config = self.configs[category]
                    persist_directory = self.settings.router_store_dir / category
                    collection_name = f"router_{category}"
                    self.stores[category] = self.store_factory.build_store(
                        config=config,
                        persist_directory=persist_directory,
                        collection_name=collection_name,
                        force_rebuild=False,
                    )
        except FileLockTimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for the router index lock. Another process may be indexing this data directory."
            ) from exc

        self.ready = True
        return True

    def route(self, query: str, already_cleaned: bool = False) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Router not initialized. Call setup() first.")

        cleaned_query = query if already_cleaned else self.input_pipeline.process_for_routing(query)
        if not cleaned_query:
            return {
                "route": "llm",
                "reason": "empty_after_preprocessing",
                "matched_category": None,
                "matched_intent": None,
                "score": None,
            }

        for category in self.ordered_categories:
            store = self.stores.get(category)
            if store is None:
                continue

            match = self._best_match(store, cleaned_query)
            if not match:
                continue

            threshold = float(self.configs.get(category, {}).get("threshold", 0.80))
            if match["score"] >= threshold:
                return self._build_static_response(category, cleaned_query, match)

        return {
            "route": "llm",
            "reason": "no_static_route_match",
            "matched_category": None,
            "matched_intent": None,
            "score": None,
        }

    def _compute_routing_hash(self) -> str:
        routing_dir = self.settings.routing_dir
        if isinstance(routing_dir, ConfigResource) and not routing_dir.is_external:
            return self._compute_traversable_hash(routing_dir.location)
        return self.hash_manager.compute_directory_hash(routing_dir.location, patterns=["*.json"])

    def _compute_router_fingerprint(self, routing_files_hash: str) -> str:
        payload = {
            "schema_version": ROUTER_INDEX_SCHEMA_VERSION,
            "index_strategy_version": ROUTER_INDEX_STRATEGY_VERSION,
            "routing_files_hash": routing_files_hash,
            "embedding_provider": self._embedding_provider_identifier(),
            "embedding_model": self._embedding_model_identifier(),
            "embedding_dimensions": self._embedding_dimensions(),
            "embedding_normalization": "provider_default",
            "input_normalization": self.settings.input_cleaner_config().get("preprocessor_config", {}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _embedding_provider_identifier(self) -> str:
        provider_type = type(self.embeddings)
        return f"{provider_type.__module__}.{provider_type.__qualname__}"

    def _embedding_model_identifier(self) -> str:
        for attribute in ("model", "model_name", "embedding_model"):
            value = getattr(self.embeddings, attribute, None)
            if value:
                return str(value)
        return self.settings.embedding_model

    def _embedding_dimensions(self) -> int | None:
        for attribute in ("dimensions", "embedding_dimensions"):
            value = getattr(self.embeddings, attribute, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _replace_routing_stores_atomically(self) -> None:
        persist_directory = self.settings.router_store_dir
        staging_directory = persist_directory.with_name(f"{persist_directory.name}.staging")
        backup_directory = persist_directory.with_name(f"{persist_directory.name}.backup")
        self._cleanup_abandoned_router_artifacts(staging_directory, backup_directory, persist_directory)

        try:
            for category in self.ordered_categories:
                config = self.configs[category]
                collection_name = f"router_{category}"
                store = self.store_factory.build_store_in_directory(
                    config=config,
                    persist_directory=staging_directory / category,
                    collection_name=collection_name,
                )
                self._validate_routing_store(store, collection_name)

            moved_current = False
            try:
                if persist_directory.exists():
                    persist_directory.replace(backup_directory)
                    moved_current = True
                staging_directory.replace(persist_directory)
                if backup_directory.exists():
                    shutil.rmtree(backup_directory)
            except Exception:
                if staging_directory.exists():
                    shutil.rmtree(staging_directory, ignore_errors=True)
                if moved_current and backup_directory.exists() and not persist_directory.exists():
                    backup_directory.replace(persist_directory)
                raise
        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory, ignore_errors=True)
            raise

    def _cleanup_abandoned_router_artifacts(
        self,
        staging_directory: Path,
        backup_directory: Path,
        persist_directory: Path,
    ) -> None:
        if staging_directory.exists():
            logger.warning("Removing abandoned router index staging directory", extra={"path": str(staging_directory)})
            shutil.rmtree(staging_directory)
        if backup_directory.exists() and persist_directory.exists():
            logger.warning("Removing stale router index backup directory", extra={"path": str(backup_directory)})
            shutil.rmtree(backup_directory)
        elif backup_directory.exists() and not persist_directory.exists():
            logger.warning("Restoring router index backup because active directory is missing", extra={"path": str(backup_directory)})
            backup_directory.replace(persist_directory)

    def _validate_routing_store(self, store: Chroma, collection_name: str) -> None:
        collection = getattr(store, "_collection", None)
        if collection is None or collection.count() <= 0:
            raise RuntimeError(f"Routing store validation failed for {collection_name}: no examples indexed.")

    def _compute_traversable_hash(self, directory: Traversable) -> str:
        if not directory.exists():
            return ""

        hasher = hashlib.sha256()
        files = self._iter_json_resources(directory)
        for relative_name, resource in sorted(files, key=lambda item: item[0]):
            hasher.update(relative_name.encode("utf-8"))
            with resource.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def _iter_json_resources(self, directory: Traversable, prefix: str = "") -> list[tuple[str, Traversable]]:
        resources: list[tuple[str, Traversable]] = []
        for item in directory.iterdir():
            item_name = f"{prefix}/{item.name}" if prefix else item.name
            if item.is_dir():
                resources.extend(self._iter_json_resources(item, item_name))
            elif item.is_file() and item.name.endswith(".json"):
                resources.append((item_name, item))
        return resources

    def _best_match(self, store: Chroma, query: str) -> dict[str, Any] | None:
        results = store.similarity_search_with_relevance_scores(query, k=3)
        if not results:
            return None

        grouped: dict[str, dict[str, Any]] = {}
        for document, score in results:
            intent_key = document.metadata.get("intent_key")
            if intent_key is None:
                continue

            bucket = grouped.setdefault(
                intent_key,
                {
                    "documents": [],
                    "scores": [],
                    "best_document": document,
                    "best_score": float(score),
                },
            )
            bucket["documents"].append(document)
            score_value = float(score)
            bucket["scores"].append(score_value)
            if score_value > bucket["best_score"]:
                bucket["best_score"] = score_value
                bucket["best_document"] = document

        if not grouped:
            return None

        selected_intent_key, selected_bucket = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]["scores"]),
                sum(item[1]["scores"]) / len(item[1]["scores"]),
                max(item[1]["scores"]),
            ),
            reverse=True,
        )[0]

        return {
            "document": selected_bucket["best_document"],
            "score": sum(selected_bucket["scores"]) / len(selected_bucket["scores"]),
            "intent_key": selected_intent_key,
        }

    def _pick_variant(self, query: str, variants: list[str], salt: str) -> str:
        if not variants:
            return ""

        seed = f"{salt}:{query.strip().lower()}".encode("utf-8")
        digest = sha1(seed).digest()
        index = digest[0] % len(variants)
        return variants[index]

    def _build_static_response(self, category: str, query: str, match: dict[str, Any]) -> dict[str, Any]:
        intent = self._get_intent(category, match["intent_key"])
        variants = self._get_static_variants(intent)
        response = self._pick_variant(query, variants, f"{category}:{intent['key']}")

        return {
            "route": "static",
            "reason": "matched_static_intent",
            "matched_category": category,
            "matched_intent": intent["key"],
            "score": match["score"],
            "result": response,
        }

    def _get_static_variants(self, intent: dict[str, Any]) -> list[str]:
        for key in ("responses", "warnings", "answers", "replies", "messages"):
            variants = intent.get(key, [])
            if isinstance(variants, list) and variants:
                return [str(variant) for variant in variants if str(variant).strip()]
        return []

    def _get_intent(self, category: str, intent_key: str) -> dict[str, Any]:
        config = self.configs.get(category, {})
        for intent in config.get("intents", []):
            if intent.get("key") == intent_key:
                return intent
        raise KeyError(f"Intent not found in category '{category}': {intent_key}")
