from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RoutingStoreFactory:
    def __init__(self, embeddings: object) -> None:
        self.embeddings = embeddings

    def build_store(
        self,
        config: dict[str, Any],
        persist_directory: Path,
        collection_name: str,
        force_rebuild: bool = False,
    ) -> Chroma:
        if force_rebuild and persist_directory.exists():
            shutil.rmtree(persist_directory)

        persist_directory.mkdir(parents=True, exist_ok=True)

        if not force_rebuild and self._database_exists(persist_directory):
            logger.info("Loading existing routing store", extra={"collection_name": collection_name})
            return Chroma(
                persist_directory=str(persist_directory),
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )

        documents: list[Document] = []
        for intent in config["intents"]:
            intent_key = intent["key"]
            for example in intent.get("examples", []):
                documents.append(
                    Document(
                        page_content=example,
                        metadata={
                            "intent_key": intent_key,
                            "collection": collection_name,
                        },
                    )
                )

        if not documents:
            raise ValueError(f"No examples found for {collection_name}")

        logger.info(
            "Creating new routing store",
            extra={"collection_name": collection_name, "document_count": len(documents)},
        )
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
        )

    def build_store_in_directory(
        self,
        config: dict[str, Any],
        persist_directory: Path,
        collection_name: str,
    ) -> Chroma:
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        return self.build_store(
            config=config,
            persist_directory=persist_directory,
            collection_name=collection_name,
            force_rebuild=True,
        )

    def _database_exists(self, persist_directory: Path) -> bool:
        return persist_directory.exists() and (persist_directory / "chroma.sqlite3").exists()
