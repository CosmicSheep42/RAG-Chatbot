from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def document():
    return lambda content, **metadata: Document(page_content=content, metadata=metadata)


@pytest.fixture
def api_config(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("WEB_DIR", "")
    from local_rag_chatbot.settings import get_settings

    get_settings.cache_clear()
    config = get_settings().with_overrides(
        data_dir=tmp_path / "data",
        documents_dir=tmp_path / "documents",
        web_dir=None,
        app_env="test",
        auto_index_on_startup=False,
    )
    yield config
    get_settings.cache_clear()


@pytest.fixture
def fake_engine(api_config):
    class FakeEngine:
        config = api_config
        ready = True

        def status(self):
            return {"ready": True, "initializing": False, "error": None, "rag": {}, "features": {}, "controls": {}}

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        async def chat(self, query, **kwargs):
            from local_rag_chatbot.models import ChatResponse, Citation, SourceSnippet

            return ChatResponse(
                result=f"Answer: {query}",
                sources=[SourceSnippet(source="guide.md", snippet="Relevant context")],
                citations=[Citation(citation_id="guide-1", source_name="guide.md", snippet="Relevant context")],
            )

    return FakeEngine()
