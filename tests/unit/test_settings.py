import pytest

from local_rag_chatbot.app_errors import ConfigurationError


def test_openai_requires_credentials(api_config):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        api_config.with_overrides(llm_provider="openai", openai_api_key="")


def test_invalid_chunk_overlap_is_rejected(api_config):
    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        api_config.with_overrides(chunk_size=10, chunk_overlap=10)
