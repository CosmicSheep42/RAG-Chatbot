from types import SimpleNamespace

import pytest

from local_rag_chatbot.rag.chatbot import RAGChatbot
from local_rag_chatbot.rag.index_manager import RAGIndexStatus


def _chatbot(result):
    chatbot = object.__new__(RAGChatbot)
    chatbot.index_manager = SimpleNamespace(
        rag_status=RAGIndexStatus.INDEXED,
        retrieval_service=object(),
    )

    async def aanswer(*_args, **_kwargs):
        return result

    chatbot.aanswer = aanswer
    return chatbot


def test_sync_answer_uses_same_async_execution_path():
    expected = {"result": "grounded"}
    assert _chatbot(expected).answer("question") == expected


@pytest.mark.asyncio
async def test_sync_answer_directs_event_loop_callers_to_async_api():
    expected = {"result": "grounded"}
    with pytest.raises(RuntimeError, match="use await aanswer"):
        _chatbot(expected).answer("question")
