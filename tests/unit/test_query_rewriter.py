from local_rag_chatbot.conversation.conversation_memory import ConversationMessage
from local_rag_chatbot.conversation.query_rewriter import QueryRewriter


def test_rewriter_rejects_new_technical_identifiers():
    rewriter = QueryRewriter(enabled=True, use_llm=False)
    history = [ConversationMessage("user", "Explain config-v2.json", 1.0)]

    assert not rewriter._valid_rewrite("what about it?", "what about unrelated-v9.json?", history)


def test_rewriter_preserves_identifiers_from_original_query():
    rewriter = QueryRewriter(enabled=True, use_llm=False)

    assert not rewriter._valid_rewrite("What is config-v2.json?", "What is config-v3.json?")
