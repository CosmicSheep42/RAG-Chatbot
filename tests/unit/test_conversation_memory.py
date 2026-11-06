from local_rag_chatbot.conversation.conversation_memory import InMemoryConversationStore


def test_conversations_are_isolated_and_history_is_limited():
    store = InMemoryConversationStore(max_messages=2, max_message_chars=5)
    store.append("one", "user", "first message")
    store.append("one", "assistant", "reply")
    store.append("one", "user", "third")
    store.append("two", "user", "other")

    assert [(message.role, message.content) for message in store.get_history("one")] == [
        ("assistant", "reply"),
        ("user", "third"),
    ]
    assert [message.content for message in store.get_history("two")] == ["other"]


def test_expiration_uses_injected_clock_without_sleeping():
    now = [100.0]
    store = InMemoryConversationStore(ttl_seconds=10, clock=lambda: now[0])
    store.append("conversation", "user", "hello")
    now[0] = 111.0

    assert store.get_history("conversation") == []
