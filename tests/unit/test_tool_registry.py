import json
from types import SimpleNamespace

import pytest

from local_rag_chatbot.documents.tool_config_repository import ToolDefinition
from local_rag_chatbot.rag.chatbot import RAGChatbot
from local_rag_chatbot.rag.tool_registry import ToolRegistry


@pytest.fixture
def tool():
    return ToolDefinition(
        name="search_documents",
        description="Search documents.",
        executor="search_documents",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
            },
            "required": ["query"],
        },
    )


def _registry(tool, handler):
    registry = ToolRegistry()
    registry.register(tool, handler, normalized_integer_fields=frozenset({"top_k"}))
    return registry


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [(4, 4), ("4", 4), (10, 6), ("10", 6)],
)
def test_search_integer_arguments_are_normalized_from_schema(tool, top_k, expected):
    seen = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(dict(arguments))
        return "{}", [object()], {}

    _, documents, payload = _registry(tool, handler).execute(
        tool.name,
        {"query": "restoration abilities", "top_k": top_k},
    )

    assert len(documents) == 1
    assert seen == [{"query": "restoration abilities", "top_k": expected}]
    assert payload["status"] == "success"


def test_missing_optional_integer_uses_schema_default(tool):
    seen = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(dict(arguments))
        return "{}", [], {}

    _, documents, payload = _registry(tool, handler).execute(tool.name, {"query": "question"})

    assert documents == []
    assert seen == [{"query": "question", "top_k": 4}]
    assert payload["status"] == "success_empty"


def test_integer_string_normalization_requires_explicit_registration(tool):
    registry = ToolRegistry()
    registry.register(tool, lambda *_args, **_kwargs: ("{}", [], {}))

    _, _, payload = registry.execute(tool.name, {"query": "question", "top_k": "4"})

    assert payload["status"] == "validation_error"


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": "4.0"},
        {"query": "question", "top_k": "+4"},
        {"query": "question", "top_k": "04"},
        {"query": "question", "top_k": "four"},
        {"query": "question", "top_k": 4.0},
        {"query": "question", "top_k": True},
        {"query": "", "top_k": 4},
        {"top_k": 4},
        {"query": "question", "top_k": 4, "extra": "no"},
    ],
)
def test_invalid_tool_arguments_are_rejected_before_handler(tool, arguments):
    called = False

    def handler(_definition, _arguments, *, request_context=None):
        nonlocal called
        called = True
        return "{}", [], {}

    content, documents, payload = _registry(tool, handler).execute(tool.name, arguments)

    assert json.loads(content)["status"] == "validation_error"
    assert payload["status"] == "validation_error"
    assert documents == []
    assert called is False


def test_invalid_tool_name_is_a_validation_error(tool):
    content, documents, payload = _registry(tool, lambda *_args, **_kwargs: ("{}", [], {})).execute(
        "unknown",
        {},
    )

    assert json.loads(content)["status"] == "validation_error"
    assert documents == []
    assert payload["status"] == "validation_error"


def test_execution_error_is_distinct_from_successful_empty_search(tool):
    def failing_handler(_definition, _arguments, *, request_context=None):
        raise RuntimeError("private failure details")

    _, documents, failed = _registry(tool, failing_handler).execute(tool.name, {"query": "question"})
    _, empty_documents, empty = _registry(
        tool,
        lambda *_args, **_kwargs: ("{}", [], {}),
    ).execute(tool.name, {"query": "question"})

    assert documents == []
    assert failed == {"status": "execution_error", "error": "Tool 'search_documents' failed."}
    assert empty_documents == []
    assert empty["status"] == "success_empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_falls_back_to_actual_search_when_model_skips_tool(tool, asynchronous):
    from local_rag_chatbot.providers.llm.types import ChatResult

    document = SimpleNamespace(metadata={"source": "rules.md"}, page_content="recovered evidence")
    seen = []
    prompt_queries = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(dict(arguments))
        return "{}", [document], {"results": ["evidence"]}

    class Model:
        def chat(self, *_args, **_kwargs):
            return ChatResult(content="", tool_calls=[])

        async def achat(self, *_args, **_kwargs):
            return ChatResult(content="", tool_calls=[])

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        tool_calling_model="test-model",
    )
    chatbot.tool_registry = _registry(tool, handler)
    chatbot.chat_model = Model()
    chatbot.answer_service = SimpleNamespace(
        build_chat_system_prompt=lambda query, *_args, **_kwargs: prompt_queries.append(query) or "prompt"
    )

    kwargs = {"standalone_query": "targeted question", "recovery_only": True}
    if asynchronous:
        result = await chatbot._agenerate_answer_with_optional_tools("Original user question", (), object(), **kwargs)
    else:
        result = chatbot._generate_answer_with_optional_tools("Original user question", (), object(), **kwargs)

    answer, documents, events, tool_used = result
    assert answer == ""
    assert documents == [document]
    assert seen == [{"query": "targeted question", "top_k": 4}]
    assert prompt_queries == ["Original user question"]
    assert events[0]["payload"]["status"] == "success"
    assert tool_used is True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_retries_when_model_search_only_returns_existing_documents(tool, asynchronous):
    from local_rag_chatbot.providers.llm.types import ChatResult, ToolCall

    existing = SimpleNamespace(metadata={"source": "rules.md"}, page_content="already retrieved")
    additional = SimpleNamespace(metadata={"source": "rules.md"}, page_content="new evidence")
    seen = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(arguments["query"])
        document = existing if len(seen) == 1 else additional
        return "{}", [document], {}

    class Model:
        def chat(self, *_args, **_kwargs):
            return ChatResult(
                content="",
                tool_calls=[ToolCall("search", "search_documents", {"query": "focused query", "top_k": 4})],
            )

        async def achat(self, *_args, **_kwargs):
            return self.chat()

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        tool_calling_model="test-model",
    )
    chatbot.tool_registry = _registry(tool, handler)
    chatbot.chat_model = Model()
    chatbot.answer_service = SimpleNamespace(build_chat_system_prompt=lambda *_args, **_kwargs: "prompt")

    kwargs = {"standalone_query": "standalone query", "recovery_only": True}
    if asynchronous:
        result = await chatbot._agenerate_answer_with_optional_tools("question", [existing], object(), **kwargs)
    else:
        result = chatbot._generate_answer_with_optional_tools("question", [existing], object(), **kwargs)

    _, documents, events, tool_used = result
    assert seen == ["focused query", "standalone query"]
    assert documents == [existing, additional]
    assert [event["new_document_count"] for event in events] == [0, 1]
    assert [event["payload"]["status"] for event in events] == ["success", "success"]
    assert tool_used is True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_executes_only_one_model_search_per_response(tool, asynchronous):
    from local_rag_chatbot.providers.llm.types import ChatResult, ToolCall

    seen = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(arguments["query"])
        return "{}", [SimpleNamespace(metadata={}, page_content=arguments["query"])], {}

    class Model:
        def chat(self, *_args, **_kwargs):
            return ChatResult(
                content="",
                tool_calls=[
                    ToolCall("one", "search_documents", {"query": "one"}),
                    ToolCall("two", "search_documents", {"query": "two"}),
                ],
            )

        async def achat(self, *_args, **_kwargs):
            return self.chat()

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        tool_calling_model="test-model",
    )
    chatbot.tool_registry = _registry(tool, handler)
    chatbot.chat_model = Model()
    chatbot.answer_service = SimpleNamespace(build_chat_system_prompt=lambda *_args, **_kwargs: "prompt")
    method = (
        chatbot._agenerate_answer_with_optional_tools if asynchronous else chatbot._generate_answer_with_optional_tools
    )
    result = method("question", (), object(), standalone_query="standalone", recovery_only=True)
    if asynchronous:
        result = await result

    _, _, events, _ = result
    assert seen == ["one"]
    assert events[1]["payload"] == {"skipped": True, "reason": "recovery_search_limit"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_retries_after_model_tool_validation_error(tool, asynchronous):
    from local_rag_chatbot.providers.llm.types import ChatResult, ToolCall

    document = SimpleNamespace(metadata={"source": "rules.md"}, page_content="recovered evidence")
    calls = 0

    def handler(_definition, _arguments, *, request_context=None):
        nonlocal calls
        calls += 1
        return "{}", [document], {}

    class Model:
        def chat(self, *_args, **_kwargs):
            return ChatResult(
                content="",
                tool_calls=[ToolCall("bad-top-k", "search_documents", {"query": "target", "top_k": "many"})],
            )

        async def achat(self, *_args, **_kwargs):
            return self.chat()

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        tool_calling_model="test-model",
    )
    chatbot.tool_registry = _registry(tool, handler)
    chatbot.chat_model = Model()
    chatbot.answer_service = SimpleNamespace(build_chat_system_prompt=lambda *_args, **_kwargs: "prompt")

    kwargs = {"standalone_query": "targeted question", "recovery_only": True}
    if asynchronous:
        result = await chatbot._agenerate_answer_with_optional_tools("question", (), object(), **kwargs)
    else:
        result = chatbot._generate_answer_with_optional_tools("question", (), object(), **kwargs)

    _, documents, events, tool_used = result
    assert documents == [document]
    assert [event["payload"]["status"] for event in events] == ["validation_error", "success"]
    assert calls == 1
    assert tool_used is True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("returns_documents", [False, True])
@pytest.mark.parametrize("recovery_query", ["target", "alternate"])
async def test_valid_model_search_distinguishes_empty_from_retrieved_evidence(
    tool,
    asynchronous,
    returns_documents,
    recovery_query,
):
    from local_rag_chatbot.providers.llm.types import ChatResult, ToolCall

    document = SimpleNamespace(metadata={"source": "rules.md"}, page_content="evidence")
    returned_documents = [document] if returns_documents else []
    seen = []

    def handler(_definition, arguments, *, request_context=None):
        seen.append(dict(arguments))
        return "{}", returned_documents, {}

    class Model:
        def chat(self, *_args, **_kwargs):
            return ChatResult(
                content="",
                tool_calls=[ToolCall("search", "search_documents", {"query": "target", "top_k": "10"})],
            )

        async def achat(self, *_args, **_kwargs):
            return self.chat()

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        tool_calling_model="test-model",
    )
    chatbot.tool_registry = _registry(tool, handler)
    chatbot.chat_model = Model()
    chatbot.answer_service = SimpleNamespace(build_chat_system_prompt=lambda *_args, **_kwargs: "prompt")

    kwargs = {"standalone_query": recovery_query, "recovery_only": True}
    if asynchronous:
        result = await chatbot._agenerate_answer_with_optional_tools("question", (), object(), **kwargs)
    else:
        result = chatbot._generate_answer_with_optional_tools("question", (), object(), **kwargs)

    _, documents, events, tool_used = result
    expected_status = "success" if returns_documents else "success_empty"
    expected_calls = [{"query": "target", "top_k": 6}]
    if not returns_documents and recovery_query != "target":
        expected_calls.append({"query": recovery_query, "top_k": 4})
    assert seen == expected_calls
    assert documents == returned_documents
    expected_events = [expected_status] * len(expected_calls)
    assert [event["payload"]["status"] for event in events] == expected_events
    assert tool_used is True


def test_recovered_documents_are_deduplicated_by_source_and_content():
    chatbot = object.__new__(RAGChatbot)
    first = SimpleNamespace(metadata={"source": "rules.md"}, page_content="same")
    duplicate = SimpleNamespace(metadata={"source": "rules.md"}, page_content="same")
    distinct_source = SimpleNamespace(metadata={"source": "other.md"}, page_content="same")
    distinct_content = SimpleNamespace(metadata={"source": "rules.md"}, page_content="different")

    assert chatbot._merge_documents([first], [duplicate, distinct_source, distinct_content]) == [
        first,
        distinct_source,
        distinct_content,
    ]
