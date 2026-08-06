from local_rag_chatbot.config import ChatbotConfig

__all__ = [
    "ChatbotConfig",
    "ChatbotEngine",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "ToolCall",
    "create_default_engine",
]


def __getattr__(name: str):
    if name in {"ChatbotEngine", "create_default_engine"}:
        from local_rag_chatbot.engine import ChatbotEngine, create_default_engine

        return {
            "ChatbotEngine": ChatbotEngine,
            "create_default_engine": create_default_engine,
        }[name]
    if name in {"ChatResponse", "Citation", "ChatRequest", "ToolCall"}:
        from local_rag_chatbot.models import ChatResponse, Citation, ChatRequest, ToolCall

        return {
            "ChatResponse": ChatResponse,
            "Citation": Citation,
            "ChatRequest": ChatRequest,
            "ToolCall": ToolCall,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
