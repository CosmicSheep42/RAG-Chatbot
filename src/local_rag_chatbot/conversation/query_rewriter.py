from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
from typing import Any

from local_rag_chatbot.conversation.conversation_memory import ConversationMessage
from local_rag_chatbot.providers.llm.contracts import ChatModel as ChatClient


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    standalone_query: str
    rewritten: bool
    method: str = "none"
    error: str | None = None


DEPENDENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(e|and)\s+",
        r"^\s*(what about|and what about|e quanto a|e sobre)\b",
        r"\b(ele|ela|eles|elas|isso|isto|esse|essa|esses|essas|aquele|aquela|deles|delas)\b",
        r"\b(he|she|it|they|them|that|this|those|these|his|her|their)\b",
        r"\b(deles|delas|qual deles|qual delas|which one|which of them)\b",
        r"\b(outro|outra|segundo caso|another one|other one|second case)\b",
        r"^\s*(quanto custa|quanto vale|qual o preço|qual e o preco|how much|what price)\b",
        r"^\s*(quando|when)\s+(ele|ela|he|she|it|they)\b",
    )
]

INDEPENDENT_SOURCE_PATTERN = re.compile(
    r"\b[\w.-]+\.(?:txt|md|pdf|docx|xlsx|csv|json|html?|pptx)\b",
    re.IGNORECASE,
)

EXPLICIT_INDEPENDENT_PATTERN = re.compile(
    r"\b(resuma|summarize|explique|explain|compare|liste|list|quem|who|qual|what|quando|when|onde|where|como|how)\b",
    re.IGNORECASE,
)

ENTITY_PATTERN = re.compile(
    r"\b(?:plano|plan|modelo|model|contrato|contract|relatorio|relatório|report|empresa|company)\s+([A-Z][\w.-]*|\w+\.(?:pdf|docx|txt|md|csv|json|xlsx))\b",
    re.IGNORECASE,
)

COMPARE_PATTERN = re.compile(
    r"\b(compare|comparar|compare os|compare as)\s+(.+?)\s+(?:e|and|vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
    re.IGNORECASE,
)


class QueryRewriter:
    def __init__(
        self,
        *,
        chat_client: ChatClient | None = None,
        model: str | None = None,
        enabled: bool = True,
        use_llm: bool = True,
        max_history_messages: int = 8,
        max_history_chars: int = 4000,
        timeout_seconds: float | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.enabled = enabled
        self.use_llm = use_llm
        self.max_history_messages = max(0, max_history_messages)
        self.max_history_chars = max(200, max_history_chars)
        self.timeout_seconds = timeout_seconds

    def rewrite(self, query: str, history: list[ConversationMessage]) -> QueryRewriteResult:
        original_query = str(query or "").strip()
        if not original_query:
            return QueryRewriteResult(original_query, original_query, False)
        if not self.enabled:
            return QueryRewriteResult(original_query, original_query, False, method="disabled")

        recent_history = self._trim_history(history)
        if not recent_history or not self._needs_context(original_query):
            return QueryRewriteResult(original_query, original_query, False)

        heuristic_query = self._heuristic_rewrite(original_query, recent_history)
        if heuristic_query and self._valid_rewrite(original_query, heuristic_query):
            return QueryRewriteResult(
                original_query=original_query,
                standalone_query=heuristic_query,
                rewritten=heuristic_query != original_query,
                method="heuristic",
            )

        if not self.use_llm or self.chat_client is None:
            return QueryRewriteResult(original_query, original_query, False, method="fallback")

        try:
            standalone_query = self._rewrite_with_llm(original_query, recent_history)
        except Exception as exc:
            return QueryRewriteResult(original_query, original_query, False, method="fallback", error=str(exc))

        if not self._valid_rewrite(original_query, standalone_query):
            return QueryRewriteResult(
                original_query,
                original_query,
                False,
                method="fallback",
                error="invalid_rewrite",
            )

        return QueryRewriteResult(
            original_query=original_query,
            standalone_query=standalone_query,
            rewritten=standalone_query != original_query,
            method="llm",
        )

    async def arewrite(self, query: str, history: list[ConversationMessage]) -> QueryRewriteResult:
        original_query = str(query or "").strip()
        if not original_query:
            return QueryRewriteResult(original_query, original_query, False)
        if not self.enabled:
            return QueryRewriteResult(original_query, original_query, False, method="disabled")

        recent_history = self._trim_history(history)
        if not recent_history or not self._needs_context(original_query):
            return QueryRewriteResult(original_query, original_query, False)

        heuristic_query = self._heuristic_rewrite(original_query, recent_history)
        if heuristic_query and self._valid_rewrite(original_query, heuristic_query):
            return QueryRewriteResult(
                original_query=original_query,
                standalone_query=heuristic_query,
                rewritten=heuristic_query != original_query,
                method="heuristic",
            )

        if not self.use_llm or self.chat_client is None:
            return QueryRewriteResult(original_query, original_query, False, method="fallback")

        try:
            rewrite = self._arewrite_with_llm(original_query, recent_history)
            standalone_query = (
                await asyncio.wait_for(rewrite, timeout=self.timeout_seconds)
                if self.timeout_seconds is not None
                else await rewrite
            )
        except Exception as exc:
            return QueryRewriteResult(original_query, original_query, False, method="fallback", error=str(exc))

        if not self._valid_rewrite(original_query, standalone_query):
            return QueryRewriteResult(
                original_query,
                original_query,
                False,
                method="fallback",
                error="invalid_rewrite",
            )

        return QueryRewriteResult(
            original_query=original_query,
            standalone_query=standalone_query,
            rewritten=standalone_query != original_query,
            method="llm",
        )

    def _needs_context(self, query: str) -> bool:
        if INDEPENDENT_SOURCE_PATTERN.search(query):
            return False
        if any(pattern.search(query) for pattern in DEPENDENT_PATTERNS):
            return True
        word_count = len(re.findall(r"\w+", query, re.UNICODE))
        if word_count <= 4 and not EXPLICIT_INDEPENDENT_PATTERN.search(query):
            return True
        return False

    def _heuristic_rewrite(self, query: str, history: list[ConversationMessage]) -> str | None:
        previous_user = self._last_user_message(history)
        if not previous_user:
            return None

        comparison_subjects = self._extract_comparison_subjects(previous_user)
        if comparison_subjects and re.search(r"\b(deles|delas|which one|which of them|them)\b", query, re.IGNORECASE):
            return self._join_question_subject(query, f"entre {comparison_subjects}")

        anchor = self._question_anchor(previous_user)
        entity = self._extract_entity(previous_user)
        if not entity:
            entity = self._extract_entity(self._last_assistant_message(history) or "")

        if query.casefold().strip(" ?.!") in {"e ele", "e ela", "and him", "and her", "what about him", "what about her"}:
            if entity:
                return f"{previous_user.rstrip('?.!')} sobre {entity}?"
            return previous_user

        if anchor and query.lower().startswith(("e ", "and ")):
            suffix = re.sub(r"^\s*(e|and)\s+", "", query, flags=re.IGNORECASE).strip(" ?.! ")
            return f"{anchor} {suffix}?"

        if anchor and re.match(r"^\s*(quanto custa|how much|what price|qual o preço|qual e o preco)\b", query, re.IGNORECASE):
            if entity:
                return self._join_question_subject(query, entity)
            return self._join_question_subject(query, anchor)

        if entity and re.search(r"\b(ele|ela|he|she|it)\b", query, re.IGNORECASE):
            return re.sub(
                r"\b(ele|ela|he|she|it)\b",
                entity,
                query,
                count=1,
                flags=re.IGNORECASE,
            )

        return None

    def _rewrite_with_llm(self, query: str, history: list[ConversationMessage]) -> str:
        messages = self._rewrite_messages(query, history)
        response = self.chat_client.chat(messages, model=self.model)
        return self._parse_rewrite_response(response)

    async def _arewrite_with_llm(self, query: str, history: list[ConversationMessage]) -> str:
        messages = self._rewrite_messages(query, history)
        async_chat = getattr(self.chat_client, "achat", None)
        if async_chat is not None:
            response = await async_chat(messages, model=self.model)
        else:
            response = await asyncio.to_thread(self.chat_client.chat, messages, model=self.model)
        return self._parse_rewrite_response(response)

    def _rewrite_messages(self, query: str, history: list[ConversationMessage]) -> list[dict[str, str]]:
        history_text = self._format_history(history)
        return [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest question into a standalone document-search query only when it depends on recent conversation context. "
                    "Do not answer the question. Do not infer new facts. Do not correct facts from the history. "
                    "Do not treat assistant messages as a source of truth. Use history only to resolve linguistic references, ellipses, pronouns, and omitted entities. "
                    "Preserve the user's meaning and the language of the latest question. "
                    "Return only compact JSON with keys standalone_query and rewritten."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "recent_history": history_text,
                        "latest_query": query,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _parse_rewrite_response(self, response: Any) -> str:
        content = str(getattr(response, "content", "") or "").strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("rewriter returned non-object JSON")
        return str(data.get("standalone_query") or "").strip()

    def _format_history(self, history: list[ConversationMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in self._trim_history(history)]

    def _trim_history(self, history: list[ConversationMessage]) -> list[ConversationMessage]:
        if self.max_history_messages <= 0:
            return []
        trimmed = list(history[-self.max_history_messages :])
        total = 0
        selected: list[ConversationMessage] = []
        for message in reversed(trimmed):
            total += len(message.content)
            if total > self.max_history_chars and selected:
                break
            selected.append(message)
        return list(reversed(selected))

    def _valid_rewrite(self, original_query: str, standalone_query: str) -> bool:
        text = str(standalone_query or "").strip()
        if not text:
            return False
        if len(text) > max(500, len(original_query) * 8):
            return False
        if "\n\n" in text:
            return False
        return True

    def _last_user_message(self, history: list[ConversationMessage]) -> str | None:
        for message in reversed(history):
            if message.role == "user" and message.content.strip():
                return message.content.strip()
        return None

    def _last_assistant_message(self, history: list[ConversationMessage]) -> str | None:
        for message in reversed(history):
            if message.role == "assistant" and message.content.strip():
                return message.content.strip()
        return None

    def _question_anchor(self, previous_user: str) -> str | None:
        text = previous_user.strip(" ?.! ")
        match = re.match(r"^(qual|what)\s+(?:é|e|is|was|were)\s+(.+)$", text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        match = re.match(r"^(me fale sobre|fale sobre|tell me about)\s+(.+)$", text, re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return text or None

    def _join_question_subject(self, query: str, subject: str) -> str:
        cleaned_query = query.strip(" ?.! ")
        cleaned_subject = subject.strip(" ?.! ")
        if not cleaned_subject:
            return query
        return f"{cleaned_query} {cleaned_subject}?"

    def _extract_entity(self, text: str) -> str | None:
        match = ENTITY_PATTERN.search(text)
        if match:
            return match.group(0)
        filename = INDEPENDENT_SOURCE_PATTERN.search(text)
        if filename:
            return filename.group(0)
        return None

    def _extract_comparison_subjects(self, text: str) -> str | None:
        match = COMPARE_PATTERN.search(text)
        if not match:
            return None
        left = match.group(2).strip(" ?.! ")
        right = match.group(3).strip(" ?.! ")
        if not left or not right:
            return None
        return f"{left} e {right}"
