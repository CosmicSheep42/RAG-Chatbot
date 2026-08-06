from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import re
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)
CODE_PATTERN = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ]{1,12}[-_/][\wÀ-ÖØ-öø-ÿ0-9]{1,24}\b", re.UNICODE)
COMPARISON_CONNECTOR_PATTERN = re.compile(
    r"\b(?:vs|versus|contra|between|among|entre|comparad[oa]s?\s+a|compared\s+(?:with|to))\b",
    re.IGNORECASE,
)
METADATA_FIELD_PATTERN = re.compile(
    r"\b(?:source|sources|fonte|fontes|file|files|arquivo|arquivos|document|documents|documento|documentos|"
    r"page|pages|pagina|paginas|página|páginas|slide|slides|sheet|sheets|aba|abas|planilha|planilhas|"
    r"metadata|metadados|title|titles|titulo|titulos|título|títulos|section|sections|secao|secoes|seção|seções)\b",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"\b[\wÀ-ÖØ-öø-ÿ ._-]+\.(?:txt|md|markdown|pdf|docx|pptx|xlsx|csv|json|html?)\b",
    re.IGNORECASE,
)
COLLECTION_SCOPE_PATTERN = re.compile(
    r"\b(?:"
    r"todos\s+os\s+(?:documentos|arquivos)|"
    r"todas\s+as\s+fontes|"
    r"toda\s+a\s+cole[cç][aã]o|"
    r"em\s+toda\s+a\s+cole[cç][aã]o|"
    r"na\s+base\s+de\s+documentos|"
    r"em\s+todos\s+os\s+(?:documentos|arquivos)|"
    r"entre\s+(?:todos\s+os\s+)?(?:documentos|arquivos)|"
    r"considerando\s+todos\s+os\s+arquivos|"
    r"all\s+(?:documents|files)|"
    r"entire\s+collection|"
    r"across\s+(?:all\s+|the\s+)?(?:documents|files)|"
    r"whole\s+corpus|"
    r"document\s+collection"
    r")\b",
    re.IGNORECASE,
)
SOURCE_SCOPE_PATTERN = re.compile(
    r"\b(?:"
    r"manual|handbook|contrato|contract|relat[oó]rio|report|"
    r"documenta[cç][aã]o|documentation|memo|policy|procedure|specification"
    r")\b",
    re.IGNORECASE,
)
MULTI_SOURCE_PATTERN = re.compile(
    r"\b(?:entre|between|across|among)\s+(?:as\s+|os\s+|the\s+)?(?:fontes|sources|documentos|documents|arquivos|files)\b",
    re.IGNORECASE,
)


class QueryIntent(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    BROAD_SEMANTIC = "broad_semantic"
    ENUMERATION = "enumeration"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    METADATA_LOOKUP = "metadata_lookup"
    UNKNOWN = "unknown"

    SPECIFIC_LOOKUP = "factual_lookup"


class QueryScope(str, Enum):
    LOCAL = "local"
    SOURCE = "source"
    MULTI_SOURCE = "multi_source"
    COLLECTION = "collection"


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True)
class RetrievalPolicy:
    intent: QueryIntent
    retrieval_limit: int
    context_limit: int
    expand_by_source: bool
    max_sources: int
    source_chunk_limit: int
    coverage_min_chunks: int
    coverage_min_ratio: float
    retry_retrieval_limit: int
    retry_context_limit: int
    dense_k: int | None = None
    dense_fetch_k: int | None = None
    sparse_k: int | None = None
    partial_when_incomplete: bool = False
    scope: QueryScope = QueryScope.LOCAL


@dataclass(frozen=True)
class IntentSignals:
    imperative_verbs: frozenset[str] = frozenset()
    nouns: frozenset[str] = frozenset()
    phrases: frozenset[str] = frozenset()
    question_words: frozenset[str] = frozenset()

    def score(self, query_text: str, tokens: set[str]) -> int:
        return (
            len(tokens & self.imperative_verbs) * 3
            + len(tokens & self.question_words) * 2
            + len(tokens & self.nouns)
            + sum(3 for phrase in self.phrases if phrase in query_text)
        )


@dataclass(frozen=True)
class PolicyTuning:
    candidate_multiplier: float
    final_multiplier: float
    min_retrieval_limit: int
    min_context_limit: int
    expand_by_source: bool
    max_sources: int
    source_chunk_multiplier: float
    min_source_chunk_limit: int
    coverage_min_chunks: int
    coverage_min_ratio: float
    retry_candidate_multiplier: float
    retry_context_multiplier: float
    min_retry_retrieval_limit: int
    min_retry_context_limit: int
    dense_multiplier: float = 1.0
    dense_fetch_multiplier: float = 2.0
    sparse_multiplier: float = 1.0
    partial_when_incomplete: bool = False


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    signals: dict[QueryIntent, IntentSignals] = field(default_factory=dict)
    policies: dict[QueryIntent, PolicyTuning] = field(default_factory=dict)

    @classmethod
    def default(cls) -> RetrievalPolicyConfig:
        return cls(
            signals={
                QueryIntent.ENUMERATION: IntentSignals(
                    imperative_verbs=frozenset({"list", "listar", "liste", "enumere", "enumerate", "mostre", "show"}),
                    nouns=frozenset(
                        {
                            "all",
                            "todos",
                            "todas",
                            "cada",
                            "every",
                            "items",
                            "itens",
                            "entries",
                            "entradas",
                            "records",
                            "registros",
                            "requirements",
                            "requisitos",
                            "sections",
                            "secoes",
                            "seções",
                            "documents",
                            "documentos",
                            "files",
                            "arquivos",
                            "sources",
                            "fontes",
                            "examples",
                            "exemplos",
                        }
                    ),
                    phrases=frozenset({"complete list", "full list", "lista completa"}),
                    question_words=frozenset({"which", "what", "quais", "qual"}),
                ),
                QueryIntent.SUMMARY: IntentSignals(
                    imperative_verbs=frozenset({"summarize", "summarise", "resuma", "sintetize", "recap"}),
                    nouns=frozenset({"summary", "resumo", "sintese", "síntese", "overview", "visao", "visão"}),
                    phrases=frozenset({"high level", "em alto nivel", "em alto nível", "in brief", "em resumo"}),
                ),
                QueryIntent.COMPARISON: IntentSignals(
                    imperative_verbs=frozenset({"compare", "comparar", "compare", "contraste", "contrast"}),
                    nouns=frozenset({"difference", "differences", "diferenca", "diferença", "diferencas", "diferenças", "similarities", "semelhancas", "semelhanças"}),
                    phrases=frozenset({"pros and cons", "prós e contras", "pontos em comum"}),
                ),
                QueryIntent.AGGREGATION: IntentSignals(
                    imperative_verbs=frozenset({"count", "conte", "calcule", "calculate", "sum", "some", "totalize", "totalizar"}),
                    nouns=frozenset({"total", "totals", "media", "média", "average", "count", "contagem", "quantos", "quantas", "how", "many"}),
                    phrases=frozenset({"how many", "quantos sao", "quantos são", "quantas sao", "quantas são"}),
                ),
                QueryIntent.METADATA_LOOKUP: IntentSignals(
                    imperative_verbs=frozenset({"find", "encontre", "localize", "cite", "citar"}),
                    nouns=frozenset({"where", "onde", "which", "qual", "quais"}),
                    phrases=frozenset({"which document", "qual documento", "which file", "qual arquivo", "what source", "qual fonte"}),
                ),
                QueryIntent.FACTUAL_LOOKUP: IntentSignals(
                    question_words=frozenset({"what", "who", "where", "when", "which", "qual", "quais", "quem", "onde", "quando"}),
                    nouns=frozenset({"exact", "exato", "exata", "specific", "especifico", "específico", "value", "valor", "date", "data", "definition", "definicao", "definição"}),
                    phrases=frozenset({"what is", "who is", "where is", "when is", "o que e", "o que é", "quem e", "quem é"}),
                ),
                QueryIntent.BROAD_SEMANTIC: IntentSignals(
                    imperative_verbs=frozenset({"explain", "explique", "describe", "descreva", "analyze", "analise"}),
                    nouns=frozenset({"why", "porque", "porquê", "how", "como", "relationship", "relacao", "relação", "context", "contexto"}),
                    phrases=frozenset({"tell me about", "fale sobre", "me explique", "explain how", "explique como"}),
                ),
            },
            policies={
                QueryIntent.FACTUAL_LOOKUP: PolicyTuning(1.0, 1.0, 0, 0, False, 1, 1.0, 0, 1, 0.20, 1.6, 1.0, 16, 0),
                QueryIntent.BROAD_SEMANTIC: PolicyTuning(1.5, 1.6, 16, 8, False, 2, 1.5, 8, 2, 0.25, 2.0, 1.4, 24, 10, dense_multiplier=1.5, sparse_multiplier=1.2),
                QueryIntent.ENUMERATION: PolicyTuning(2.4, 4.0, 24, 24, True, 2, 3.0, 24, 4, 0.60, 3.2, 5.0, 32, 32, dense_multiplier=2.0, dense_fetch_multiplier=4.0, sparse_multiplier=2.0, partial_when_incomplete=True),
                QueryIntent.SUMMARY: PolicyTuning(1.2, 1.2, 12, 6, False, 1, 1.0, 0, 2, 0.30, 1.6, 1.6, 16, 8),
                QueryIntent.COMPARISON: PolicyTuning(2.0, 2.5, 20, 10, True, 3, 2.0, 10, 3, 0.45, 3.0, 3.0, 32, 16, dense_multiplier=2.0, sparse_multiplier=1.5),
                QueryIntent.AGGREGATION: PolicyTuning(2.5, 4.0, 24, 20, True, 3, 3.0, 20, 4, 0.55, 3.5, 4.5, 36, 28, dense_multiplier=2.0, dense_fetch_multiplier=4.0, sparse_multiplier=2.0, partial_when_incomplete=True),
                QueryIntent.METADATA_LOOKUP: PolicyTuning(1.4, 1.2, 12, 6, False, 2, 1.0, 6, 1, 0.20, 2.0, 1.4, 20, 8, dense_multiplier=1.0, sparse_multiplier=1.4),
                QueryIntent.UNKNOWN: PolicyTuning(1.0, 1.0, 0, 0, False, 1, 1.0, 0, 1, 0.20, 1.6, 1.0, 16, 0),
            },
        )


class RetrievalPolicyResolver:
    def __init__(
        self,
        *,
        default_candidate_k: int,
        default_final_k: int,
        config: RetrievalPolicyConfig | None = None,
    ) -> None:
        self.default_candidate_k = default_candidate_k
        self.default_final_k = default_final_k
        self.config = config or RetrievalPolicyConfig.default()

    def resolve(self, query: str) -> RetrievalPolicy:
        intent = self._detect_intent(query)
        scope = self._detect_scope(query, intent)
        tuning = self.config.policies.get(intent, self.config.policies[QueryIntent.UNKNOWN])
        policy = self._policy_from_tuning(intent, tuning, scope)

        if intent in {QueryIntent.ENUMERATION, QueryIntent.AGGREGATION} and scope not in {
            QueryScope.COLLECTION,
            QueryScope.MULTI_SOURCE,
        }:
            policy = replace(policy, max_sources=1, context_limit=max(policy.source_chunk_limit, policy.context_limit // 2))

        return policy

    def _policy_from_tuning(self, intent: QueryIntent, tuning: PolicyTuning, scope: QueryScope) -> RetrievalPolicy:
        retrieval_limit = max(int(self.default_candidate_k * tuning.candidate_multiplier), tuning.min_retrieval_limit)
        context_limit = max(int(self.default_final_k * tuning.final_multiplier), tuning.min_context_limit)
        source_chunk_limit = max(int(self.default_final_k * tuning.source_chunk_multiplier), tuning.min_source_chunk_limit)
        retry_retrieval_limit = max(int(self.default_candidate_k * tuning.retry_candidate_multiplier), tuning.min_retry_retrieval_limit)
        retry_context_limit = max(int(self.default_final_k * tuning.retry_context_multiplier), tuning.min_retry_context_limit, context_limit)

        return RetrievalPolicy(
            intent=intent,
            retrieval_limit=retrieval_limit,
            context_limit=context_limit,
            expand_by_source=tuning.expand_by_source,
            max_sources=tuning.max_sources,
            source_chunk_limit=source_chunk_limit,
            coverage_min_chunks=tuning.coverage_min_chunks,
            coverage_min_ratio=tuning.coverage_min_ratio,
            retry_retrieval_limit=retry_retrieval_limit,
            retry_context_limit=retry_context_limit,
            dense_k=max(int(self.default_candidate_k * tuning.dense_multiplier), 1),
            dense_fetch_k=max(int(self.default_candidate_k * tuning.dense_fetch_multiplier), 1),
            sparse_k=max(int(self.default_candidate_k * tuning.sparse_multiplier), 1),
            partial_when_incomplete=tuning.partial_when_incomplete,
            scope=scope,
        )

    def _detect_intent(self, query: str) -> QueryIntent:
        query_text = query.casefold()
        tokens = _tokenize(query)

        if self._is_metadata_lookup(query_text, tokens):
            return QueryIntent.METADATA_LOOKUP
        if self._is_enumeration_query(query_text, tokens):
            return QueryIntent.ENUMERATION
        if self._is_comparison_query(query_text, tokens):
            return QueryIntent.COMPARISON
        if self._is_aggregation_query(query_text, tokens):
            return QueryIntent.AGGREGATION

        ranked_intents = self._rank_by_signals(
            query_text,
            tokens,
            (
                QueryIntent.SUMMARY,
                QueryIntent.BROAD_SEMANTIC,
                QueryIntent.FACTUAL_LOOKUP,
            ),
        )
        if ranked_intents:
            return ranked_intents[0]

        if CODE_PATTERN.search(query):
            return QueryIntent.FACTUAL_LOOKUP

        return QueryIntent.UNKNOWN

    def _rank_by_signals(
        self,
        query_text: str,
        tokens: set[str],
        intents: Iterable[QueryIntent],
    ) -> list[QueryIntent]:
        scored: list[tuple[int, QueryIntent]] = []
        for intent in intents:
            signals = self.config.signals.get(intent)
            if signals is None:
                continue
            score = signals.score(query_text, tokens)
            if score > 0:
                scored.append((score, intent))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [intent for _score, intent in scored]

    def _is_enumeration_query(self, query_text: str, tokens: set[str]) -> bool:
        signals = self.config.signals[QueryIntent.ENUMERATION]
        action_score = len(tokens & signals.imperative_verbs) + sum(1 for phrase in signals.phrases if phrase in query_text)
        scope_score = len(tokens & signals.nouns)
        interrogative_score = len(tokens & signals.question_words)
        return action_score > 0 and (scope_score > 0 or interrogative_score > 0)

    def _is_comparison_query(self, query_text: str, tokens: set[str]) -> bool:
        signals = self.config.signals[QueryIntent.COMPARISON]
        return signals.score(query_text, tokens) >= 3 or bool(COMPARISON_CONNECTOR_PATTERN.search(query_text))

    def _is_aggregation_query(self, query_text: str, tokens: set[str]) -> bool:
        signals = self.config.signals[QueryIntent.AGGREGATION]
        quantity_markers = {"quantos", "quantas", "quanto", "quanta"}
        return (
            signals.score(query_text, tokens) >= 4
            or bool(tokens & quantity_markers)
            or "how many" in query_text
        )

    def _is_metadata_lookup(self, query_text: str, tokens: set[str]) -> bool:
        signals = self.config.signals[QueryIntent.METADATA_LOOKUP]
        has_lookup_signal = signals.score(query_text, tokens) > 0
        has_metadata_field = bool(METADATA_FIELD_PATTERN.search(query_text))
        return has_lookup_signal and has_metadata_field

    def _detect_scope(self, query: str, intent: QueryIntent) -> QueryScope:
        query_text = query.casefold()

        has_explicit_source = bool(DOCUMENT_REFERENCE_PATTERN.search(query_text))
        has_collection_signal = bool(COLLECTION_SCOPE_PATTERN.search(query_text))
        has_multi_source_signal = bool(MULTI_SOURCE_PATTERN.search(query_text))

        if has_explicit_source and not has_multi_source_signal and not has_collection_signal:
            return QueryScope.SOURCE

        if has_collection_signal:
            return QueryScope.COLLECTION

        if has_multi_source_signal or (
            intent == QueryIntent.COMPARISON
            and bool({"documents", "documentos", "files", "arquivos", "sources", "fontes"} & _tokenize(query_text))
        ):
            return QueryScope.MULTI_SOURCE

        if has_explicit_source or SOURCE_SCOPE_PATTERN.search(query_text):
            return QueryScope.SOURCE

        return QueryScope.LOCAL
