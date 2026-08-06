from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from local_rag_chatbot.rag.context_assembler import CoverageReport
    from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceAssessment
    from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicy


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    documents: tuple[Any, ...]
    candidates: tuple[Any, ...]
    confidence: RetrievalConfidenceAssessment | None
    policy: RetrievalPolicy | None
    coverage: CoverageReport | None
    reason_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def confidence_score(self) -> float:
        if self.confidence is None:
            return 0.0
        return self.confidence.score
