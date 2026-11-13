from __future__ import annotations

import asyncio
import re
from typing import Any, TYPE_CHECKING

from local_rag_chatbot.observability import metrics
from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.rag.contracts import ChatModel
from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy

if TYPE_CHECKING:
    from langchain_core.prompts import PromptTemplate
    from local_rag_chatbot.rag.context_assembler import CoverageReport


class AnswerService:
    def __init__(
        self,
        *,
        chat_model: ChatModel,
        generation_model: str,
        prompt_loader: Any,
        answer_prompt_template: str,
        system_prompt_template: str,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.generation_model = generation_model
        self.prompt_loader = prompt_loader
        self.answer_prompt_template = answer_prompt_template
        self.system_prompt_template = system_prompt_template
        self.context_builder = context_builder or ContextBuilder()

    def build_prompt(
        self,
        policy: RetrievalPolicy | None = None,
        coverage: CoverageReport | None = None,
        partial_context: bool = False,
    ) -> PromptTemplate:
        from langchain_core.prompts import PromptTemplate

        instruction = self.policy_instruction(policy, coverage=coverage, partial_context=partial_context) or ""
        instruction += (
            " Use only facts stated in the selected context; do not use prior knowledge or combine unrelated passages to infer a rule. "
            "For each requested part, verify that the selected text directly supports the response. If any part is missing, say so plainly and leave it unanswered. "
            "Cite each factual claim with the directly supporting valid reference such as [S1]; never invent a reference."
        )
        return PromptTemplate.from_template(self.answer_prompt_template).partial(
            master_prompt=self.prompt_loader(),
            policy_instruction=instruction,
        )

    def build_chat_system_prompt(
        self,
        query: str,
        documents: list[Any],
        policy: RetrievalPolicy,
        *,
        coverage: CoverageReport | None = None,
        partial_context: bool = False,
    ) -> str:
        instruction = self.policy_instruction(policy, coverage=coverage, partial_context=partial_context) or ""
        instruction += (
            " Use only facts stated in the selected context; do not use prior knowledge or combine unrelated passages to infer a rule. "
            "For each requested part, verify that the selected text directly supports the response. If any part is missing, say so plainly and leave it unanswered. "
            "Cite each factual claim with the directly supporting valid reference such as [S1]; never invent a reference."
        )
        return self.system_prompt_template.format(
            master_prompt=self.prompt_loader(),
            user_question=query,
            current_context=self.context_builder.format_context(documents),
            retrieval_guidance=instruction,
        )

    def generate(
        self,
        query: str,
        documents: list[Any],
        policy: RetrievalPolicy,
        *,
        coverage: CoverageReport | None = None,
        partial_context: bool = False,
    ) -> str:
        prompt = self.build_prompt(policy, coverage=coverage, partial_context=partial_context).format(
            context=self.context_builder.format_context(documents),
            question=query,
        )
        with metrics.timer("final_generation_seconds"):
            response = self.chat_model.chat(
                [{"role": "user", "content": prompt}],
                model=self.generation_model,
            )
        return self.validate_citations(response.content.strip(), documents)

    async def agenerate(
        self,
        query: str,
        documents: list[Any],
        policy: RetrievalPolicy,
        *,
        coverage: CoverageReport | None = None,
        partial_context: bool = False,
    ) -> str:
        prompt = self.build_prompt(policy, coverage=coverage, partial_context=partial_context).format(
            context=self.context_builder.format_context(documents),
            question=query,
        )
        with metrics.timer("final_generation_seconds"):
            response = await self._achat(
                [{"role": "user", "content": prompt}],
                model=self.generation_model,
            )
        return self.validate_citations(response.content.strip(), documents)

    def validate_citations(self, answer: str, documents: list[Any]) -> str:
        valid_ids = {citation["citation_id"] for citation in self.context_builder.build_citations(documents)}
        return re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0) if match.group(1) in valid_ids else "",
            answer,
        )

    async def _achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        async_chat = getattr(self.chat_model, "achat", None)
        if async_chat is not None:
            return await async_chat(messages, tools=tools, model=model)
        return await asyncio.to_thread(self.chat_model.chat, messages, tools=tools, model=model)

    def policy_instruction(
        self,
        policy: RetrievalPolicy | None,
        *,
        coverage: CoverageReport | None = None,
        partial_context: bool = False,
    ) -> str:
        if policy is None:
            return ""

        if policy.intent == QueryIntent.ENUMERATION:
            if partial_context:
                coverage_note = ""
                if coverage is not None and coverage.dominant_source:
                    coverage_note = f" The strongest source is {coverage.dominant_source}, but the retrieved context may be incomplete."
                return (
                    "The question asks for a complete list, but the retrieved context may be incomplete. "
                    "Provide a cautious partial answer based only on what is confirmed in the context, and explicitly say that the list may not be exhaustive."
                    f"{coverage_note} "
                    "Enumerate every distinct item that appears in the context, preserve the document's original order when possible, and do not stop early after a few examples. "
                    "Use headings, numbered sections, tables, and bullet structure to identify distinct entries. "
                    "Do not promote supporting notes or subsections into separate entries unless the context presents them as standalone entries."
                )

            return (
                "The question asks for a complete list. Enumerate every distinct item that appears in the context, "
                "preserve the document's original order when possible, and do not stop early after a few examples. "
                "Use headings, numbered sections, tables, and bullet structure to identify distinct entries. "
                "Do not promote supporting notes or subsections into separate entries unless the context presents them as standalone entries."
            )

        if policy.intent == QueryIntent.SPECIFIC_LOOKUP:
            return (
                "Prioritize exact factual values from the context. Keep names, numbers, dates, and definitions faithful, "
                "but still express the answer naturally rather than copying large passages."
            )
        if policy.intent == QueryIntent.SUMMARY:
            return "Provide a concise synthesis of the context. Do not invent details, but you may reorganize related facts into a cleaner summary."
        if policy.intent == QueryIntent.COMPARISON:
            return "Compare only points supported by the context. Separate confirmed similarities and differences, and say when one side lacks enough evidence."
        if policy.intent == QueryIntent.AGGREGATION:
            return "Compute counts, totals, or grouped results only from evidence present in the context. Show the basis for the aggregation briefly and state when the retrieved context may be incomplete."
        if policy.intent == QueryIntent.METADATA_LOOKUP:
            return "Answer with the relevant source, file, page, slide, sheet, section, or other metadata visible in the context. Do not infer locations that are not present in metadata or quoted context."
        if policy.intent == QueryIntent.BROAD_SEMANTIC:
            return "Synthesize related evidence across the retrieved context, keeping the answer grounded and noting any important uncertainty."
        return ""
