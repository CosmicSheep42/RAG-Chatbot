from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_rag_chatbot.conversation import InMemoryConversationStore, QueryRewriter
from local_rag_chatbot.documents.document_repository import DocumentRepository
from local_rag_chatbot.documents.document_splitter import DocumentSplitter
from local_rag_chatbot.documents.hash_manager import HashManager
from local_rag_chatbot.documents.index_manifest import IndexManifest
from local_rag_chatbot.documents.ingestion_security import DocumentSecurityValidator, IngestionSecurityConfig
from local_rag_chatbot.documents.loaders import DocumentLoaderRegistry
from local_rag_chatbot.documents.prompt_repository import MasterPromptRepository
from local_rag_chatbot.documents.tool_config_repository import ToolCallingConfigRepository
from local_rag_chatbot.input_cleaner import TextPreprocessorPipeline
from local_rag_chatbot.rag.answer_service import AnswerService
from local_rag_chatbot.rag.chatbot import RAGChatbot
from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.rag.contracts import ChatModel, ConversationStore
from local_rag_chatbot.rag.hybrid_retriever import RetrievalTuning
from local_rag_chatbot.rag.index_manager import IndexManager
from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceEstimator
from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicyResolver
from local_rag_chatbot.rag.tool_registry import ToolRegistry
from local_rag_chatbot.router import EmbeddingRouter
from local_rag_chatbot.settings import AppSettings
from local_rag_chatbot.vectorstores import VectorStoreBackend, create_vector_store_backend


@dataclass
class IndexingServices:
    settings: AppSettings
    document_repository: DocumentRepository
    document_splitter: DocumentSplitter
    index_manager: IndexManager


@dataclass
class ApplicationServices:
    settings: AppSettings
    input_pipeline: TextPreprocessorPipeline
    conversation_store: ConversationStore | None
    chat_model: ChatModel
    index_manager: IndexManager
    context_builder: ContextBuilder
    answer_service: AnswerService
    tool_registry: ToolRegistry
    chatbot: RAGChatbot
    router: EmbeddingRouter
    query_rewriter: QueryRewriter


def create_indexing_services(
    settings: AppSettings,
    *,
    embeddings: object | None = None,
    vector_backend: VectorStoreBackend | None = None,
) -> IndexingServices:
    if embeddings is None:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    vector_backend = vector_backend or create_vector_store_backend(settings, embeddings)
    loader_registry = _create_loader_registry(settings)
    document_repository = DocumentRepository(
        settings.documents_dir,
        loader_registry=loader_registry,
        max_file_size_bytes=settings.max_document_file_size_bytes,
        security_validator=DocumentSecurityValidator(
            settings.documents_dir,
            IngestionSecurityConfig(
                max_file_size_bytes=settings.max_document_file_size_bytes,
                max_zip_entries=settings.max_zip_entries,
                max_zip_uncompressed_bytes=settings.max_zip_uncompressed_bytes,
                max_zip_compression_ratio=settings.max_zip_compression_ratio,
            ),
        ),
        ignored_paths=_ignored_file_paths(settings),
        ignored_directory_paths=_ignored_directory_paths(settings),
    )
    document_splitter = DocumentSplitter(settings.chunk_size, settings.chunk_overlap)
    index_manager = IndexManager(
        settings=settings,
        document_repository=document_repository,
        splitter=document_splitter,
        hash_manager=HashManager(settings.data_dir / "rag_hash.json"),
        manifest=IndexManifest(settings.data_dir / "rag_manifest.json").load(),
        embeddings=embeddings,
        vector_backend=vector_backend,
        retrieval_tuning=_retrieval_tuning(settings),
        confidence_estimator=RetrievalConfidenceEstimator(),
    )
    return IndexingServices(
        settings=settings,
        document_repository=document_repository,
        document_splitter=document_splitter,
        index_manager=index_manager,
    )


def create_application_services(
    settings: AppSettings,
    *,
    chat_model: ChatModel | None = None,
    embeddings: object | None = None,
    conversation_store: ConversationStore | None = None,
    tool_registry: ToolRegistry | None = None,
    vector_backend: VectorStoreBackend | None = None,
) -> ApplicationServices:
    indexing = create_indexing_services(settings, embeddings=embeddings, vector_backend=vector_backend)
    input_pipeline = TextPreprocessorPipeline(config=settings.input_cleaner_config())
    if chat_model is None:
        from local_rag_chatbot.rag.ollama_tool_client import OllamaToolCallingClient

        chat_model = OllamaToolCallingClient(
            base_url=settings.ollama_base_url,
            default_model=settings.generation_model,
            temperature=settings.generation_temperature,
            num_predict=2048,
            connect_timeout_seconds=settings.ollama_connect_timeout_seconds,
            read_timeout_seconds=settings.ollama_read_timeout_seconds,
            max_response_bytes=settings.ollama_max_response_bytes,
        )
    context_builder = ContextBuilder()
    tool_config_repository = ToolCallingConfigRepository(settings.tool_calling_config_file)
    tool_calling_config = tool_config_repository.load()
    answer_service = AnswerService(
        chat_model=chat_model,
        generation_model=settings.generation_model,
        prompt_loader=MasterPromptRepository(settings.master_prompt_file).load,
        answer_prompt_template=tool_calling_config.answer_prompt_template,
        system_prompt_template=tool_calling_config.system_prompt_template,
        context_builder=context_builder,
    )
    tool_registry = tool_registry or ToolRegistry()
    policy_resolver = RetrievalPolicyResolver(
        default_candidate_k=settings.hybrid_candidate_k,
        default_final_k=settings.hybrid_final_k,
    )
    chatbot = RAGChatbot(
        settings=settings,
        input_pipeline=input_pipeline,
        index_manager=indexing.index_manager,
        policy_resolver=policy_resolver,
        answer_service=answer_service,
        context_builder=context_builder,
        tool_registry=tool_registry,
        tool_calling_config=tool_calling_config,
        chat_model=chat_model,
    )
    if conversation_store is None and settings.conversation_memory_enabled and settings.conversation_max_messages > 0:
        conversation_store = InMemoryConversationStore(
            max_messages=settings.conversation_max_messages,
            ttl_seconds=settings.conversation_ttl_seconds,
            max_sessions=settings.conversation_max_sessions,
            max_message_chars=settings.conversation_max_message_chars,
        )
    query_rewriter = QueryRewriter(
        chat_client=chat_model,
        model=settings.query_rewriting_model,
        enabled=settings.query_rewriting_enabled and settings.conversation_memory_enabled,
        use_llm=settings.query_rewriting_use_llm,
        max_history_messages=settings.conversation_max_messages,
    )
    router = EmbeddingRouter(
        settings=settings,
        embeddings=indexing.index_manager.embeddings,
        vector_backend=indexing.index_manager.vector_backend,
        input_pipeline=input_pipeline,
    )
    return ApplicationServices(
        settings=settings,
        input_pipeline=input_pipeline,
        conversation_store=conversation_store,
        chat_model=chat_model,
        index_manager=indexing.index_manager,
        context_builder=context_builder,
        answer_service=answer_service,
        tool_registry=tool_registry,
        chatbot=chatbot,
        router=router,
        query_rewriter=query_rewriter,
    )


def _create_loader_registry(settings: AppSettings) -> DocumentLoaderRegistry:
    return DocumentLoaderRegistry(
        max_pdf_pages=settings.max_pdf_pages,
        max_spreadsheet_sheets=settings.max_spreadsheet_sheets,
        max_spreadsheet_rows=settings.max_spreadsheet_rows,
        max_presentation_slides=settings.max_presentation_slides,
    )


def _ignored_file_paths(settings: AppSettings) -> set[Path]:
    paths = {
        settings.master_prompt_file.location,
        settings.tool_calling_config_file.location,
    }
    return {path for path in paths if isinstance(path, Path)}


def _ignored_directory_paths(settings: AppSettings) -> set[Path]:
    routing_dir = settings.routing_dir.location
    return {routing_dir} if isinstance(routing_dir, Path) else set()


def _retrieval_tuning(settings: AppSettings) -> RetrievalTuning:
    return RetrievalTuning(
        dense_k=settings.hybrid_dense_k,
        dense_fetch_k=settings.hybrid_dense_fetch_k,
        sparse_k=settings.hybrid_sparse_k,
        candidate_k=settings.hybrid_candidate_k,
        final_k=settings.hybrid_final_k,
        dense_lambda_mult=settings.hybrid_dense_lambda_mult,
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        metadata_weight=settings.metadata_boost_weight,
    )
