# FILE: main.py
# CHANGES: Added memory fulltext health-check and rebuild flow after profile selection.

import asyncio

from loguru import logger

from src.chat.chat_session import ChatSession
from src.config import Settings
from src.graph.chunk_repository import ChunkRepository
from src.graph.entity_repository import EntityRepository
from src.graph.memory_repository import MemoryRepository
from src.graph.neo4j_client import Neo4jClient
from src.graph.profile_repository import ProfileRepository
from src.graph.schema import (
    check_memory_index_health,
    initialize_schema,
    rebuild_memory_fulltext_index,
    verify_indexes,
)
from src.ingestion.chunk_splitter import ChunkSplitter
from src.ingestion.document_parser import DocumentParser
from src.ingestion.embedder import Embedder
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.llm.context_assembler import ContextAssembler
from src.llm.fact_extractor import FactExtractor
from src.llm.intent_classifier import IntentClassifier
from src.llm.llm_client import LLMClient
from src.memory.memory_service import MemoryService
from src.memory.profile_manager import ProfileManager
from src.retrieval.bm25_search import BM25Search
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.hybrid_ranker import HybridRanker
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.vector_search import VectorSearch


def _print_banner() -> None:
    """Print a simple ASCII startup banner."""
    print("===================================")
    print(" Knowledge Graph RAG Assistant")
    print("===================================\n")


async def main() -> None:
    """Initialize dependencies and run the terminal chat loop."""
    _print_banner()
    settings = Settings()
    neo4j_client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    await neo4j_client.connect()
    try:
        await initialize_schema(neo4j_client)
        await verify_indexes(neo4j_client)

        embedder = Embedder(settings.embedding_model, settings.embedding_device)
        chunk_repo = ChunkRepository(neo4j_client)
        entity_repo = EntityRepository(neo4j_client)
        memory_repo = MemoryRepository(neo4j_client)
        profile_repo = ProfileRepository(neo4j_client)

        parser = DocumentParser()
        splitter = ChunkSplitter(settings.chunk_size, settings.chunk_overlap)
        extractor = EntityExtractor()
        pipeline = IngestionPipeline(
            neo4j_client=neo4j_client,
            chunk_repo=chunk_repo,
            entity_repo=entity_repo,
            embedder=embedder,
            extractor=extractor,
            splitter=splitter,
            parser=parser,
            settings=settings,
        )

        llm = LLMClient(settings.groq_api_key, settings.groq_model)

        vector_search = VectorSearch(neo4j_client)
        bm25_search = BM25Search(neo4j_client)
        graph_traversal = GraphTraversal(neo4j_client)
        hybrid_ranker = HybridRanker()
        retrieval = RetrievalService(
            vector_search=vector_search,
            bm25_search=bm25_search,
            graph_traversal=graph_traversal,
            hybrid_ranker=hybrid_ranker,
            embedder=embedder,
            settings=settings,
            llm_client=llm,
            chunk_repo=chunk_repo,
        )

        classifier = IntentClassifier(llm)
        fact_extractor = FactExtractor(llm)
        assembler = ContextAssembler()
        memory_service = MemoryService(memory_repo, embedder, fact_extractor)

        profile_manager = ProfileManager(neo4j_client)
        profile = await profile_manager.get_or_create_profile()
        user_id = profile.user_id
        memory_index_ok = await check_memory_index_health(
            neo4j_client, user_id
        )
        if not memory_index_ok:
            await rebuild_memory_fulltext_index(neo4j_client)
        username = profile.username

        memories = await memory_service.load_for_session(user_id, limit=30)
        if memories:
            print(f"[Loaded {len(memories)} memories]")

        chat = ChatSession(
            neo4j_client=neo4j_client,
            pipeline=pipeline,
            retrieval=retrieval,
            memory_service=memory_service,
            memory_repo=memory_repo,
            classifier=classifier,
            assembler=assembler,
            llm=llm,
            user_id=user_id,
            chunk_repo=chunk_repo,
            username=username,
            profile_repo=profile_repo,
            profile_manager=profile_manager,
        )
        await chat.run()
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
