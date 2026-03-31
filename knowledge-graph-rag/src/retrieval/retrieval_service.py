# FILE: src/retrieval/retrieval_service.py
# CHANGES: Added filename lookup and optional LLM-based query expansion while preserving the existing retrieval entrypoints.

import asyncio
from typing import List, Optional

from loguru import logger

from src.config import Settings
from src.graph.chunk_repository import ChunkRepository
from src.graph_models import RetrievalHit
from src.ingestion.embedder import Embedder
from src.llm.llm_client import LLMClient
from src.retrieval.bm25_search import BM25Search
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.hybrid_ranker import HybridRanker
from src.retrieval.vector_search import VectorSearch


class RetrievalService:
    """Main orchestrator for all retrieval operations."""

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        graph_traversal: GraphTraversal,
        hybrid_ranker: HybridRanker,
        embedder: Embedder,
        settings: Settings,
        llm_client: Optional[LLMClient] = None,
        chunk_repo: Optional[ChunkRepository] = None,
    ) -> None:
        """Initialize retrieval dependencies."""
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.graph_traversal = graph_traversal
        self.hybrid_ranker = hybrid_ranker
        self.embedder = embedder
        self.settings = settings
        self.llm_client = llm_client
        self.chunk_repo = chunk_repo

    async def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """Retrieve chunk-only knowledge results for a query."""
        embedding = self.embedder.embed_text(query)
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_search.search_chunks(embedding, top_k=self.settings.vector_top_k),
            self.bm25_search.search_chunks(query, top_k=self.settings.bm25_top_k),
        )
        seed_ids = [hit.id for hit in vector_hits[:3]]
        graph_hits = await self.graph_traversal.expand_from_chunks(seed_ids)
        return self.hybrid_ranker.rank(vector_hits, bm25_hits, graph_hits, top_k)

    async def retrieve_memory(self, query: str, user_id: str, top_k: int = 5) -> List[RetrievalHit]:
        """Retrieve memory-only results for a query and specific user."""
        embedding = self.embedder.embed_text(query)
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_search.search_memories(embedding, user_id, top_k=self.settings.vector_top_k),
            self.bm25_search.search_memories(query, user_id, top_k=self.settings.bm25_top_k),
        )
        return self.hybrid_ranker.rank(vector_hits, bm25_hits, [], top_k)

    async def retrieve_by_filename(self, filename: str, top_k: int = 5) -> List[RetrievalHit]:
        """
        Direct lookup of chunks by source filename.
        Used when user asks about a specific file or paper.
        """
        if self.chunk_repo is None:
            logger.warning("Filename retrieval requested but chunk repository is not configured")
            return []

        results = await self.chunk_repo.get_chunks_by_source(source_file=filename, limit=top_k)
        if not results and not filename.lower().endswith((".pdf", ".txt")):
            results = await self.chunk_repo.get_chunks_by_source(source_file=f"{filename}.pdf", limit=top_k)
        if not results and not filename.lower().endswith(".txt"):
            results = await self.chunk_repo.get_chunks_by_source(source_file=f"{filename}.txt", limit=top_k)

        if not results:
            logger.warning("No chunks found for filename: {}", filename)

        return [
            RetrievalHit(
                id=result["chunk_id"],
                content=result["content"],
                score=1.0,
                source_type="chunk",
                source_file=result["source_file"],
                page_number=result["page_number"],
            )
            for result in results
        ]

    async def retrieve_with_query_expansion(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """
        Two-stage retrieval with LLM query expansion for better recall.
        Stage 1: Expand query using LLM to add synonyms and related terms.
        Stage 2: Run expanded query through standard retrieve_knowledge().
        Falls back to original query if expansion fails.
        """
        if self.llm_client is None:
            return await self.retrieve_knowledge(query, top_k=top_k)

        try:
            expansion_prompt = (
                f"Expand this search query with 5-8 related academic terms and synonyms. "
                f"Return only the expanded query string, no explanation:\n\n{query}"
            )
            expanded = await self.llm_client.complete(
                "You expand search queries for academic paper retrieval. Return only the expanded query string.",
                expansion_prompt,
                max_tokens=80,
                temperature=0.0,
            )
            expanded_query = f"{query} {expanded}".strip()
            logger.debug(
                "Expanded query: '{}'",
                expanded_query[:100] + ("..." if len(expanded_query) > 100 else ""),
            )
        except Exception as exc:
            logger.warning("Query expansion failed, using original: {}", exc)
            expanded_query = query

        return await self.retrieve_knowledge(expanded_query, top_k=top_k)
