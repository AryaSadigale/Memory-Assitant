# FILE: src/retrieval/retrieval_service.py
# CHANGES: Added self-query memory supplementation and low-relevance filtering while preserving user-scoped retrieval.

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

    def _is_generic_self_query(self, query: str) -> bool:
        """Return True if the query is a broad self-query needing all memories."""
        generic_patterns = {
            "myself",
            "about me",
            "who am i",
            "what am i",
            "tell me about me",
            "what do you know",
            "my profile",
            "what have i told",
            "what do i",
        }
        q_lower = query.lower()
        return any(pattern in q_lower for pattern in generic_patterns)

    async def _get_all_user_memories(
        self, user_id: str, limit: int = 20
    ) -> List[RetrievalHit]:
        """Load recent memories directly from repository for broad self-queries."""
        if self.chunk_repo is None or getattr(self.chunk_repo, "client", None) is None:
            return []
        try:
            results = await self.chunk_repo.client.run_query(
                "MATCH (m:Memory {user_id: $user_id}) "
                "RETURN m.memory_id AS id, m.content AS content "
                "ORDER BY m.created_at DESC LIMIT $limit",
                {"user_id": user_id, "limit": limit},
            )
            return [
                RetrievalHit(
                    id=row["id"],
                    content=row["content"],
                    score=0.5,
                    source_type="memory",
                )
                for row in results
            ]
        except Exception as exc:
            logger.debug("Direct memory load failed: {}", exc)
            return []

    async def retrieve_knowledge(self, query: str, top_k: int = 5, user_id: str = "default") -> List[RetrievalHit]:
        """Retrieve chunk-only knowledge results for a query."""
        embedding = self.embedder.embed_text(query)
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_search.search_chunks(embedding, top_k=max(20, self.settings.vector_top_k), user_id=user_id),
            self.bm25_search.search_chunks(query, top_k=max(20, self.settings.bm25_top_k), user_id=user_id),
        )
        combined: dict[str, RetrievalHit] = {}

        def merge(hit: RetrievalHit, origin: str) -> None:
            if hit.id not in combined:
                combined[hit.id] = RetrievalHit(
                    id=hit.id,
                    content=hit.content,
                    score=0.0,
                    source_type=hit.source_type,
                    source_file=hit.source_file,
                    page_number=hit.page_number,
                    vector_score=hit.vector_score,
                    bm25_score=hit.bm25_score,
                    graph_score=hit.graph_score,
                )
            existing = combined[hit.id]
            existing.content = existing.content or hit.content
            existing.source_file = existing.source_file or hit.source_file
            existing.page_number = existing.page_number or hit.page_number
            if origin == "vector":
                existing.vector_score = max(existing.vector_score, hit.vector_score or hit.score)
            elif origin == "bm25":
                existing.bm25_score = max(existing.bm25_score, hit.bm25_score or hit.score)

        for hit in vector_hits:
            merge(hit, "vector")
        for hit in bm25_hits:
            merge(hit, "bm25")

        if not combined:
            logger.warning("Knowledge retrieval returned 0 combined hits for query '{}'", query)
            return []

        max_vector = max((hit.vector_score for hit in combined.values()), default=0.0) or 1.0
        max_bm25 = max((hit.bm25_score for hit in combined.values()), default=0.0) or 1.0
        for hit in combined.values():
            normalized_vector = hit.vector_score / max_vector if hit.vector_score else 0.0
            normalized_bm25 = hit.bm25_score / max_bm25 if hit.bm25_score else 0.0
            hit.score = (0.6 * normalized_vector) + (0.4 * normalized_bm25)

        final_hits = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        min_vector_score = 0.30
        high_quality = [
            hit for hit in final_hits
            if hit.vector_score >= min_vector_score or hit.vector_score == 0.0
        ]
        if len(high_quality) >= 3:
            final_hits = high_quality[: max(10, top_k)]
        else:
            final_hits = final_hits[: max(10, top_k)]
        logger.debug("Knowledge retrieval returned {} chunks for query '{}'", len(final_hits), query)
        for index, hit in enumerate(final_hits, start=1):
            logger.debug(
                "Retrieved chunk {} [{} p{}]: {}",
                index,
                hit.source_file,
                hit.page_number,
                hit.content[:200],
            )
        return final_hits

    async def retrieve_memory(self, query: str, user_id: str, top_k: int = 5) -> List[RetrievalHit]:
        """Retrieve memory-only results for a query and specific user."""
        embedding = self.embedder.embed_text(query)
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_search.search_memories(embedding, user_id, top_k=self.settings.vector_top_k),
            self.bm25_search.search_memories(query, user_id, top_k=self.settings.bm25_top_k),
        )
        ranked = self.hybrid_ranker.rank(vector_hits, bm25_hits, [], top_k)

        if self._is_generic_self_query(query) and self.chunk_repo is not None:
            try:
                recent = await self._get_all_user_memories(user_id, limit=20)
                existing_ids = {hit.id for hit in ranked}
                for hit in recent:
                    if hit.id not in existing_ids:
                        ranked.append(hit)
                        existing_ids.add(hit.id)
            except Exception as exc:
                logger.debug("Recent memory supplement failed: {}", exc)

        return ranked

    async def retrieve_by_filename(self, filename: str, user_id: str = "default", top_k: int = 5) -> List[RetrievalHit]:
        """
        Direct lookup of chunks by source filename.
        Used when user asks about a specific file or paper.
        """
        if self.chunk_repo is None:
            logger.warning("Filename retrieval requested but chunk repository is not configured")
            return []

        results = await self.chunk_repo.get_chunks_by_source(source_file=filename, user_id=user_id, limit=top_k)
        if not results and not filename.lower().endswith((".pdf", ".txt")):
            results = await self.chunk_repo.get_chunks_by_source(
                source_file=f"{filename}.pdf",
                user_id=user_id,
                limit=top_k,
            )
        if not results and not filename.lower().endswith(".txt"):
            results = await self.chunk_repo.get_chunks_by_source(
                source_file=f"{filename}.txt",
                user_id=user_id,
                limit=top_k,
            )

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

    async def retrieve_with_query_expansion(self, query: str, top_k: int = 5, user_id: str = "default") -> List[RetrievalHit]:
        """
        Apply lightweight domain-safe synonym expansion before retrieval.
        """
        normalized = query.strip()
        lowered = normalized.lower()
        synonym_map = {
            "nontowered airport": [
                "nontowered airport",
                "uncontrolled airport",
                "airport without control tower",
                "airport with no control tower",
            ],
            "non towered airport": [
                "non towered airport",
                "nontowered airport",
                "uncontrolled airport",
                "airport without control tower",
            ],
            "towered airport": [
                "towered airport",
                "controlled airport",
                "airport with control tower",
            ],
            "aeronautical charts": [
                "aeronautical charts",
                "aviation charts",
                "flight charts",
            ],
            "ctaf": [
                "ctaf",
                "common traffic advisory frequency",
                "traffic advisory frequency",
            ],
        }
        expanded_terms = [normalized]
        for key, synonyms in synonym_map.items():
            if key in lowered:
                expanded_terms = synonyms[:5]
                break
        expanded_query = " OR ".join(dict.fromkeys(term for term in expanded_terms if term))
        logger.debug("Expanded query: '{}'", expanded_query)
        return await self.retrieve_knowledge(expanded_query, top_k=top_k, user_id=user_id)
