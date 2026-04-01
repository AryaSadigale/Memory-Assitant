# FILE: src/retrieval/vector_search.py
# CHANGES: Added user-scoped vector retrieval while preserving diagnostics and shared-pool visibility.

from typing import List

from loguru import logger

from src.graph.neo4j_client import Neo4jClient
from src.graph_models import RetrievalHit


class VectorSearch:
    """Semantic similarity search over Neo4j vector indexes."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the vector search service."""
        self.client = client

    async def search_chunks(self, embedding: List[float], top_k: int = 10, user_id: str = "default") -> List[RetrievalHit]:
        """Search chunk_vector index for semantically similar chunks."""
        if not embedding:
            logger.error("Vector search called with empty embedding")
            return []

        if len(embedding) != 384:
            logger.error(
                "Embedding dimension mismatch: got {}, expected 384. Skipping vector search.",
                len(embedding),
            )
            return []

        try:
            results = await self.client.run_query(
                "CALL db.index.vector.queryNodes("
                "  'chunk_vector', $top_k, $embedding"
                ") YIELD node, score "
                "WHERE node.user_id = $user_id "
                "   OR node.user_id IS NULL "
                "RETURN node.chunk_id AS chunk_id, "
                "       node.content AS content, "
                "       node.source_file AS source_file, "
                "       node.page_number AS page_number, "
                "       score "
                "ORDER BY score DESC",
                {"top_k": top_k, "embedding": embedding, "user_id": user_id},
            )

            logger.debug("Vector search: {} hits returned", len(results))

            if not results:
                logger.warning(
                    "Vector search returned zero results. Possible causes: index still POPULATING, "
                    "embedding dimension mismatch, or no similar content."
                )

            return [
                RetrievalHit(
                    id=result["chunk_id"],
                    content=result["content"],
                    score=float(result["score"]),
                    source_type="chunk",
                    source_file=result["source_file"] or "",
                    page_number=result["page_number"] or 0,
                    vector_score=float(result["score"]),
                )
                for result in results
            ]

        except Exception as exc:
            logger.error("Vector search failed: {}", exc)
            return []

    async def search_memories(self, embedding: List[float], user_id: str, top_k: int = 10) -> List[RetrievalHit]:
        """Search memory nodes by vector similarity for a specific user."""
        if not embedding:
            logger.error("Memory vector search called with empty embedding")
            return []

        try:
            rows = await self.client.run_query(
                """
                CALL db.index.vector.queryNodes('memory_vector', $top_k, $embedding)
                YIELD node, score
                WHERE node.user_id = $user_id
                RETURN node.memory_id AS id,
                       node.content AS content,
                       score AS score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {"embedding": embedding, "user_id": user_id, "top_k": top_k},
            )
            logger.debug("Memory vector search: {} hits returned", len(rows))
            if not rows:
                logger.warning("Memory vector search returned zero results for user {}", user_id)
            return [
                RetrievalHit(
                    id=row["id"],
                    content=row["content"],
                    score=float(row["score"]),
                    source_type="memory",
                    vector_score=float(row["score"]),
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error("Memory vector search failed: {}", exc)
            return []
