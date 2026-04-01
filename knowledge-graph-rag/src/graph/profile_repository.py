# FILE: src/graph/profile_repository.py
# CHANGES: Added Neo4j repository methods for Profile node access and live per-user stats.

from typing import Optional

from loguru import logger

from src.graph.neo4j_client import Neo4jClient


class ProfileRepository:
    """Handles all Neo4j operations for Profile nodes."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize with Neo4j client."""
        self.client = client

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """Fetch profile by user_id. Returns None if not found."""
        try:
            result = await self.client.run_query(
                "MATCH (p:Profile {user_id: $user_id}) "
                "RETURN p.user_id AS user_id, "
                "       p.username AS username, "
                "       p.display_name AS display_name, "
                "       p.chunk_count AS chunk_count, "
                "       p.document_count AS document_count, "
                "       p.memory_count AS memory_count, "
                "       p.created_at AS created_at",
                {"user_id": user_id},
            )
            return result[0] if result else None
        except Exception as exc:
            logger.error("get_profile failed: {}", exc)
            return None

    async def get_user_stats(self, user_id: str) -> dict:
        """
        Return storage statistics for a user.
        Counts are computed live from actual graph data.
        """
        try:
            chunk_result = await self.client.run_query(
                "MATCH (c:Chunk {user_id: $user_id}) "
                "RETURN count(c) AS chunk_count",
                {"user_id": user_id},
            )
            doc_result = await self.client.run_query(
                "MATCH (d:Document {user_id: $user_id}) "
                "RETURN count(d) AS doc_count",
                {"user_id": user_id},
            )
            mem_result = await self.client.run_query(
                "MATCH (m:Memory {user_id: $user_id}) "
                "RETURN count(m) AS mem_count",
                {"user_id": user_id},
            )
            return {
                "chunks": chunk_result[0]["chunk_count"] if chunk_result else 0,
                "documents": doc_result[0]["doc_count"] if doc_result else 0,
                "memories": mem_result[0]["mem_count"] if mem_result else 0,
            }
        except Exception as exc:
            logger.error("get_user_stats failed: {}", exc)
            return {"chunks": 0, "documents": 0, "memories": 0}
