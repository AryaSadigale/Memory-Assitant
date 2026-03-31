# FILE: src/graph/memory_repository.py
# PURPOSE: Persist and query user memory nodes in Neo4j.

from datetime import datetime
from typing import Any, List

from src.graph.neo4j_client import Neo4jClient
from src.graph_models import MemoryNode


def _to_datetime(value: Any) -> datetime:
    """Convert a Neo4j datetime-like value into a native datetime."""
    if hasattr(value, "to_native"):
        return value.to_native()
    if isinstance(value, datetime):
        return value
    return datetime.utcnow()


class MemoryRepository:
    """Repository for memory graph operations."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the repository with a Neo4j client."""
        self.client = client

    async def create(self, memory: MemoryNode) -> MemoryNode:
        """Create a memory node and link it to its user node."""
        query = """
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET
            u.name = coalesce(u.name, 'User'),
            u.created_at = datetime()
        CREATE (m:Memory {
            memory_id: $memory_id,
            user_id: $user_id,
            content: $content,
            embedding: $embedding,
            tier: $tier,
            created_at: datetime(),
            last_accessed: datetime(),
            access_count: 0
        })
        CREATE (u)-[:HAS_MEMORY]->(m)
        """
        params = {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": memory.content,
            "embedding": memory.embedding,
            "tier": memory.tier,
        }
        await self.client.run_write(query, params)
        return memory

    async def list_recent(self, user_id: str, limit: int = 30) -> List[MemoryNode]:
        """List recent memories for a user."""
        query = """
        MATCH (m:Memory {user_id: $user_id})
        RETURN m
        ORDER BY m.created_at DESC
        LIMIT $limit
        """
        rows = await self.client.run_query(query, {"user_id": user_id, "limit": limit})
        memories: List[MemoryNode] = []
        for row in rows:
            node = row["m"]
            memories.append(
                MemoryNode(
                    memory_id=node["memory_id"],
                    user_id=node["user_id"],
                    content=node["content"],
                    tier=node.get("tier", "STM"),
                    created_at=_to_datetime(node.get("created_at")),
                    last_accessed=_to_datetime(node.get("last_accessed")),
                    access_count=int(node.get("access_count", 0)),
                    embedding=node.get("embedding"),
                )
            )
        return memories

    async def delete(self, memory_id: str) -> None:
        """Delete a memory node by its id."""
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        DETACH DELETE m
        """
        await self.client.run_write(query, {"memory_id": memory_id})

    async def count(self, user_id: str) -> int:
        """Return the number of memories stored for a user."""
        query = """
        MATCH (m:Memory {user_id: $user_id})
        RETURN count(m) AS count
        """
        rows = await self.client.run_query(query, {"user_id": user_id})
        return int(rows[0]["count"]) if rows else 0

    async def find_by_text(self, user_id: str, text: str, limit: int = 10) -> List[MemoryNode]:
        """Find memories whose content contains the given text."""
        query = """
        MATCH (m:Memory {user_id: $user_id})
        WHERE toLower(m.content) CONTAINS toLower($text)
        RETURN m
        ORDER BY m.created_at DESC
        LIMIT $limit
        """
        rows = await self.client.run_query(query, {"user_id": user_id, "text": text, "limit": limit})
        matches: List[MemoryNode] = []
        for row in rows:
            node = row["m"]
            matches.append(
                MemoryNode(
                    memory_id=node["memory_id"],
                    user_id=node["user_id"],
                    content=node["content"],
                    tier=node.get("tier", "STM"),
                    created_at=_to_datetime(node.get("created_at")),
                    last_accessed=_to_datetime(node.get("last_accessed")),
                    access_count=int(node.get("access_count", 0)),
                    embedding=node.get("embedding"),
                )
            )
        return matches

    async def touch_many(self, memory_ids: List[str]) -> None:
        """Update access metadata for the provided memory ids."""
        if not memory_ids:
            return
        query = """
        UNWIND $batch AS row
        MATCH (m:Memory {memory_id: row.memory_id})
        SET m.last_accessed = datetime(),
            m.access_count = coalesce(m.access_count, 0) + 1
        """
        await self.client.run_batch_write(query, [{"memory_id": memory_id} for memory_id in memory_ids])
