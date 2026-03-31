# FILE: src/retrieval/graph_traversal.py
# PURPOSE: Expand seed chunk retrieval results through graph relationships.

from typing import List

from src.graph.neo4j_client import Neo4jClient
from src.graph_models import RetrievalHit


class GraphTraversal:
    """Expand retrieval results by traversing graph relationships."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the graph traversal service."""
        self.client = client

    async def expand_from_chunks(self, chunk_ids: List[str], depth: int = 1) -> List[RetrievalHit]:
        """Find neighboring chunks connected through shared entities."""
        if not chunk_ids:
            return []
        query = """
        MATCH (seed:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbor:Chunk)
        WHERE seed.chunk_id IN $chunk_ids
          AND NOT neighbor.chunk_id IN $chunk_ids
        RETURN DISTINCT neighbor.chunk_id AS id,
               neighbor.content AS content,
               neighbor.source_file AS source_file,
               neighbor.page_number AS page_number,
               count(e) AS shared_entities
        ORDER BY shared_entities DESC
        LIMIT 5
        """
        rows = await self.client.run_query(query, {"chunk_ids": chunk_ids, "depth": depth})
        return [
            RetrievalHit(
                id=row["id"],
                content=row["content"],
                score=0.5,
                source_type="chunk",
                source_file=row.get("source_file", ""),
                page_number=int(row.get("page_number") or 0),
                graph_score=0.5,
            )
            for row in rows
        ]
