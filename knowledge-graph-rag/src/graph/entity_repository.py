# FILE: src/graph/entity_repository.py
# PURPOSE: Persist entities, topics, and graph relationships derived from ingested chunks.

from typing import Dict, List

from src.graph.neo4j_client import Neo4jClient
from src.graph_models import EntityNode, TopicNode


class EntityRepository:
    """Repository for entity and relationship graph operations."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the repository with a Neo4j client."""
        self.client = client

    async def batch_upsert(self, entities: List[EntityNode]) -> int:
        """Upsert entity nodes in batches of 200 and return the processed count."""
        if not entities:
            return 0
        query = """
        UNWIND $batch AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        ON CREATE SET
            e.name = row.name,
            e.label = row.label,
            e.mention_count = row.mention_count
        ON MATCH SET
            e.name = row.name,
            e.label = row.label,
            e.mention_count = coalesce(e.mention_count, 0) + row.mention_count
        """
        total = 0
        batch_payload = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "label": entity.label,
                "mention_count": entity.mention_count,
            }
            for entity in entities
        ]
        for index in range(0, len(batch_payload), 200):
            batch = batch_payload[index : index + 200]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total

    async def batch_upsert_topics(self, topics: List[TopicNode]) -> int:
        """Upsert topic nodes in batches of 200 and return the processed count."""
        if not topics:
            return 0
        query = """
        UNWIND $batch AS row
        MERGE (t:Topic {topic_id: row.topic_id})
        ON CREATE SET
            t.name = row.name
        ON MATCH SET
            t.name = row.name
        """
        total = 0
        batch_payload = [{"topic_id": topic.topic_id, "name": topic.name} for topic in topics]
        for index in range(0, len(batch_payload), 200):
            batch = batch_payload[index : index + 200]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total

    async def link_chunk_entities(self, links: List[Dict[str, str]]) -> int:
        """Create chunk-to-entity MENTIONS relationships in batches."""
        if not links:
            return 0
        query = """
        UNWIND $batch AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MATCH (e:Entity {entity_id: row.entity_id})
        MERGE (c)-[:MENTIONS]->(e)
        """
        total = 0
        for index in range(0, len(links), 500):
            batch = links[index : index + 500]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total

    async def link_chunk_topics(self, links: List[Dict[str, str]]) -> int:
        """Create chunk-to-topic ABOUT_TOPIC relationships in batches."""
        if not links:
            return 0
        query = """
        UNWIND $batch AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MATCH (t:Topic {topic_id: row.topic_id})
        MERGE (c)-[:ABOUT_TOPIC]->(t)
        """
        total = 0
        for index in range(0, len(links), 500):
            batch = links[index : index + 500]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total

    async def create_cooccurrence(self, pairs: List[Dict[str, str]]) -> int:
        """Create entity co-occurrence relationships in batches."""
        if not pairs:
            return 0
        query = """
        UNWIND $batch AS row
        MATCH (a:Entity {entity_id: row.entity_a})
        MATCH (b:Entity {entity_id: row.entity_b})
        MERGE (a)-[r:CO_OCCURS_WITH]->(b)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = coalesce(r.weight, 0) + 1
        """
        total = 0
        for index in range(0, len(pairs), 500):
            batch = pairs[index : index + 500]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total
