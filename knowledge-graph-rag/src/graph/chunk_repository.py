# FILE: src/graph/chunk_repository.py
# CHANGES: Added document metadata helpers and source-based chunk lookup without altering existing chunk CRUD.

from typing import TYPE_CHECKING, Dict, List

from loguru import logger

from src.graph.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from src.graph_models import DocumentNode


class ChunkRepository:
    """Repository for chunk graph operations."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the repository with a Neo4j client."""
        self.client = client

    async def batch_upsert(self, chunks: List[Dict]) -> int:
        """Upsert chunk nodes in batches of 100 and return the processed count."""
        if not chunks:
            return 0
        query = """
        UNWIND $batch AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        ON CREATE SET
            c.source_file = row.source_file,
            c.page_number = row.page_number,
            c.content = row.content,
            c.embedding = row.embedding,
            c.token_count = row.token_count,
            c.ingested_at = datetime()
        ON MATCH SET
            c.source_file = row.source_file,
            c.page_number = row.page_number,
            c.content = row.content,
            c.embedding = row.embedding,
            c.token_count = row.token_count
        """
        total = 0
        for index in range(0, len(chunks), 100):
            batch = chunks[index : index + 100]
            await self.client.run_batch_write(query, batch)
            total += len(batch)
        return total

    async def exists_by_source(self, source_file: str) -> bool:
        """Return True if any chunks for the given source file already exist."""
        query = """
        MATCH (c:Chunk {source_file: $source_file})
        RETURN count(c) > 0 AS exists
        """
        rows = await self.client.run_query(query, {"source_file": source_file})
        return bool(rows and rows[0]["exists"])

    async def count(self) -> int:
        """Return the total number of chunk nodes."""
        rows = await self.client.run_query("MATCH (c:Chunk) RETURN count(c) AS count")
        return int(rows[0]["count"]) if rows else 0

    async def create_document_node(self, doc: "DocumentNode") -> None:
        """Create or update a Document node and link it to its chunks."""
        try:
            await self.client.run_write(
                "MERGE (d:Document {filename: $filename}) "
                "ON CREATE SET d.doc_id = $doc_id "
                "SET d.filename = $filename, "
                "    d.title = $title, "
                "    d.authors = $authors, "
                "    d.abstract = $abstract, "
                "    d.page_count = $page_count, "
                "    d.chunk_count = $chunk_count, "
                "    d.ingested_at = datetime() "
                "WITH d "
                "MATCH (c:Chunk {source_file: $filename}) "
                "MERGE (d)-[:CONTAINS]->(c)",
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "title": doc.title,
                    "authors": doc.authors,
                    "abstract": doc.abstract,
                    "page_count": doc.page_count,
                    "chunk_count": doc.chunk_count,
                },
            )
        except Exception as exc:
            logger.warning("Document node creation failed for {}: {}", doc.filename, exc)

    async def get_chunks_by_source(self, source_file: str, limit: int = 5) -> List[dict]:
        """Retrieve chunks belonging to a specific source file."""
        try:
            return await self.client.run_query(
                "MATCH (c:Chunk {source_file: $source_file}) "
                "RETURN c.chunk_id AS chunk_id, "
                "       c.content AS content, "
                "       c.source_file AS source_file, "
                "       c.page_number AS page_number "
                "ORDER BY c.page_number ASC "
                "LIMIT $limit",
                {"source_file": source_file, "limit": limit},
            )
        except Exception as exc:
            logger.warning("Chunk lookup by source failed for {}: {}", source_file, exc)
            return []

    async def list_all_documents(self) -> List[dict]:
        """List all Document nodes with metadata."""
        try:
            return await self.client.run_query(
                "MATCH (d:Document) "
                "RETURN d.filename AS filename, "
                "       d.title AS title, "
                "       d.chunk_count AS chunk_count, "
                "       d.page_count AS page_count, "
                "       d.ingested_at AS ingested_at "
                "ORDER BY d.ingested_at DESC"
            )
        except Exception as exc:
            logger.warning("Document listing failed: {}", exc)
            return []
