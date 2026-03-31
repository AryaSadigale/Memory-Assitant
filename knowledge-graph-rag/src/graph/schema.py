# FILE: src/graph/schema.py
# CHANGES: Added Document schema comments plus new document constraints and indexes.

from loguru import logger

from src.graph.neo4j_client import Neo4jClient

# (:Document {
#     doc_id: string,       # uuid
#     filename: string,     # original filename
#     title: string,        # from PDF metadata or filename
#     authors: string,      # from PDF metadata
#     abstract: string,     # first 500 chars of content
#     page_count: int,
#     chunk_count: int,
#     ingested_at: datetime
# })
# (:Document)-[:CONTAINS]->(:Chunk)


async def initialize_schema(client: Neo4jClient) -> None:
    """Create all constraints and indexes. Safe to run multiple times."""
    statements = [
        "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT entity_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE CONSTRAINT memory_memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE",
        "CREATE CONSTRAINT user_user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE VECTOR INDEX chunk_vector IF NOT EXISTS FOR (c:Chunk) ON c.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
        "CREATE VECTOR INDEX memory_vector IF NOT EXISTS FOR (m:Memory) ON m.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
        "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content]",
        "CREATE FULLTEXT INDEX memory_fulltext IF NOT EXISTS FOR (m:Memory) ON EACH [m.content]",
        "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source_file)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX memory_user IF NOT EXISTS FOR (m:Memory) ON (m.user_id)",
        "CREATE INDEX memory_tier IF NOT EXISTS FOR (m:Memory) ON (m.tier)",
        "CREATE FULLTEXT INDEX document_fulltext IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.filename, d.abstract]",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename)",
    ]
    for statement in statements:
        await client.run_write(statement)
    logger.info("Schema initialized successfully")
