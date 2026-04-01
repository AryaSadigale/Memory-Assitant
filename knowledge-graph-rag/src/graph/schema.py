# FILE: src/graph/schema.py
# CHANGES: Added index verification plus memory fulltext health-check and rebuild helpers.

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
        "CREATE CONSTRAINT profile_user_id_unique IF NOT EXISTS FOR (p:Profile) REQUIRE p.user_id IS UNIQUE",
        "CREATE INDEX profile_username IF NOT EXISTS FOR (p:Profile) ON (p.username)",
        "CREATE INDEX chunk_user_id IF NOT EXISTS FOR (c:Chunk) ON (c.user_id)",
        "CREATE INDEX document_user_id IF NOT EXISTS FOR (d:Document) ON (d.user_id)",
        "CREATE RANGE INDEX chunk_user_source IF NOT EXISTS FOR (c:Chunk) ON (c.user_id, c.source_file)",
    ]
    for statement in statements:
        await client.run_write(statement)
    logger.info("Schema initialized successfully")


async def verify_indexes(client: Neo4jClient) -> None:
    """
    Log the state of all critical indexes.
    Called at startup to catch misconfigured indexes early.
    """
    try:
        result = await client.run_query(
            "SHOW INDEXES YIELD name, state, type "
            "WHERE name IN ["
            "  'chunk_vector', 'memory_vector', "
            "  'chunk_fulltext', 'memory_fulltext', "
            "  'document_fulltext'"
            "] "
            "RETURN name, state, type "
            "ORDER BY name"
        )
        for row in result:
            status = "OK" if row["state"] == "ONLINE" else "PROBLEM"
            logger.info(
                "Index {} [{}] - {} ({})",
                row["name"],
                row["type"],
                row["state"],
                status,
            )
        offline = [row["name"] for row in result if row["state"] != "ONLINE"]
        if offline:
            logger.warning(
                "These indexes are NOT ONLINE: {}. Queries may return zero results.",
                offline,
            )
    except Exception as exc:
        logger.warning("Index verification failed: {}", exc)


async def rebuild_memory_fulltext_index(client: Neo4jClient) -> None:
    """
    Drop and recreate the memory fulltext index to force reindexing
    of all existing Memory nodes. Called once at startup if the index
    returns zero results despite nodes existing.
    """
    import asyncio

    logger.info("Rebuilding memory_fulltext index...")
    try:
        await client.run_write(
            "DROP INDEX memory_fulltext IF EXISTS"
        )
        logger.info("Dropped old memory_fulltext index")
    except Exception as exc:
        logger.warning("Drop memory_fulltext failed (may not exist): {}", exc)

    await asyncio.sleep(2)

    try:
        await client.run_write(
            "CREATE FULLTEXT INDEX memory_fulltext IF NOT EXISTS "
            "FOR (m:Memory) ON EACH [m.content]"
        )
        logger.info("Recreated memory_fulltext index")
    except Exception as exc:
        logger.error("Failed to recreate memory_fulltext: {}", exc)
        return

    max_wait = 60
    elapsed = 0
    poll = 3
    while elapsed < max_wait:
        await asyncio.sleep(poll)
        elapsed += poll
        try:
            result = await client.run_query(
                "SHOW INDEXES YIELD name, state "
                "WHERE name = 'memory_fulltext' "
                "RETURN state"
            )
            state = result[0]["state"] if result else "UNKNOWN"
            if state == "ONLINE":
                logger.info("memory_fulltext index is ONLINE")
                return
            logger.info(
                "memory_fulltext state: {} - waiting {}s...",
                state, poll
            )
        except Exception as exc:
            logger.warning("Index poll failed: {}", exc)

    logger.warning("memory_fulltext rebuild timeout")


async def check_memory_index_health(
    client: Neo4jClient, user_id: str
) -> bool:
    """
    Test whether memory_fulltext index actually returns results.
    Returns True if healthy, False if needs rebuilding.
    """
    try:
        count_result = await client.run_query(
            "MATCH (m:Memory {user_id: $user_id}) "
            "RETURN count(m) AS total",
            {"user_id": user_id}
        )
        total = count_result[0]["total"] if count_result else 0
        if total == 0:
            return True

        test_result = await client.run_query(
            "CALL db.index.fulltext.queryNodes("
            "  'memory_fulltext', 'the OR a OR is OR in OR of OR and'"
            ") YIELD node "
            "WHERE node.user_id = $user_id "
            "RETURN count(node) AS hits",
            {"user_id": user_id}
        )
        hits = test_result[0]["hits"] if test_result else 0
        if hits == 0 and total > 0:
            logger.warning(
                "memory_fulltext index has {} Memory nodes "
                "but returned 0 hits on test query - needs rebuild",
                total
            )
            return False
        return True
    except Exception as exc:
        logger.warning("Memory index health check failed: {}", exc)
        return True
