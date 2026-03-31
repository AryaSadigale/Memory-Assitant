# FILE: ingest.py
# PURPOSE: Run standalone document ingestion without starting the interactive chat loop.

import asyncio
import os
import sys

from src.config import Settings
from src.graph.chunk_repository import ChunkRepository
from src.graph.entity_repository import EntityRepository
from src.graph.neo4j_client import Neo4jClient
from src.graph.schema import initialize_schema
from src.ingestion.chunk_splitter import ChunkSplitter
from src.ingestion.document_parser import DocumentParser
from src.ingestion.embedder import Embedder
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.pipeline import IngestionPipeline


async def main() -> None:
    """Initialize dependencies and ingest a file or directory from the CLI."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python ingest.py /app/data/myfile.pdf or python ingest.py /app/data/")
    path = sys.argv[1]
    settings = Settings()
    neo4j_client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    await neo4j_client.connect()
    try:
        await initialize_schema(neo4j_client)
        embedder = Embedder(settings.embedding_model, settings.embedding_device)
        chunk_repo = ChunkRepository(neo4j_client)
        entity_repo = EntityRepository(neo4j_client)
        pipeline = IngestionPipeline(
            neo4j_client=neo4j_client,
            chunk_repo=chunk_repo,
            entity_repo=entity_repo,
            embedder=embedder,
            extractor=EntityExtractor(),
            splitter=ChunkSplitter(settings.chunk_size, settings.chunk_overlap),
            parser=DocumentParser(),
            settings=settings,
        )
        if os.path.isdir(path):
            await pipeline.ingest_directory(path)
        elif os.path.isfile(path):
            await pipeline.ingest_file(path)
        else:
            raise SystemExit(f"Path not found: {path}")
        print(f"Total chunks: {await chunk_repo.count()}")
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())
