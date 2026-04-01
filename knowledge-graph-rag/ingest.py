# FILE: ingest.py
# CHANGES: Added profile-aware ingestion so files are stamped into the correct private user namespace.

import argparse
import asyncio
import os

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
from src.memory.profile_manager import ProfileManager


async def main() -> None:
    """Initialize dependencies and ingest a file or directory from the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--user",
        help="Username to ingest as (default: current session user)",
        default=None,
    )
    args = parser.parse_args()

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

        profile_manager = ProfileManager(neo4j_client)

        if args.user:
            result = await neo4j_client.run_query(
                "MATCH (p:Profile {username: $username}) "
                "RETURN p.user_id AS user_id",
                {"username": args.user.lower()},
            )
            if not result:
                print(f"User '{args.user}' not found. Run main.py first to create a profile.")
                return
            user_id = result[0]["user_id"]
            print(f"Ingesting as user: {args.user}")
        else:
            profile = await profile_manager.get_or_create_profile()
            user_id = profile.user_id

        if os.path.isdir(args.path):
            await pipeline.ingest_directory(args.path, user_id=user_id)
        elif os.path.isfile(args.path):
            await pipeline.ingest_file(args.path, user_id=user_id)
        else:
            raise SystemExit(f"Path not found: {args.path}")
        print(f"Total chunks: {await chunk_repo.count()}")
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())
