# FILE: src/ingestion/pipeline.py
# CHANGES: Added vector-index readiness warmup and document node creation while keeping the existing ingestion flow intact.

import os
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Set
from uuid import NAMESPACE_URL, uuid5

from loguru import logger
from tqdm import tqdm

from src.config import Settings
from src.graph.chunk_repository import ChunkRepository
from src.graph.entity_repository import EntityRepository
from src.graph.neo4j_client import Neo4jClient
from src.graph_models import EntityNode, IngestionResult, TopicNode
from src.ingestion.chunk_splitter import ChunkSplitter
from src.ingestion.document_parser import DocumentParser
from src.ingestion.embedder import Embedder
from src.ingestion.entity_extractor import EntityExtractor


class IngestionPipeline:
    """Orchestrates full document ingestion into Neo4j."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        chunk_repo: ChunkRepository,
        entity_repo: EntityRepository,
        embedder: Embedder,
        extractor: EntityExtractor,
        splitter: ChunkSplitter,
        parser: DocumentParser,
        settings: Settings,
    ) -> None:
        """Initialize ingestion dependencies."""
        self.neo4j_client = neo4j_client
        self.chunk_repo = chunk_repo
        self.entity_repo = entity_repo
        self.embedder = embedder
        self.extractor = extractor
        self.splitter = splitter
        self.parser = parser
        self.settings = settings
        self.topic_stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "have", "were", "their",
            "there", "which", "about", "into", "after", "before", "than", "then", "them",
            "they", "your", "will", "would", "could", "should", "while", "where", "when",
            "what", "been", "being", "each", "also", "more", "most", "some", "such",
            "only", "other", "over", "under", "just", "like", "much", "many", "very",
        }

    async def _wait_for_index_ready(self) -> None:
        """Poll chunk_vector index until state is ONLINE or timeout."""
        import asyncio

        max_wait_seconds = 180
        poll_interval = 5
        elapsed = 0
        logger.info("Waiting for vector index to reach ONLINE state...")
        while elapsed < max_wait_seconds:
            try:
                result = await self.neo4j_client.run_query(
                    "SHOW INDEXES YIELD name, state "
                    "WHERE name = 'chunk_vector' "
                    "RETURN state"
                )
                state = result[0]["state"] if result else "UNKNOWN"
                if state == "ONLINE":
                    logger.info("Vector index is ONLINE and ready for queries.")
                    return
                logger.info(
                    "Vector index state: {} - waiting {}s ({}/{}s)...",
                    state,
                    poll_interval,
                    elapsed,
                    max_wait_seconds,
                )
            except Exception as exc:
                logger.warning("Index status check failed: {}", exc)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning("Vector index warmup timeout reached. Queries may return incomplete results.")

    async def _warm_vector_index(self) -> None:
        """Fire a dummy vector query to prime the Neo4j query planner."""
        try:
            dummy_embedding = [0.0] * 384
            await self.neo4j_client.run_query(
                "CALL db.index.vector.queryNodes("
                "  'chunk_vector', 1, $embedding"
                ") YIELD node RETURN node.chunk_id LIMIT 1",
                {"embedding": dummy_embedding},
            )
            logger.info("Vector index warmed successfully.")
        except Exception as exc:
            logger.warning("Index warm failed (non-fatal): {}", exc)

    def _extract_topics(self, text: str, limit: int = 3) -> List[TopicNode]:
        """Extract simple keyword topics from a text chunk."""
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", text.lower())
        frequencies = Counter(token for token in tokens if token not in self.topic_stopwords)
        return [
            TopicNode(topic_id=str(uuid5(NAMESPACE_URL, f"topic:{name}")), name=name)
            for name, _ in frequencies.most_common(limit)
        ]

    async def ingest_file(self, file_path: str) -> IngestionResult:
        """Parse, chunk, embed, and persist a single document into Neo4j."""
        filename = os.path.basename(file_path)
        logger.info("Parsing {}...", filename)
        pages = self.parser.parse_file(file_path)
        chunks = self.splitter.split(pages)
        logger.info("Split into {} chunks", len(chunks))
        if not chunks:
            return IngestionResult(source_file=filename, chunk_count=0, entity_count=0, relationship_count=0)
        if await self.chunk_repo.exists_by_source(filename):
            try:
                import fitz
                import uuid as _uuid

                with fitz.open(file_path) as document:
                    doc_meta = document.metadata or {}
                    title = doc_meta.get("title", "") or os.path.splitext(filename)[0]
                    authors = doc_meta.get("author", "") or ""
                    page_count = len(document)

                abstract = chunks[0]["content"][:500] if chunks else ""

                from src.graph_models import DocumentNode

                doc_node = DocumentNode(
                    doc_id=str(_uuid.uuid4()),
                    filename=filename,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    page_count=page_count,
                    chunk_count=len(chunks),
                )
                await self.chunk_repo.create_document_node(doc_node)
                logger.info("Created Document node for {}", filename)
            except Exception as exc:
                logger.warning("Document node creation failed for {}: {}", filename, exc)
            logger.info("Skipping {} - already ingested", filename)
            return IngestionResult(
                source_file=filename,
                chunk_count=0,
                entity_count=0,
                relationship_count=0,
                skipped=True,
            )

        embedded_chunks: List[Dict] = []
        progress = tqdm(
            range(0, len(chunks), self.settings.ingestion_batch_size),
            total=(len(chunks) + self.settings.ingestion_batch_size - 1) // self.settings.ingestion_batch_size,
            desc=f"Ingesting {filename}",
        )
        for start in progress:
            batch = chunks[start : start + self.settings.ingestion_batch_size]
            texts = [chunk["content"] for chunk in batch]
            embeddings = self.embedder.embed_batch(texts)
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
                embedded_chunks.append(chunk)
            if len(embedded_chunks) % 5000 == 0:
                logger.info("Progress: {}/{} chunks", len(embedded_chunks), len(chunks))

        created_chunks = await self.chunk_repo.batch_upsert(embedded_chunks)

        entity_batches = self.extractor.extract_batch([chunk["content"] for chunk in embedded_chunks])
        entity_counter: Counter[str] = Counter()
        canonical_entities: Dict[str, EntityNode] = {}
        for entities in entity_batches:
            for entity in entities:
                entity_counter[entity.entity_id] += 1
                canonical_entities.setdefault(entity.entity_id, entity)
        frequent_entities = {
            entity_id
            for entity_id, count in entity_counter.items()
            if count >= 5
        }
        entities_to_upsert = []
        for entity_id in frequent_entities:
            entity = canonical_entities[entity_id]
            entity.mention_count = entity_counter[entity_id]
            entities_to_upsert.append(entity)
        created_entities = await self.entity_repo.batch_upsert(entities_to_upsert)

        mention_links: List[Dict[str, str]] = []
        cooccurrence_pairs: Set[tuple[str, str]] = set()
        topic_map: Dict[str, TopicNode] = {}
        topic_links: List[Dict[str, str]] = []

        for chunk, entities in zip(embedded_chunks, entity_batches):
            selected_entities = [entity for entity in entities if entity.entity_id in frequent_entities]
            for entity in selected_entities:
                mention_links.append({"chunk_id": chunk["chunk_id"], "entity_id": entity.entity_id})
            selected_ids = sorted({entity.entity_id for entity in selected_entities})
            for entity_a, entity_b in combinations(selected_ids, 2):
                cooccurrence_pairs.add((entity_a, entity_b))
            for topic in self._extract_topics(chunk["content"]):
                topic_map[topic.topic_id] = topic
                topic_links.append({"chunk_id": chunk["chunk_id"], "topic_id": topic.topic_id})

        await self.entity_repo.batch_upsert_topics(list(topic_map.values()))
        mention_count = await self.entity_repo.link_chunk_entities(mention_links)
        topic_count = await self.entity_repo.link_chunk_topics(topic_links)
        cooccurrence_count = await self.entity_repo.create_cooccurrence(
            [{"entity_a": entity_a, "entity_b": entity_b} for entity_a, entity_b in sorted(cooccurrence_pairs)]
        )

        try:
            import fitz
            import uuid as _uuid

            with fitz.open(file_path) as document:
                doc_meta = document.metadata or {}
                title = doc_meta.get("title", "") or os.path.splitext(filename)[0]
                authors = doc_meta.get("author", "") or ""
                page_count = len(document)

            abstract = chunks[0]["content"][:500] if chunks else ""

            from src.graph_models import DocumentNode

            doc_node = DocumentNode(
                doc_id=str(_uuid.uuid4()),
                filename=filename,
                title=title,
                authors=authors,
                abstract=abstract,
                page_count=page_count,
                chunk_count=len(chunks),
            )
            await self.chunk_repo.create_document_node(doc_node)
            logger.info("Created Document node for {}", filename)
        except Exception as exc:
            logger.warning("Document node creation failed for {}: {}", filename, exc)

        relationship_count = mention_count + topic_count + cooccurrence_count
        logger.info("Extracted {} entities. Created {} relationships.", created_entities, relationship_count)
        logger.info("Ingestion complete: {} chunks in Neo4j.", created_chunks)
        return IngestionResult(
            source_file=filename,
            chunk_count=created_chunks,
            entity_count=created_entities,
            relationship_count=relationship_count,
        )

    async def ingest_directory(self, dir_path: str) -> None:
        """Recursively ingest all supported files from a directory one by one."""
        candidates: List[str] = []
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if filename.lower().endswith((".pdf", ".txt", ".md", ".text")):
                    candidates.append(os.path.join(root, filename))
        candidates.sort()
        total_chunks = 0
        total_entities = 0
        total_relationships = 0
        skipped_files = 0
        for path in candidates:
            result = await self.ingest_file(path)
            total_chunks += result.chunk_count
            total_entities += result.entity_count
            total_relationships += result.relationship_count
            skipped_files += int(result.skipped)

        await self._wait_for_index_ready()
        await self._warm_vector_index()

        logger.info(
            "Directory ingestion summary: files={}, skipped={}, chunks={}, entities={}, relationships={}",
            len(candidates),
            skipped_files,
            total_chunks,
            total_entities,
            total_relationships,
        )
