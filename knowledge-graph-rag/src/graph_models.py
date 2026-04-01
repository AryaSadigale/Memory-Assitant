# FILE: src/graph_models.py
# CHANGES: Added ProfileNode and user-scoped DocumentNode while preserving existing graph and retrieval dataclasses.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ChunkNode:
    """Represents a knowledge chunk stored in Neo4j."""

    chunk_id: str
    source_file: str
    page_number: int
    content: str
    token_count: int
    ingested_at: datetime
    embedding: Optional[List[float]] = None


@dataclass
class EntityNode:
    """Represents an extracted named entity."""

    entity_id: str
    name: str
    label: str
    mention_count: int = 1


@dataclass
class TopicNode:
    """Represents a lightweight keyword topic."""

    topic_id: str
    name: str


@dataclass
class ProfileNode:
    """Represents a registered user profile in the knowledge graph."""

    user_id: str
    username: str
    display_name: str
    created_at: str
    chunk_count: int = 0
    document_count: int = 0
    memory_count: int = 0


@dataclass
class DocumentNode:
    """Represents a top-level ingested document in the knowledge graph."""

    doc_id: str
    filename: str
    title: str
    authors: str
    abstract: str
    page_count: int
    chunk_count: int = 0
    user_id: str = "default"
    ingested_at: Optional[datetime] = None


@dataclass
class MemoryNode:
    """Represents a stored personal memory."""

    memory_id: str
    user_id: str
    content: str
    tier: str = "STM"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalHit:
    """Represents a retrieval candidate from chunks or memories."""

    id: str
    content: str
    score: float
    source_type: str
    source_file: str = ""
    page_number: int = 0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0


@dataclass
class ContextPacket:
    """Represents the assembled prompt payload for an LLM completion."""

    query: str
    chunk_hits: List[RetrievalHit]
    memory_hits: List[RetrievalHit]
    system_prompt: str
    user_prompt: str
    has_chunks: bool
    has_memories: bool


@dataclass
class IngestionResult:
    """Summarizes the outcome of a document ingestion run."""

    source_file: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    skipped: bool = False
