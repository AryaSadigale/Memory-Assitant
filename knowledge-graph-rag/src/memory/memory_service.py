# FILE: src/memory/memory_service.py
# PURPOSE: Manage extraction, storage, and retrieval of personal user memories.

from typing import List
from uuid import uuid4

from src.graph.memory_repository import MemoryRepository
from src.graph_models import MemoryNode
from src.ingestion.embedder import Embedder
from src.llm.fact_extractor import FactExtractor


class MemoryService:
    """Manages personal memory lifecycle."""

    def __init__(self, memory_repo: MemoryRepository, embedder: Embedder, fact_extractor: FactExtractor) -> None:
        """Initialize memory dependencies."""
        self.memory_repo = memory_repo
        self.embedder = embedder
        self.fact_extractor = fact_extractor

    async def process_and_store(self, message: str, user_id: str) -> List[MemoryNode]:
        """Extract personal facts from a message and persist them as memory nodes."""
        facts = await self.fact_extractor.extract(message)
        if not facts:
            return []
        created: List[MemoryNode] = []
        for fact in facts:
            embedding = self.embedder.embed_text(fact)
            node = MemoryNode(
                memory_id=str(uuid4()),
                user_id=user_id,
                content=fact,
                embedding=embedding,
            )
            await self.memory_repo.create(node)
            created.append(node)
        print(f"[Stored {len(created)} memories]")
        return created

    async def load_for_session(self, user_id: str, limit: int = 30) -> List[MemoryNode]:
        """Load recent memories for a returning user."""
        return await self.memory_repo.list_recent(user_id, limit=limit)
