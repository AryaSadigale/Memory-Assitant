# FILE: src/llm/context_assembler.py
# PURPOSE: Assemble retrieval hits into bounded prompts for the answering model.

from typing import List

from src.graph_models import ContextPacket, RetrievalHit


class ContextAssembler:
    """Build structured LLM prompts from retrieval hits."""

    KNOWLEDGE_SYSTEM = """You are a precise knowledge assistant.
Answer questions using ONLY the provided source passages.
Cite sources as [SOURCE 1], [SOURCE 2] etc.
If the passages don't contain the answer, say exactly:
"I don't have information on this in my knowledge base."
Never answer from general knowledge."""

    MEMORY_SYSTEM = """You are a personal memory assistant.
Answer questions about the user using ONLY their stored memories.
Use only facts explicitly present in the provided memories.
Do not add opinions, praise, speculation, or unrelated information.
Do not mention your own preferences or personal thoughts.
If the user asks for a general summary about themselves, give a short plain summary of the stored facts only.
If you don't have the information, say exactly:
"I don't have that stored yet. Want to tell me?"
Never guess or infer."""

    @staticmethod
    def _trim_hits(hits: List[RetrievalHit], max_words: int) -> List[RetrievalHit]:
        """Trim a hit list to stay within an approximate word budget."""
        selected: List[RetrievalHit] = []
        total_words = 0
        for hit in hits:
            words = len(hit.content.split())
            if selected and total_words + words > max_words:
                break
            selected.append(hit)
            total_words += words
        return selected

    def build_knowledge_context(self, query: str, hits: List[RetrievalHit]) -> ContextPacket:
        """Build a knowledge-answering prompt packet from chunk hits."""
        trimmed_hits = self._trim_hits(hits, max_words=1100)
        formatted_sources = []
        for index, hit in enumerate(trimmed_hits, start=1):
            if hit.page_number:
                location = f"(from: {hit.source_file}, page {hit.page_number})"
            else:
                location = f"(from: {hit.source_file})"
            formatted_sources.append(f"[SOURCE {index}] {location}\n{hit.content}")
        sources = "\n\n".join(formatted_sources)
        return ContextPacket(
            query=query,
            chunk_hits=trimmed_hits,
            memory_hits=[],
            system_prompt=self.KNOWLEDGE_SYSTEM,
            user_prompt=f"Sources:\n{sources}\n\nQuestion: {query}",
            has_chunks=bool(trimmed_hits),
            has_memories=False,
        )

    def build_memory_context(self, query: str, hits: List[RetrievalHit]) -> ContextPacket:
        """Build a memory-answering prompt packet from memory hits."""
        trimmed_hits = self._trim_hits(hits, max_words=450)
        formatted_memories = [f"[MEMORY {index}] {hit.content}" for index, hit in enumerate(trimmed_hits, start=1)]
        memories = "\n".join(formatted_memories)
        guidance = (
            "Answer in 1-3 short sentences. "
            "Use only the memory facts below. "
            "If the answer is missing, reply exactly with the fallback sentence."
        )
        return ContextPacket(
            query=query,
            chunk_hits=[],
            memory_hits=trimmed_hits,
            system_prompt=self.MEMORY_SYSTEM,
            user_prompt=f"Instructions: {guidance}\n\nMemories:\n{memories}\n\nQuestion: {query}",
            has_chunks=False,
            has_memories=bool(trimmed_hits),
        )
