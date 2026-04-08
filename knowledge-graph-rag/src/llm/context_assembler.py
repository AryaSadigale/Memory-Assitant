# FILE: src/llm/context_assembler.py
# CHANGES: Added cover/title page detection to the structural chunk filter.

from collections import OrderedDict
from typing import List

from loguru import logger

from src.graph_models import ContextPacket, RetrievalHit


class ContextAssembler:
    """Build structured LLM prompts from retrieval hits."""

    _LEGACY_KNOWLEDGE_SYSTEM = """You are an expert assistant answering STRICTLY from provided context.

RULES:

* Use ONLY the given context
* NEVER say 'no information' if context exists
* Extract FULL explanation, not summary
* Minimum answer length: 8–12 lines
* If definition exists -> explain it fully
* Combine multiple context sections if needed

If context contains:
'Nontowered Airport'

You MUST explain:

* definition
* characteristics
* communication method
* pilot behavior

DO NOT say:
'not enough information'
Instead extract what is present.

Cite sources as [SOURCE 1], [SOURCE 2] etc."""

    _LEGACY_MEMORY_SYSTEM = """You are a personal memory assistant.
Answer questions about the user using ONLY their stored memories.
Use only facts explicitly present in the provided memories.
Do not add opinions, praise, speculation, or unrelated information.
Do not mention your own preferences or personal thoughts.
If the user asks for a general summary about themselves, give a short plain summary of the stored facts only.
If you don't have the information, say exactly:
"I don't have that stored yet. Want to tell me?"
Never guess or infer."""

    KNOWLEDGE_SYSTEM = """You are a precise knowledge assistant.
Answer using ONLY the source passages provided.
Cite as [SOURCE 1], [SOURCE 2] when drawing from them.
Write in plain conversational paragraphs.
Do NOT use markdown headers (##), numbered sections, or bullet outlines.
If passages do not contain the answer, say:
"I don't have information on this in my knowledge base."
Never use general knowledge outside the provided sources."""

    MEMORY_SYSTEM = """You are a personal memory assistant.
Answer using ONLY the stored memories listed below.
Be direct and conversational. No markdown headers or sections.
If the information is not stored, say:
"I don't have that stored yet. Want to tell me?"
Never guess or infer beyond what is in the memories."""

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

    def _is_toc_or_index_chunk(self, content: str) -> bool:
        """
        Return True if chunk looks like a table of contents,
        index page, or other non-explanatory structural content.
        These chunks match many queries but answer none.
        """
        if not content or len(content.strip()) < 50:
            return True

        stripped = content.strip()
        word_count = len(stripped.split())

        cover_signals = [
            "department of transportation",
            "federal aviation administration",
            "pilot's handbook",
            "handbook of aeronautical",
            "u.s. department",
            "flight standards service",
            "advisory circular",
        ]
        if word_count < 60:
            content_lower = stripped.lower()
            if any(signal in content_lower for signal in cover_signals):
                return True

        if word_count < 30:
            return True

        if content.count(".....") >= 3:
            return True

        lines = stripped.split("\n")
        if not lines:
            return True

        toc_line_count = 0
        for line in lines[:30]:
            line = line.strip()
            if "....." in line and len(line) < 120:
                toc_line_count += 1
            if line and len(line.split()) <= 3 and any(
                part.replace("-", "").isdigit()
                for part in line.split()
            ):
                toc_line_count += 1

        if len(lines) > 0 and toc_line_count / len(lines) > 0.30:
            return True

        return False

    def build_knowledge_context(self, query: str, hits: List[RetrievalHit]) -> ContextPacket:
        """Build a knowledge-answering prompt packet from chunk hits."""
        hits = self._trim_hits(hits, max_words=1800)
        filtered_hits: List[RetrievalHit] = []
        for hit in hits:
            if not self._is_toc_or_index_chunk(hit.content):
                filtered_hits.append(hit)

        if len(filtered_hits) < len(hits):
            logger.debug(
                "Filtered {} TOC/index chunks from context ({} remaining)",
                len(hits) - len(filtered_hits),
                len(filtered_hits)
            )

        grouped: "OrderedDict[tuple[str, int], List[RetrievalHit]]" = OrderedDict()
        seen_ids = set()
        for hit in filtered_hits:
            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            key = (hit.source_file or "unknown", int(hit.page_number or 0))
            grouped.setdefault(key, []).append(hit)

        formatted_sources = []
        ordered_hits: List[RetrievalHit] = []
        for index, ((source_file, page_number), group_hits) in enumerate(grouped.items(), start=1):
            merged_content = "\n\n".join(
                dict.fromkeys(hit.content.strip() for hit in group_hits if hit.content.strip())
            )
            ordered_hits.extend(group_hits)
            formatted_sources.append(
                f"[SOURCE {index}]\n[DOCUMENT: {source_file} | PAGE {page_number}]\n\n{merged_content}"
            )
        sources = "\n\n".join(formatted_sources)
        logger.debug("FINAL CONTEXT:\n{}", sources)
        return ContextPacket(
            query=query,
            chunk_hits=ordered_hits,
            memory_hits=[],
            system_prompt=self.KNOWLEDGE_SYSTEM,
            user_prompt=(
                f"Context:\n{sources}\n\n"
                f"Question: {query}\n\n"
                "Answer in plain conversational paragraphs using the context above. "
                "Combine relevant sections when needed."
            ),
            has_chunks=bool(ordered_hits),
            has_memories=False,
        )

    def build_memory_context(self, query: str, hits: List[RetrievalHit]) -> ContextPacket:
        """Build a memory-answering prompt packet from memory hits."""
        trimmed_hits = self._trim_hits(hits, max_words=450)
        formatted_memories = [f"[MEMORY {index}] {hit.content}" for index, hit in enumerate(trimmed_hits, start=1)]
        memories = "\n".join(formatted_memories)
        guidance = (
            "Answer in 1-3 short conversational sentences. "
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
