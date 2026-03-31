# FILE: src/ingestion/chunk_splitter.py
# PURPOSE: Split parsed document pages into overlapping approximate-token chunks.

from typing import Dict, List
from uuid import NAMESPACE_URL, uuid5


class ChunkSplitter:
    """Split page text into overlapping token chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        """Initialize chunking parameters."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.words_per_chunk = max(1, int(chunk_size / 0.75))
        self.overlap_words = max(0, int(overlap / 0.75))

    def split(self, pages: List[Dict]) -> List[Dict]:
        """Split page records into overlapping chunk dictionaries ready for embedding."""
        chunks: List[Dict] = []
        for page in pages:
            words = page["content"].split()
            if not words:
                continue
            start = 0
            chunk_index = 0
            while start < len(words):
                end = min(len(words), start + self.words_per_chunk)
                chunk_words = words[start:end]
                content = " ".join(chunk_words).strip()
                if content:
                    chunk_id = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{page['source_file']}:{page['page_number']}:{chunk_index}:{content[:200]}",
                        )
                    )
                    token_count = max(1, int(len(chunk_words) * 0.75))
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "source_file": page["source_file"],
                            "page_number": int(page["page_number"]),
                            "content": content,
                            "token_count": token_count,
                        }
                    )
                if end == len(words):
                    break
                start = max(0, end - self.overlap_words)
                chunk_index += 1
        return chunks
