# FILE: src/ingestion/chunk_splitter.py
# PURPOSE: Split parsed document pages into overlapping approximate-token chunks.

from typing import Dict, List
from uuid import NAMESPACE_URL, uuid5


class ChunkSplitter:
    """Split page text into overlapping token chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        """Initialize chunking parameters."""
        self.chunk_size = max(500, min(800, chunk_size))
        self.overlap = max(100, min(150, overlap))
        self.words_per_chunk = max(1, int(self.chunk_size / 0.75))
        self.overlap_words = max(0, int(self.overlap / 0.75))

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Heuristically detect section headings that should stay with following content."""
        stripped = line.strip()
        if not stripped:
            return False
        words = stripped.split()
        if len(words) > 10:
            return False
        alpha_words = [word for word in words if any(char.isalpha() for char in word)]
        if not alpha_words:
            return False
        if stripped.isupper():
            return True
        title_case_words = [
            word for word in alpha_words
            if word[:1].isupper() and word[1:].lower() == word[1:]
        ]
        if len(title_case_words) >= max(1, len(alpha_words) - 1):
            return True
        return False

    def split(self, pages: List[Dict]) -> List[Dict]:
        """Split page records into overlapping chunk dictionaries ready for embedding."""
        chunks: List[Dict] = []
        for page in pages:
            raw_lines = [line.strip() for line in page["content"].splitlines()]
            paragraphs: List[str] = []
            index = 0
            while index < len(raw_lines):
                line = raw_lines[index]
                if not line:
                    index += 1
                    continue
                if self._is_heading(line):
                    merged = line
                    next_index = index + 1
                    collected = []
                    while next_index < len(raw_lines):
                        next_line = raw_lines[next_index].strip()
                        if not next_line:
                            if collected:
                                break
                            next_index += 1
                            continue
                        if self._is_heading(next_line) and collected:
                            break
                        collected.append(next_line)
                        if len(" ".join(collected).split()) >= 120:
                            break
                        next_index += 1
                    if collected:
                        merged = f"{line}\n" + "\n".join(collected)
                        index = next_index + 1
                    else:
                        index += 1
                    paragraphs.append(merged.strip())
                    continue

                collected = [line]
                next_index = index + 1
                while next_index < len(raw_lines):
                    next_line = raw_lines[next_index].strip()
                    if not next_line:
                        break
                    if self._is_heading(next_line):
                        break
                    collected.append(next_line)
                    next_index += 1
                paragraphs.append(" ".join(collected).strip())
                index = next_index + 1 if next_index < len(raw_lines) and not raw_lines[next_index].strip() else next_index

            units = [paragraph for paragraph in paragraphs if paragraph]
            if not units:
                continue

            unit_words = [unit.split() for unit in units]
            start = 0
            chunk_index = 0
            while start < len(unit_words):
                end = start
                chunk_words: List[str] = []
                while end < len(unit_words):
                    next_words = unit_words[end]
                    if chunk_words and len(chunk_words) + len(next_words) > self.words_per_chunk:
                        break
                    chunk_words.extend(next_words)
                    end += 1
                    if len(chunk_words) >= self.words_per_chunk:
                        break

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
                if end == len(unit_words):
                    break

                overlap_target = min(self.overlap_words, len(chunk_words))
                overlap_words = 0
                new_start = end
                while new_start > start and overlap_words < overlap_target:
                    new_start -= 1
                    overlap_words += len(unit_words[new_start])
                start = max(start + 1, new_start)
                chunk_index += 1
        return chunks
