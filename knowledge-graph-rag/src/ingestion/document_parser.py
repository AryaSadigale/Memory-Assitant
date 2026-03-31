# FILE: src/ingestion/document_parser.py
# PURPOSE: Parse PDF and plain text documents into normalized page-like records.

import os
import re
from typing import Dict, List

import fitz


class DocumentParser:
    """Parse PDF and plain text files into raw text chunks by page."""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize repeated whitespace into single spaces."""
        return re.sub(r"\s+", " ", text).strip()

    def parse_pdf(self, file_path: str) -> List[Dict]:
        """Extract normalized text from each PDF page and skip very short pages."""
        pages: List[Dict] = []
        source_file = os.path.basename(file_path)
        with fitz.open(file_path) as document:
            for page_index, page in enumerate(document, start=1):
                content = self._normalize_whitespace(page.get_text("text"))
                if len(content) < 50:
                    continue
                pages.append(
                    {
                        "page_number": page_index,
                        "content": content,
                        "source_file": source_file,
                    }
                )
        return pages

    def parse_text(self, file_path: str) -> List[Dict]:
        """Stream a text file into logical sections of roughly 2000 characters each."""
        sections: List[Dict] = []
        source_file = os.path.basename(file_path)
        buffer: List[str] = []
        current_length = 0
        target_size = 2000
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                normalized = self._normalize_whitespace(line)
                if not normalized:
                    continue
                buffer.append(normalized)
                current_length += len(normalized) + 1
                if current_length >= target_size:
                    content = self._normalize_whitespace(" ".join(buffer))
                    if content:
                        sections.append(
                            {
                                "page_number": 0,
                                "content": content,
                                "source_file": source_file,
                            }
                        )
                    buffer = []
                    current_length = 0
        if buffer:
            content = self._normalize_whitespace(" ".join(buffer))
            if content:
                sections.append(
                    {
                        "page_number": 0,
                        "content": content,
                        "source_file": source_file,
                    }
                )
        return sections

    def parse_file(self, file_path: str) -> List[Dict]:
        """Route parsing based on file extension."""
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return self.parse_pdf(file_path)
        if extension in {".txt", ".md", ".text"}:
            return self.parse_text(file_path)
        raise ValueError(f"Unsupported file type: {extension}")
