# FILE: src/retrieval/bm25_search.py
# CHANGES: Added user-scoped BM25 retrieval while preserving diagnostics and shared-pool visibility.

import re
from typing import List

from loguru import logger

from src.graph.neo4j_client import Neo4jClient
from src.graph_models import RetrievalHit


class BM25Search:
    """Fulltext search over Neo4j fulltext indexes."""

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize the BM25 search service."""
        self.client = client

    def _escape_query(self, query: str) -> str:
        """
        Escape only true Lucene special characters.
        Preserve all alphanumeric content and spaces.
        """
        lucene_special = set('+-&|!(){}[]^"~*?:\\/')
        escaped = ""
        for char in query:
            if char in lucene_special:
                escaped += "\\" + char
            else:
                escaped += char
        escaped = escaped.strip()
        if not escaped:
            return "*"
        return escaped

    async def search_chunks(self, query: str, top_k: int = 10, user_id: str = "default") -> List[RetrievalHit]:
        """Fulltext BM25 search over chunk content."""
        if not query or not query.strip():
            logger.warning("BM25 search called with empty query")
            return []

        escaped = self._escape_query(query)
        logger.debug("BM25 search query: '{}'", escaped)

        try:
            results = await self.client.run_query(
                "CALL db.index.fulltext.queryNodes("
                "  'chunk_fulltext', $query"
                ") YIELD node, score "
                "WHERE node.user_id = $user_id "
                "   OR node.user_id IS NULL "
                "RETURN node.chunk_id AS chunk_id, "
                "       node.content AS content, "
                "       node.source_file AS source_file, "
                "       node.page_number AS page_number, "
                "       score "
                "LIMIT $top_k",
                {"query": escaped, "top_k": top_k, "user_id": user_id},
            )

            logger.debug("BM25 search: {} hits returned", len(results))
            if not results:
                logger.warning("BM25 search returned zero results for query '{}'", query)

            return [
                RetrievalHit(
                    id=result["chunk_id"],
                    content=result["content"],
                    score=float(result["score"]),
                    source_type="chunk",
                    source_file=result["source_file"] or "",
                    page_number=result["page_number"] or 0,
                    bm25_score=float(result["score"]),
                )
                for result in results
            ]

        except Exception as exc:
            logger.error("BM25 search failed: {}", exc)
            return []

    async def search_memories(self, query: str, user_id: str, top_k: int = 10) -> List[RetrievalHit]:
        """Search memory nodes by fulltext score for a specific user."""
        if not query or not query.strip():
            logger.warning("Memory BM25 search called with empty query")
            return []

        escaped = self._escape_query(query)
        logger.debug("Memory BM25 search query: '{}'", escaped)

        stopwords = {
            "the", "and", "for", "with", "that", "this", "from",
            "tell", "about", "what", "know", "myself", "your", "you",
            "are", "how", "who", "why", "when", "where", "give", "show",
            "can", "did", "have", "been", "into", "more", "also",
        }
        words = [
            word
            for word in re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
            if word not in stopwords
        ]

        if not words:
            logger.debug("Memory BM25: no meaningful words, skipping fulltext")
            return []

        queries_to_try = [escaped]
        keyword_query = " OR ".join(words)
        if keyword_query != escaped:
            queries_to_try.append(keyword_query)

        for attempt_query in queries_to_try:
            try:
                rows = await self.client.run_query(
                    "CALL db.index.fulltext.queryNodes("
                    "  'memory_fulltext', $query"
                    ") YIELD node, score "
                    "WHERE node.user_id = $user_id "
                    "RETURN node.memory_id AS id, "
                    "       node.content AS content, "
                    "       score AS score "
                    "LIMIT $top_k",
                    {
                        "query": attempt_query,
                        "user_id": user_id,
                        "top_k": top_k,
                    },
                )
                logger.debug(
                    "Memory BM25 search: {} hits returned (query: '{}')",
                    len(rows),
                    attempt_query[:50],
                )
                if rows:
                    return [
                        RetrievalHit(
                            id=row["id"],
                            content=row["content"],
                            score=float(row["score"]),
                            source_type="memory",
                            bm25_score=float(row["score"]),
                        )
                        for row in rows
                    ]
            except Exception as exc:
                logger.error("Memory BM25 search failed: {}", exc)
                return []

        logger.debug("Memory BM25: all query attempts returned zero results")
        return []
