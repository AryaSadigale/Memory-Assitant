# FILE: src/retrieval/hybrid_ranker.py
# CHANGES: Added source diversity enforcement on top of reciprocal rank fusion results.

from typing import Dict, List

from src.graph_models import RetrievalHit


class HybridRanker:
    """Fuse vector, BM25, and graph scores using Reciprocal Rank Fusion."""

    def rank(
        self,
        vector_hits: List[RetrievalHit],
        bm25_hits: List[RetrievalHit],
        graph_hits: List[RetrievalHit],
        top_k: int = 5,
    ) -> List[RetrievalHit]:
        """Rank unique retrieval hits using reciprocal rank fusion."""
        k = 60
        aggregated: Dict[str, RetrievalHit] = {}

        def merge_hits(hits: List[RetrievalHit], kind: str) -> None:
            """Merge one ranked list into the aggregated score map."""
            for rank, hit in enumerate(hits, start=1):
                contribution = 1.0 / (k + rank)
                if hit.id not in aggregated:
                    aggregated[hit.id] = RetrievalHit(
                        id=hit.id,
                        content=hit.content,
                        score=0.0,
                        source_type=hit.source_type,
                        source_file=hit.source_file,
                        page_number=hit.page_number,
                        vector_score=hit.vector_score,
                        bm25_score=hit.bm25_score,
                        graph_score=hit.graph_score,
                    )
                existing = aggregated[hit.id]
                existing.score += contribution
                existing.content = existing.content or hit.content
                existing.source_file = existing.source_file or hit.source_file
                existing.page_number = existing.page_number or hit.page_number
                if kind == "vector":
                    existing.vector_score = max(existing.vector_score, hit.score)
                elif kind == "bm25":
                    existing.bm25_score = max(existing.bm25_score, hit.score)
                elif kind == "graph":
                    existing.graph_score = max(existing.graph_score, hit.graph_score or hit.score)

        merge_hits(vector_hits, "vector")
        merge_hits(bm25_hits, "bm25")
        merge_hits(graph_hits, "graph")

        sorted_hits = sorted(aggregated.values(), key=lambda hit: hit.score, reverse=True)

        seen_sources: dict = {}
        diverse_results: List[RetrievalHit] = []
        for hit in sorted_hits:
            source = hit.source_file or "unknown"
            count = seen_sources.get(source, 0)
            if count < 2:
                diverse_results.append(hit)
                seen_sources[source] = count + 1
            if len(diverse_results) >= top_k:
                break

        return diverse_results
