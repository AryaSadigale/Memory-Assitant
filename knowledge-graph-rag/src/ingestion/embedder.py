# FILE: src/ingestion/embedder.py
# CHANGES: Configured SentenceTransformer to use a persistent cache folder for model downloads.

import os
import time
from typing import List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embedder:
    """Lazy-loading sentence transformer embedding service."""

    def __init__(self, model_name: str, device: str) -> None:
        """Store the embedding model configuration."""
        self.model_name = model_name
        self.device = device
        self.model: Optional[SentenceTransformer] = None

    def _load_model(self) -> None:
        """Load the embedding model on first use and log loading time."""
        if self.model is not None:
            return
        resolved_device = self.device
        if self.device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        started = time.perf_counter()
        cache_dir = os.environ.get(
            "TRANSFORMERS_CACHE",
            "/root/.cache/huggingface"
        )
        self.model = SentenceTransformer(
            self.model_name,
            device=resolved_device,
            cache_folder=cache_dir
        )
        logger.info(
            "Loaded embedding model {} on {} in {:.2f}s",
            self.model_name,
            resolved_device,
            time.perf_counter() - started,
        )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        self._load_model()
        assert self.model is not None
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches and show progress for larger workloads."""
        self._load_model()
        assert self.model is not None
        if not texts:
            return []
        batch_size = 32
        embeddings: List[List[float]] = []
        iterator = range(0, len(texts), batch_size)
        if len(texts) > 10:
            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Embedding")
        for start in iterator:
            batch = texts[start : start + batch_size]
            batch_vectors = self.model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_vectors.tolist())
        return embeddings
