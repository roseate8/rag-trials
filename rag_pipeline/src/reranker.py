"""
Reranker utilities for re-scoring retrieved chunks.

Default implementation uses a lightweight Cross-Encoder for passage reranking.
"""
from typing import List, Dict, Any, Optional
import logging

import torch
from sentence_transformers import CrossEncoder
import math

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """A simple, fast reranker using a cross-encoder model.

    Defaults to a compact model that runs well on CPU while providing solid quality.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        max_passage_chars: int = 1200,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_passage_chars = max_passage_chars

        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieved chunks using the cross-encoder.

        Args:
            query: User query
            retrieved_chunks: List of chunks with payloads (output of vector search)
            top_k: How many reranked chunks to keep
            batch_size: Cross-encoder batch size

        Returns:
            Top-k reranked chunks with an added 'rerank_score'.
        """
        if not retrieved_chunks:
            return []

        # Prepare query-passage pairs
        pairs = []
        passages: List[str] = []
        for item in retrieved_chunks:
            passage = item["payload"].get("text", "")
            if self.max_passage_chars and len(passage) > self.max_passage_chars:
                passage = passage[: self.max_passage_chars]
            passages.append(passage)
            pairs.append((query, passage))

        # Score pairs
        try:
            logger.info(
                f"Scoring {len(pairs)} (query, passage) pairs with reranker='{self.model_name}', batch_size={batch_size}"
            )
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranker scoring failed: {e}")
            return retrieved_chunks  # Fallback to original order

        # Attach scores and sort (handle NaN/inf robustly)
        bad_score_count = 0
        for item, score in zip(retrieved_chunks, scores):
            try:
                s = float(score)
            except Exception:
                s = float("nan")
            if not math.isfinite(s):
                bad_score_count += 1
                # Fallback to the original vector similarity score if available
                s = float(item.get("score", -1e9))
            item["rerank_score"] = s
        if bad_score_count:
            logger.warning(
                f"Reranker produced {bad_score_count} non-finite scores; fell back to vector similarity for those."
            )

        reranked = sorted(retrieved_chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[: top_k]


