"""Maximal Marginal Relevance (MMR) reranking — real implementation.

score(d) = lambda * relevance(d) - (1 - lambda) * max_{s in selected} sim(d, s)

Greedy selection; ties break toward the lower candidate index so the
output is fully deterministic. This module is unit-tested against a
hand-computable fixture in tests/test_mmr.py proving that redundant
documents are demoted.
"""

from __future__ import annotations

from typing import List, Sequence

from .textutils import SparseVec, cosine


def mmr_rerank(
    doc_vectors: Sequence[SparseVec],
    relevance: Sequence[float],
    k: int,
    lam: float = 0.7,
) -> List[int]:
    """Return indices of up to k documents in MMR selection order.

    Args:
        doc_vectors: one L2-normalized sparse vector per candidate.
        relevance: query relevance score per candidate (same order).
        k: number of documents to select.
        lam: trade-off in [0, 1]. 1.0 = pure relevance ranking,
            0.0 = pure diversity.

    Raises:
        ValueError: on length mismatch or lam outside [0, 1].
    """
    if len(doc_vectors) != len(relevance):
        raise ValueError("doc_vectors and relevance must be equal length")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")

    candidates = list(range(len(doc_vectors)))
    selected: List[int] = []
    while candidates and len(selected) < k:
        best_idx = candidates[0]
        best_score = float("-inf")
        for i in candidates:
            max_sim = max(
                (cosine(doc_vectors[i], doc_vectors[j]) for j in selected),
                default=0.0,
            )
            score = lam * relevance[i] - (1.0 - lam) * max_sim
            if score > best_score:  # strict '>' keeps lowest index on ties
                best_score = score
                best_idx = i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected
