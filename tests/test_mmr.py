"""MMR reranker tests, including a hand-computable redundancy fixture.

The fixture below is small enough to verify on paper, proving the MMR
formula (lambda * relevance - (1 - lambda) * max-sim-to-selected) is
actually implemented, not a placeholder.
"""

import pytest

from agent.mmr import mmr_rerank

# Hand-computable fixture: docs 0 and 1 are IDENTICAL vectors (cosine 1),
# doc 2 is orthogonal to both (cosine 0). Relevance favors 0, then 1.
VECS = [{"x": 1.0}, {"x": 1.0}, {"y": 1.0}]
REL = [1.0, 0.95, 0.50]


def test_mmr_demotes_redundant_document():
    """With lambda=0.5 the duplicate must drop below the diverse doc.

    Hand computation:
      pick 1: scores = 0.5*rel = [0.500, 0.475, 0.250]      -> doc 0
      pick 2: doc1 = 0.5*0.95 - 0.5*sim(1,0)=0.475-0.500 = -0.025
              doc2 = 0.5*0.50 - 0.5*sim(2,0)=0.250-0.000 =  0.250 -> doc 2
      pick 3: only doc 1 remains.
    """
    assert mmr_rerank(VECS, REL, k=3, lam=0.5) == [0, 2, 1]


def test_mmr_lambda_one_is_pure_relevance_order():
    """lambda=1 removes the diversity term entirely."""
    assert mmr_rerank(VECS, REL, k=3, lam=1.0) == [0, 1, 2]


def test_mmr_lambda_zero_is_pure_diversity_after_first_pick():
    # First pick: all scores are -0*sim... = 0*rel - 1*0 = 0 for everyone;
    # strict '>' keeps the first candidate, doc 0. Second pick must be the
    # orthogonal doc 2 (penalty 0) over the duplicate doc 1 (penalty 1).
    order = mmr_rerank(VECS, REL, k=2, lam=0.0)
    assert order[1] == 2


def test_mmr_returns_k_unique_indices():
    order = mmr_rerank(VECS, REL, k=2, lam=0.7)
    assert len(order) == 2 and len(set(order)) == 2


def test_mmr_k_larger_than_pool_returns_all():
    assert sorted(mmr_rerank(VECS, REL, k=10, lam=0.7)) == [0, 1, 2]


def test_mmr_is_deterministic():
    runs = [mmr_rerank(VECS, REL, k=3, lam=0.5) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_mmr_input_validation():
    with pytest.raises(ValueError):
        mmr_rerank(VECS, [1.0], k=1)
    with pytest.raises(ValueError):
        mmr_rerank(VECS, REL, k=1, lam=1.5)
