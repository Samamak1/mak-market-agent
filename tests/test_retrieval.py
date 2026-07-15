"""Retrieval tests: BM25 sanity, TF-IDF cosine, fusion math, determinism."""

import pytest

from agent.retrieval import (HybridRetriever, fuse_scores, load_corpus,
                             minmax_normalize)
from agent.textutils import TfidfVectorizer, cosine


def test_corpus_loads_thirty_docs_with_titles():
    docs = load_corpus()
    assert len(docs) == 30
    assert all(d.title and d.text for d in docs)


def test_bm25_relevance_sanity():
    """A rates question must rank the rates doc above the volume doc."""
    r = HybridRetriever()
    scores = r.score("how do central bank interest rates move markets")
    ids = [d.doc_id for d in r.documents]
    bm25 = dict(zip(ids, scores["bm25"]))
    assert bm25["01_interest_rates"] > bm25["15_volume_analysis"]


def test_hybrid_retrieval_finds_topical_doc():
    r = HybridRetriever()
    got = r.retrieve("what is the vix fear gauge implied volatility", k=3)
    assert "05_vix_volatility" in [p.doc_id for p in got]


def test_tfidf_cosine_identical_and_disjoint_texts():
    v = TfidfVectorizer().fit(["bonds yields curve", "gold dollar haven"])
    a = v.transform("bonds yields curve")
    b = v.transform("bonds yields curve")
    c = v.transform("gold dollar haven")
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, c) == pytest.approx(0.0)


def test_minmax_normalize_known_values():
    assert minmax_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert minmax_normalize([3.0, 3.0]) == [0.0, 0.0]  # constant guard


def test_fusion_math_hand_computed():
    """alpha=0.5, bm25=[0,10], cosine=[1.0,0.0].

    minmax(bm25) = [0, 1]
    fused = [0.5*0 + 0.5*1.0, 0.5*1 + 0.5*0.0] = [0.5, 0.5]
    """
    fused = fuse_scores([0.0, 10.0], [1.0, 0.0], alpha=0.5)
    assert fused == pytest.approx([0.5, 0.5])


def test_fusion_alpha_extremes():
    bm25, cos = [0.0, 10.0], [1.0, 0.0]
    assert fuse_scores(bm25, cos, alpha=1.0) == pytest.approx([0.0, 1.0])
    assert fuse_scores(bm25, cos, alpha=0.0) == pytest.approx([1.0, 0.0])


def test_fusion_input_validation():
    with pytest.raises(ValueError):
        fuse_scores([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        fuse_scores([1.0], [1.0], alpha=2.0)


def test_retrieval_is_deterministic():
    r = HybridRetriever()
    q = "sector rotation between defensives and cyclicals"
    a = [(p.doc_id, p.score) for p in r.retrieve(q, k=5)]
    b = [(p.doc_id, p.score) for p in r.retrieve(q, k=5)]
    assert a == b


def test_retrieve_respects_k():
    r = HybridRetriever()
    assert len(r.retrieve("interest rates", k=2)) == 2
    with pytest.raises(ValueError):
        r.retrieve("interest rates", k=0)
