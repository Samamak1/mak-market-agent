"""Hybrid retrieval: BM25 + TF-IDF cosine, score fusion, MMR reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi

from .mmr import mmr_rerank
from .textutils import SparseVec, TfidfVectorizer, cosine, tokenize

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedPassage:
    doc_id: str
    title: str
    text: str
    score: float


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> List[Document]:
    """Load *.txt docs; first line is the title, rest is the body."""
    docs: List[Document] = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        docs.append(Document(doc_id=path.stem, title=title, text=body))
    if not docs:
        raise FileNotFoundError(f"no .txt documents found in {corpus_dir}")
    return docs


def minmax_normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores to [0, 1]; a constant vector maps to all zeros."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_scores(
    bm25_scores: Sequence[float],
    cosine_scores: Sequence[float],
    alpha: float = 0.5,
) -> List[float]:
    """Convex fusion: alpha * minmax(bm25) + (1 - alpha) * cosine.

    Cosine scores are already in [0, 1]; BM25 scores are min-max scaled
    over the current candidate set before mixing.
    """
    if len(bm25_scores) != len(cosine_scores):
        raise ValueError("score lists must be equal length")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    bm = minmax_normalize(bm25_scores)
    return [alpha * b + (1.0 - alpha) * c for b, c in zip(bm, cosine_scores)]


class HybridRetriever:
    """BM25 + TF-IDF cosine with score fusion and MMR reranking."""

    def __init__(self, documents: List[Document] | None = None,
                 alpha: float = 0.5, mmr_lambda: float = 0.7) -> None:
        self.documents = documents if documents is not None else load_corpus()
        self.alpha = alpha
        self.mmr_lambda = mmr_lambda
        full_texts = [f"{d.title}\n{d.text}" for d in self.documents]
        self._bm25 = BM25Okapi([tokenize(t) for t in full_texts])
        self._vectorizer = TfidfVectorizer().fit(full_texts)
        self._doc_vecs: List[SparseVec] = [
            self._vectorizer.transform(t) for t in full_texts
        ]

    def score(self, question: str) -> Dict[str, List[float]]:
        """Per-document bm25 / cosine / fused scores (for inspection)."""
        q_tokens = tokenize(question)
        bm25 = [float(s) for s in self._bm25.get_scores(q_tokens)]
        q_vec = self._vectorizer.transform(question)
        cos = [cosine(q_vec, dv) for dv in self._doc_vecs]
        return {"bm25": bm25, "cosine": cos,
                "fused": fuse_scores(bm25, cos, self.alpha)}

    def retrieve(self, question: str, k: int = 4) -> List[RetrievedPassage]:
        """Top-k passages: fused-score candidates reranked with MMR."""
        if k < 1:
            raise ValueError("k must be >= 1")
        fused = self.score(question)["fused"]
        # Candidate pool: top 3k by fused score (deterministic tie-break
        # on index), then MMR-rerank the pool for diversity.
        pool_size = min(len(self.documents), max(3 * k, k))
        pool = sorted(range(len(fused)),
                      key=lambda i: (-fused[i], i))[:pool_size]
        order = mmr_rerank(
            [self._doc_vecs[i] for i in pool],
            [fused[i] for i in pool],
            k=k, lam=self.mmr_lambda,
        )
        out: List[RetrievedPassage] = []
        for local_idx in order:
            i = pool[local_idx]
            d = self.documents[i]
            out.append(RetrievedPassage(d.doc_id, d.title, d.text,
                                        round(fused[i], 6)))
        return out
