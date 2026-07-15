"""Tokenization and hand-rolled TF-IDF cosine similarity.

Kept dependency-light on purpose: no sklearn, just dict-based sparse
vectors with L2 normalization, which is plenty for a ~30-doc corpus and
keeps every number in the pipeline easy to verify by hand in tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or "
    "that the this to was were will with".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with a minimal stopword filter."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


SparseVec = Dict[str, float]


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer producing L2-normalized sparse vectors.

    idf(t) = ln((1 + N) / (1 + df(t))) + 1  (smoothed, sklearn-style)
    tf(t, d) = count(t, d) / len(d)
    """

    def __init__(self) -> None:
        self.idf: Dict[str, float] = {}
        self._n_docs = 0

    def fit(self, docs: Sequence[str]) -> "TfidfVectorizer":
        self._n_docs = len(docs)
        df: Counter = Counter()
        for doc in docs:
            df.update(set(tokenize(doc)))
        self.idf = {
            t: math.log((1 + self._n_docs) / (1 + c)) + 1.0
            for t, c in df.items()
        }
        return self

    def transform(self, text: str) -> SparseVec:
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec = {
            t: (c / len(tokens)) * self.idf.get(t, 0.0)
            for t, c in counts.items()
            if self.idf.get(t, 0.0) > 0.0
        }
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0.0:
            return {}
        return {t: v / norm for t, v in vec.items()}


def cosine(a: SparseVec, b: SparseVec) -> float:
    """Cosine similarity of two L2-normalized sparse vectors."""
    if len(b) < len(a):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
