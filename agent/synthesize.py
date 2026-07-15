"""Answer synthesis: deterministic extractive summary, optional LLM.

Default path is purely extractive: sentences from the retrieved passages
are scored by IDF-weighted overlap with the question and the top ones are
returned verbatim (deterministic, no network). If use_llm=True AND an
ANTHROPIC_API_KEY or GROQ_API_KEY exists in the environment, an LLM
drafts the answer from the same passages instead; failures fall back to
the extractive path. The answer is always labeled with its source.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .retrieval import RetrievedPassage
from .textutils import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    source: str  # "extractive" or "llm:<provider>"


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text.replace("\n", " "))
            if s.strip()]


def extractive_answer(question: str, passages: List[RetrievedPassage],
                      max_sentences: int = 3) -> Answer:
    """Pick the sentences most lexically aligned with the question.

    Sentence score = sum over overlapping tokens of an IDF-like weight
    computed across the candidate sentences. Deterministic: ties break
    on (passage order, sentence order).
    """
    sentences: List[Tuple[int, int, str]] = []  # (passage_idx, sent_idx, s)
    for p_idx, p in enumerate(passages):
        for s_idx, s in enumerate(split_sentences(p.text)):
            sentences.append((p_idx, s_idx, s))
    if not sentences:
        return Answer("No relevant passages found in the corpus.",
                      "extractive")

    df: Counter = Counter()
    for _, _, s in sentences:
        df.update(set(tokenize(s)))
    n = len(sentences)
    q_tokens = set(tokenize(question))

    def score(s: str) -> float:
        return sum(
            math.log((1 + n) / (1 + df[t])) + 1.0
            for t in set(tokenize(s)) & q_tokens
        )

    ranked = sorted(
        sentences, key=lambda item: (-score(item[2]), item[0], item[1])
    )
    chosen = [item for item in ranked[:max_sentences] if score(item[2]) > 0]
    if not chosen:
        chosen = ranked[:1]
    # Present chosen sentences in their original corpus order.
    chosen.sort(key=lambda item: (item[0], item[1]))
    return Answer(" ".join(s for _, _, s in chosen), "extractive")


def _llm_provider() -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def _llm_answer(question: str, passages: List[RetrievedPassage],
                provider: str) -> Answer:
    import httpx  # lazy: never needed on the default path

    context = "\n\n".join(f"[{p.doc_id}] {p.title}\n{p.text}"
                          for p in passages)
    prompt = (
        "Answer the question using ONLY the context passages. "
        "Be concise and cite doc ids in brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    if provider == "anthropic":
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": "claude-haiku-4-5", "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return Answer(resp.json()["content"][0]["text"].strip(),
                      "llm:anthropic")
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={"model": "llama-3.1-8b-instant", "max_tokens": 400,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    resp.raise_for_status()
    return Answer(
        resp.json()["choices"][0]["message"]["content"].strip(), "llm:groq"
    )


def synthesize(question: str, passages: List[RetrievedPassage],
               use_llm: bool = False) -> Answer:
    """Produce an answer; extractive by default, LLM only if opted in."""
    provider = _llm_provider() if use_llm else None
    if provider is not None:
        try:
            return _llm_answer(question, passages, provider)
        except Exception:
            pass  # never let the optional path break the agent
    return extractive_answer(question, passages)
