# mak-market-agent

Market research agent with hybrid retrieval done properly: BM25 + TF-IDF
cosine score fusion, a real (implemented and unit-tested) MMR reranker,
deterministic extractive answers, and a FastAPI service — over a bundled,
original explainer corpus.

Author: Sama Mushtaq

## What it is

Ask a market-mechanics question ("why does the yield curve invert?",
"what is gamma hedging?") and the agent retrieves the most relevant,
least redundant passages from a 30-document corpus of short, original,
evergreen market explainers, then synthesizes an answer.

The answer path is honest by default: an extractive summarizer returns
top sentences verbatim from the retrieved passages (deterministic, no
network, no hallucination surface). An LLM synthesis path exists but is
strictly opt-in and only activates when an `ANTHROPIC_API_KEY` or
`GROQ_API_KEY` is present in the environment; it is never required.

## Architecture

```
                      +--------------------------+
  corpus/*.txt  --->  |  load_corpus()           |
  (30 original        |  title + body per doc    |
   explainers)        +------------+-------------+
                                   |
                 +-----------------+------------------+
                 |                                    |
        +--------v--------+                  +--------v--------+
        | BM25 (rank_bm25)|                  | TF-IDF cosine   |
        | lexical scoring |                  | (hand-rolled,   |
        +--------+--------+                  |  L2-normalized) |
                 |                           +--------+--------+
                 +---------—--+  +--------------------+
                              |  |
                      +-------v--v--------+
                      | score fusion      |   fused = a*minmax(bm25)
                      | (convex, tested)  |         + (1-a)*cosine
                      +---------+---------+
                                |  top-3k candidate pool
                      +---------v---------+
                      | MMR reranker      |   lam*rel - (1-lam)*maxsim
                      | (real, unit-      |   demotes redundant docs
                      |  tested fixture)  |
                      +---------+---------+
                                |  top-k passages
                      +---------v---------+
                      | synthesis         |   default: extractive
                      | extractive | LLM  |   optional: env-keyed LLM
                      +---------+---------+
                                |
                 CLI (python -m agent ask)  /  FastAPI (POST /query)
```

## Quickstart

```bash
pip install -r requirements.txt

# CLI
python -m agent ask "what happens when the yield curve inverts" --k 4

# API
python -m agent serve --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"question": "how do buybacks support the market", "k": 3}'
```

## Tests

```bash
pytest -v
```

24 tests cover: BM25 relevance sanity on the real corpus, TF-IDF cosine
on identical vs disjoint texts, fusion math against hand-computed
values plus alpha extremes and validation, the MMR fixture (below), MMR
determinism / k-handling / validation, both API endpoints via
`TestClient` including request validation and byte-for-byte determinism,
and a guarantee that every extractive answer sentence exists verbatim in
the retrieved passages.

**The MMR test is the point of this repo.** MMR here is implemented and
tested, not a placeholder. `tests/test_mmr.py` uses a three-document
fixture small enough to verify on paper — two identical vectors and one
orthogonal vector — and asserts the exact selection order `[0, 2, 1]`
at lambda 0.5: the duplicate of an already-selected document is demoted
below a less relevant but diverse document, which is the entire reason
MMR exists.

## Design decisions

- **Hybrid retrieval, then diversity.** BM25 catches exact-term matches;
  TF-IDF cosine catches distributional similarity; min-max fusion puts
  them on one scale. MMR then trades a little relevance for coverage,
  which matters when several corpus docs overlap (VIX vs gamma, rates vs
  yield curve).
- **Hand-rolled TF-IDF instead of sklearn.** For a 30-doc corpus a
  dict-based sparse vectorizer is a few dozen lines, keeps the
  dependency footprint at five packages, and makes every score in the
  tests hand-checkable.
- **Extractive default.** Sentences are quoted, not generated, so the
  default agent cannot fabricate claims about markets. The LLM path is
  labeled (`answer_source`) so callers always know which path produced
  the text.
- **Determinism everywhere:** stable tie-breaks in fusion sort, MMR
  selection and sentence ranking; the same query returns the same bytes.

## Limitations

- The corpus is bundled and static: 30 short educational explainers
  written for this repo. This is not live market data, not news, and
  not investment advice; answers describe general mechanics only.
- Retrieval quality is bounded by corpus size. With 30 docs, BM25 +
  TF-IDF + MMR is appropriate; a real deployment would need embeddings,
  chunking and a far larger corpus before the same design pays off.
- The extractive summarizer selects sentences; it does not paraphrase or
  resolve conflicts between passages.
- No authentication or rate limiting on the API; it is a local demo
  service, not a production deployment.
