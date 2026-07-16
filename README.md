# MAK Market Agent: Educational Retrieval Prototype

> **Status - portfolio prototype:** This is a bounded educational demonstration over a bundled, static corpus. It is not connected to live market data, news, brokerage, client, or production systems. It does not generate trade recommendations, investment advice, forecasts, or performance claims.

This repository demonstrates how an educational market-mechanics question can move through retrieval, diversity ranking, evidence selection, and a labeled answer path. The default behavior is deterministic and extractive so users can trace each answer sentence to the source corpus.

The MAK name links the prototype thematically to my earlier education work; this 2026 build is not a reconstruction of a production Trade with MAK system.

## Business purpose

| Area | Scope |
|---|---|
| User need | Find concise explanations of evergreen market mechanics without presenting generated text as live market research |
| Product decision | Use a small original corpus and an extractive default to maximize traceability |
| Controls | Deterministic retrieval, labeled answer source, optional environment-keyed LLM path, and no required network access |
| Delivery surfaces | Command-line interface and local FastAPI service |
| Acceptance evidence | 24 tests covering retrieval, fusion, MMR, API validation, determinism, and extractive grounding |

## My role

I defined the use case, operating boundaries, architecture requirements, safe-default behavior, acceptance criteria, and review process. The project was deliberately scoped as a local educational prototype rather than a live research or advisory product.

## What it does

A user can ask a general market-mechanics question such as 'why does the yield curve invert?' or 'what is gamma hedging?' The service:

1. loads 30 bundled evergreen explainers;
2. scores documents with BM25 and TF-IDF cosine similarity;
3. fuses normalized lexical and vector scores;
4. applies maximal marginal relevance (MMR) to reduce redundancy;
5. returns the top passages; and
6. produces either a deterministic extractive answer or, only when explicitly configured, a labeled LLM-assisted synthesis.

The default extractive summarizer selects sentences verbatim from retrieved passages. The optional LLM path activates only when `ANTHROPIC_API_KEY` or `GROQ_API_KEY` is present and the caller requests it. Responses expose `answer_source` so the path is visible.

## Architecture

```text
                      +--------------------------+
  corpus/*.txt  --->  | load_corpus()            |
  30 explainers       | title + body per document|
                      +------------+-------------+
                                   |
                 +-----------------+------------------+
                 |                                    |
        +--------v--------+                  +--------v--------+
        | BM25            |                  | TF-IDF cosine   |
        | lexical scoring |                  | hand-rolled,    |
        +--------+--------+                  | L2-normalized    |
                 |                           +--------+--------+
                 +------------+  +--------------------+
                              |  |
                      +-------v--v--------+
                      | score fusion      | fused = a*minmax(BM25)
                      | convex, tested    |       + (1-a)*cosine
                      +---------+---------+
                                | top-3k candidate pool
                      +---------v---------+
                      | MMR reranker      | lambda*relevance
                      | deterministic     | - (1-lambda)*redundancy
                      +---------+---------+
                                | top-k passages
                      +---------v---------+
                      | synthesis         | default: extractive
                      | extractive or LLM | optional: labeled LLM
                      +---------+---------+
                                |
                 CLI (`python -m agent ask`) / FastAPI (`POST /query`)
```

## Repository structure

```text
agent/api.py          FastAPI endpoints and request validation
agent/retrieval.py    BM25, TF-IDF, score normalization, and fusion
agent/mmr.py          maximal marginal relevance reranking
agent/synthesize.py   extractive default and optional LLM path
agent/textutils.py    text parsing and sentence utilities
agent/__main__.py     CLI and local service entry points
corpus/               30 bundled market-mechanics explainers
tests/                retrieval, reranking, synthesis, and API tests
```

## Quickstart

```bash
pip install -r requirements.txt

# CLI
python -m agent ask "what happens when the yield curve inverts" --k 4

# Local API
python -m agent serve --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"question": "how do buybacks support the market", "k": 3}'
```

No external key is required for the default path.

## Verification

```bash
pytest -v
```

The 24-test suite covers:

- BM25 relevance behavior on the bundled corpus;
- TF-IDF cosine behavior on identical and disjoint text;
- fusion math against hand-computed values, including alpha extremes and validation;
- MMR determinism, input validation, and exact selection order on a three-document fixture;
- both FastAPI endpoints through `TestClient`, including request validation;
- byte-for-byte deterministic responses; and
- confirmation that each extractive answer sentence exists verbatim in a retrieved passage.

The MMR fixture uses two identical vectors and one orthogonal vector. At lambda 0.5, the expected selection order is `[0, 2, 1]`, demonstrating that a duplicate of an already selected document is demoted below a less redundant candidate.

## Design decisions

- **Hybrid retrieval before diversity:** BM25 captures exact-term matches while TF-IDF cosine captures broader term-distribution similarity. Normalized fusion puts both scores on a comparable range before MMR trades some relevance for coverage.
- **Small, inspectable vectorizer:** for a 30-document corpus, a dictionary-based sparse TF-IDF implementation keeps the dependency footprint limited and makes score calculations easier to inspect.
- **Extractive default:** the local default cannot invent new market claims because it selects sentences from retrieved text. Optional generated synthesis is explicitly labeled.
- **Determinism:** stable tie-breaking is used in fusion, MMR selection, and sentence ranking.
- **No live-data pretense:** the corpus is static and the service does not represent itself as news, research, or a current-market system.

## Acceptance status

| Requirement | Status |
|---|---|
| Deterministic local answer path | Implemented and tested |
| Hybrid BM25 and TF-IDF retrieval | Implemented and tested |
| MMR diversity reranking | Implemented and tested |
| CLI and local API | Implemented and tested |
| Optional labeled LLM synthesis | Implemented; requires caller configuration |
| Live market/news ingestion | Not implemented; out of scope |
| Authentication and rate limiting | Not implemented; local demo only |
| Production deployment | Not claimed |

## Limitations

- The corpus is bundled and static: 30 short educational explainers created for this repository.
- Retrieval quality is bounded by corpus size. A larger deployment would require source governance, chunking, evaluation data, and likely embeddings.
- Extractive summarization selects sentences; it does not resolve conflicts or guarantee completeness.
- Optional LLM output can be incomplete or incorrect even when grounded passages are supplied.
- The API has no authentication, authorization, rate limiting, monitoring, or production hardening.
- Nothing in the repository is live market research or investment advice.

## Authorship and provenance

I defined the use case, requirements, controls, acceptance criteria, and review process. Implementation and documentation were developed with AI assistance under my direction and validated against the repository's tests and stated limitations.

Author: Sama Mushtaq
