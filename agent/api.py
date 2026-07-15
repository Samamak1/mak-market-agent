"""FastAPI app exposing the market research agent.

GET  /health          -> status + corpus size
POST /query           -> retrieved passages + synthesized answer
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .retrieval import HybridRetriever
from .synthesize import synthesize


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="natural-language question")
    k: int = Field(default=4, ge=1, le=10, description="passages to retrieve")
    use_llm: bool = Field(
        default=False,
        description="use an LLM for synthesis if an API key is set "
                    "(default: deterministic extractive summary)",
    )


class PassageOut(BaseModel):
    doc_id: str
    title: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    passages: list[PassageOut]
    answer: str
    answer_source: str


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Build the index once per process."""
    return HybridRetriever()


def create_app() -> FastAPI:
    app = FastAPI(
        title="mak-market-agent",
        version=__version__,
        description="Hybrid-retrieval market research agent over a "
                    "bundled, original explainer corpus (not live data).",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "corpus_docs": len(get_retriever().documents),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        retriever = get_retriever()
        passages = retriever.retrieve(req.question, k=req.k)
        answer = synthesize(req.question, passages, use_llm=req.use_llm)
        return QueryResponse(
            question=req.question,
            passages=[PassageOut(doc_id=p.doc_id, title=p.title,
                                 score=p.score, text=p.text)
                      for p in passages],
            answer=answer.text,
            answer_source=answer.source,
        )

    return app


app = create_app()
