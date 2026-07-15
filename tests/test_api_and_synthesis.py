"""FastAPI endpoint tests (TestClient) and synthesis tests."""

from fastapi.testclient import TestClient

from agent.api import create_app
from agent.retrieval import HybridRetriever, RetrievedPassage
from agent.synthesize import extractive_answer, split_sentences, synthesize

client = TestClient(create_app())


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["corpus_docs"] == 30


def test_query_endpoint_returns_passages_and_answer():
    resp = client.post("/query", json={
        "question": "why does the yield curve invert before recessions",
        "k": 3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["passages"]) == 3
    assert body["answer"].strip()
    assert body["answer_source"] == "extractive"  # no key, no LLM
    ids = [p["doc_id"] for p in body["passages"]]
    assert "06_yield_curve" in ids


def test_query_endpoint_is_deterministic():
    payload = {"question": "what moves gold prices", "k": 4}
    a = client.post("/query", json=payload).json()
    b = client.post("/query", json=payload).json()
    assert a == b


def test_query_validation_rejects_bad_k():
    assert client.post("/query", json={"question": "x", "k": 0}).status_code == 422
    assert client.post("/query", json={"question": "", "k": 3}).status_code == 422


def test_extractive_answer_uses_only_retrieved_text():
    r = HybridRetriever()
    q = "what does high short interest mean"
    passages = r.retrieve(q, k=3)
    ans = extractive_answer(q, passages)
    assert ans.source == "extractive"
    corpus_text = " ".join(p.text.replace("\n", " ") for p in passages)
    for sentence in split_sentences(ans.text):
        assert sentence in corpus_text, f"fabricated sentence: {sentence}"


def test_synthesize_without_key_stays_extractive(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    p = [RetrievedPassage("d1", "T", "Rates set the price of money. "
                          "Bonds reprice first.", 1.0)]
    ans = synthesize("what do rates do", p, use_llm=True)
    assert ans.source == "extractive"


def test_extractive_answer_empty_passages():
    ans = extractive_answer("anything", [])
    assert ans.source == "extractive" and ans.text
