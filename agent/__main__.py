"""CLI: python -m agent ask "..." --k 4  |  python -m agent serve"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent",
        description="Market research agent over a bundled explainer corpus.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask a question from the terminal")
    ask.add_argument("question", help="natural-language question")
    ask.add_argument("--k", type=int, default=4, help="passages to retrieve")
    ask.add_argument("--use-llm", action="store_true",
                     help="LLM synthesis if an API key is set")

    serve = sub.add_parser("serve", help="run the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return p


def cmd_ask(args: argparse.Namespace) -> int:
    from .retrieval import HybridRetriever
    from .synthesize import synthesize

    retriever = HybridRetriever()
    passages = retriever.retrieve(args.question, k=args.k)
    answer = synthesize(args.question, passages, use_llm=args.use_llm)
    out = sys.stdout
    out.write(f"Q: {args.question}\n\n")
    out.write(f"Answer ({answer.source}):\n{answer.text}\n\nSources:\n")
    for p in passages:
        out.write(f"  [{p.score:.4f}] {p.doc_id}: {p.title}\n")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("agent.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_ask(args) if args.command == "ask" else cmd_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
