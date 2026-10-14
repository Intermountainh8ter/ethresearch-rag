from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"
SUBCOMMANDS = ("ask", "build-index", "serve", "mcp", "related", "stats")


class CliError(Exception):
    """A user-facing failure: printed as ``error: ...`` with exit status 1, no traceback."""


def _add_source(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=os.environ.get("ETHRAG_DATA", str(DEFAULT_DATA)),
                   help="directory of markdown specs (searched recursively)")
    p.add_argument("--index", default=os.environ.get("ETHRAG_INDEX"),
                   help="directory written by `ethrag build-index` (preferred over --data)")


def _add_pipeline_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", action="store_true", help="use Anthropic (needs ANTHROPIC_API_KEY)")
    p.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    p.add_argument("--rerank", metavar="WEIGHTS.json", help="rerank with trained logistic weights")
    p.add_argument("--expand-graph", action="store_true", help="add chunks from `requires` neighbours")
    p.add_argument("--min-coverage", type=float, default=0.7,
                   help="share of an answer sentence's content words that the cited chunks must contain")
    p.add_argument("--min-relevance", type=float, default=0.1,
                   help="share of the question's topical words the answer must repeat")
    p.add_argument("--min-novelty", type=float, default=0.0,
                   help="reject answers that mostly restate the question (0 = off)")
    p.add_argument("--no-negation-check", action="store_true")


def build_pipeline(a: argparse.Namespace):
    from .pipeline import RagPipeline

    llm = None
    if getattr(a, "llm", False):
        from .llm import AnthropicLLM, api_key_available

        if not api_key_available():
            raise CliError("--llm needs ANTHROPIC_API_KEY in the environment (omit --llm to use "
                           "the offline extractive baseline)")
        try:
            llm = AnthropicLLM()
        except ImportError as e:
            raise CliError("--llm needs the optional dependency: pip install 'ethrag[llm]'") from e
    kw = {}
    if hasattr(a, "k"):
        kw = {"k": a.k, "expand_graph": a.expand_graph, "min_coverage": a.min_coverage,
              "min_relevance": a.min_relevance, "min_novelty": a.min_novelty,
              "check_negation": not a.no_negation_check}
        if a.rerank:
            from .rerank import LogisticReranker

            kw["reranker"] = LogisticReranker.load(a.rerank)
    if a.index:
        from .store import is_index_dir

        if not is_index_dir(a.index):
            raise CliError(f"{a.index} is not an index directory; build one with "
                           f"`ethrag build-index --data DIR --out {a.index}`")
        return RagPipeline.from_index(a.index, llm, **kw)
    if not Path(a.data).is_dir():
        raise CliError(f"data directory {a.data} not found; pass --data DIR (markdown specs), "
                       "--index DIR (from `ethrag build-index`), or run scripts/fetch_corpus.py")
    return RagPipeline.from_directory(a.data, llm, **kw)


def cmd_ask(a: argparse.Namespace) -> int:
    ans = build_pipeline(a).ask(a.question)
    if a.explain:
        print(ans.text if ans.verified else f"(withheld) {ans.text}")
        print(ans.verification.explain() if ans.verification else "; ".join(ans.issues))
    elif a.json:
        print(json.dumps(ans.to_dict(), indent=2))
    else:
        print(json.dumps({"verified": ans.verified, "answer": ans.text, "retrieved": ans.retrieved,
                          "issues": ans.issues, "warnings": ans.warnings,
                          "provenance": ans.provenance}, indent=2))
    return 0 if ans.verified else 2


def cmd_build_index(a: argparse.Namespace) -> int:
    from .ingest import load_corpus
    from .retrieval import HybridIndex
    from .store import save_index

    if not Path(a.data).is_dir():
        raise CliError(f"data directory {a.data} not found; run scripts/fetch_corpus.py or pass --data")
    chunks = load_corpus(a.data)
    if not chunks:
        raise CliError(f"no markdown documents under {a.data}")
    index = HybridIndex(chunks)
    out = save_index(index, a.out)
    print(f"indexed {len(index.graph.docs)} docs / {len(chunks)} chunks -> {out}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    serve(build_pipeline(a), a.host, a.port)
    return 0


def cmd_mcp(a: argparse.Namespace) -> int:
    from .mcp_server import McpServer

    return McpServer(build_pipeline(a)).serve()


def cmd_related(a: argparse.Namespace) -> int:
    p = build_pipeline(a)
    doc = a.doc.lower()
    if doc not in p.index.graph.docs:
        print(f"unknown doc {doc!r}", file=sys.stderr)
        return 1
    rows = [{"doc": d, "distance": dist, "title": p.index.graph.docs[d].title}
            for d, dist in p.related(doc, a.depth)]
    print(json.dumps({"doc": doc, "related": rows}, indent=2))
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    from collections import Counter

    p = build_pipeline(a)
    chunks = list(p.chunks.values())
    print(json.dumps({
        "version": __version__, "docs": len(p.index.graph.docs), "chunks": len(chunks),
        "kinds": dict(Counter(c.kind for c in chunks)),
        "status": dict(Counter(m.status or "unknown" for m in p.index.graph.docs.values())),
        "graph": p.index.graph.stats(),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ethrag", description="citation-verified RAG over Ethereum specs")
    p.add_argument("--version", action="version", version=f"ethrag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="answer a question (exit code 2 if unverified)")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="full JSON including per-claim provenance")
    ask.add_argument("--explain", action="store_true",
                     help="print the per-claim verification table (token, supporting chunk, source line)")
    _add_source(ask)
    _add_pipeline_opts(ask)
    ask.set_defaults(fn=cmd_ask)

    bi = sub.add_parser("build-index", help="persist the index (npz + json) for fast startup")
    bi.add_argument("--data", default=os.environ.get("ETHRAG_DATA", str(DEFAULT_DATA)))
    bi.add_argument("--out", required=True)
    bi.set_defaults(fn=cmd_build_index)

    sv = sub.add_parser("serve", help="stdlib HTTP JSON API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    _add_source(sv)
    _add_pipeline_opts(sv)
    sv.set_defaults(fn=cmd_serve)

    mcp = sub.add_parser("mcp", help="MCP-style stdio JSON-RPC server (tools: ask_eips, ...)")
    _add_source(mcp)
    _add_pipeline_opts(mcp)
    mcp.set_defaults(fn=cmd_mcp)

    rel = sub.add_parser("related", help="walk the requires/citation graph from a doc id")
    rel.add_argument("doc")
    rel.add_argument("--depth", type=int, default=1)
    _add_source(rel)
    rel.set_defaults(fn=cmd_related)

    st = sub.add_parser("stats", help="corpus / graph statistics")
    _add_source(st)
    st.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backwards compatible: `ethrag "question"` behaves like `ethrag ask "question"`.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "ask")
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
