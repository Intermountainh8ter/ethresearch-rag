"""Minimal MCP-style stdio JSON-RPC 2.0 server exposing the pipeline as tools.

Implements the handshake (``initialize`` / ``notifications/initialized``), ``ping``,
``tools/list`` and ``tools/call`` over newline-delimited JSON on stdin/stdout, with no
third-party dependency. Tools: ``ask_eips`` (verified answer), ``search_eips`` (retrieval
only) and ``related_eips`` (citation-graph neighbours).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import IO, Any

from . import __version__
from .pipeline import RagPipeline

PROTOCOL_VERSION = "2024-11-05"
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700, -32600, -32601, -32602, -32603)

TOOLS = [
    {
        "name": "ask_eips",
        "description": "Answer a question about Ethereum EIPs/ERCs/consensus specs with a "
                       "citation-verified answer. Returns verified=false and the issues when the "
                       "draft could not be grounded in the retrieved text.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"},
                           "k": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["question"],
        },
    },
    {
        "name": "search_eips",
        "description": "Retrieve the top-k spec chunks for a query (hybrid BM25 + dense).",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"},
                           "k": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["query"],
        },
    },
    {
        "name": "related_eips",
        "description": "Documents linked to a doc id (e.g. eip-1559) through the requires graph.",
        "inputSchema": {
            "type": "object",
            "properties": {"doc": {"type": "string"},
                           "depth": {"type": "integer", "minimum": 1, "maximum": 3}},
            "required": ["doc"],
        },
    },
]


class McpServer:
    def __init__(self, pipeline: RagPipeline, name: str = "ethrag"):
        self.pipeline, self.name = pipeline, name
        self.initialized = False
        self._methods: dict[str, Callable[[dict], Any]] = {
            "initialize": self._initialize,
            "ping": lambda params: {},
            "tools/list": lambda params: {"tools": TOOLS},
            "tools/call": self._tools_call,
        }

    # -- protocol --------------------------------------------------------------------------
    def _initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": __version__},
        }

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise McpError(INVALID_PARAMS, "arguments must be an object")
        tool = {t["name"]: t for t in TOOLS}.get(name)
        if tool is None:
            raise McpError(INVALID_PARAMS, f"unknown tool {name!r}")
        for req in tool["inputSchema"]["required"]:
            if not isinstance(args.get(req), str) or not args[req].strip():
                raise McpError(INVALID_PARAMS, f"missing required argument {req!r}")
        payload = getattr(self, f"_tool_{name}")(args)
        return {"content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": bool(payload.get("error"))}

    # -- tools -----------------------------------------------------------------------------
    def _bounded(self, value: Any, default: int, lo: int = 1, hi: int = 50) -> int:
        try:
            return max(lo, min(hi, int(value)))
        except (TypeError, ValueError):
            return default

    def _tool_ask_eips(self, args: dict) -> dict:
        p = self.pipeline
        old_k, p.k = p.k, self._bounded(args.get("k"), p.k)
        try:
            a = p.ask(args["question"])
        finally:
            p.k = old_k
        return {"verified": a.verified, "answer": a.text, "retrieved": a.retrieved,
                "issues": a.issues, "warnings": a.warnings, "provenance": a.provenance}

    def _tool_search_eips(self, args: dict) -> dict:
        hits = self.pipeline.index.search(args["query"], self._bounded(args.get("k"), 5))
        return {"results": [{"id": c.id, "doc": c.doc, "title": c.title, "section": c.section,
                             "score": s, "text": c.body} for c, s in hits]}

    def _tool_related_eips(self, args: dict) -> dict:
        doc = args["doc"].strip().lower()
        graph = self.pipeline.index.graph
        if doc not in graph.docs:
            return {"error": f"unknown doc {doc!r}", "related": []}
        depth = self._bounded(args.get("depth"), 1, 1, 3)
        return {"doc": doc, "related": [{"doc": d, "distance": dist, "title": graph.docs[d].title}
                                        for d, dist in graph.related(doc, depth)]}

    # -- dispatch --------------------------------------------------------------------------
    def handle(self, msg: Any) -> dict | None:
        """Handle one decoded JSON-RPC message; ``None`` means no response (a notification)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST,
                          "invalid JSON-RPC 2.0 request")
        method, params, rid = msg["method"], msg.get("params") or {}, msg.get("id")
        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self.initialized = True
            return None
        if not isinstance(params, dict):
            return _error(rid, INVALID_PARAMS, "params must be an object")
        fn = self._methods.get(method)
        if fn is None:
            return _error(rid, METHOD_NOT_FOUND, f"method not found: {method}")
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": fn(params)}
        except McpError as e:
            return _error(rid, e.code, e.message)
        except Exception as e:  # noqa: BLE001 - surface as JSON-RPC internal error
            return _error(rid, INTERNAL_ERROR, f"{type(e).__name__}: {e}")

    def handle_line(self, line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except ValueError as e:
            return _error(None, PARSE_ERROR, f"parse error: {e}")
        return self.handle(msg)

    def serve(self, inp: IO[str] | None = None, out: IO[str] | None = None) -> int:
        inp, out = inp or sys.stdin, out or sys.stdout
        for line in inp:
            resp = self.handle_line(line)
            if resp is not None:
                out.write(json.dumps(resp) + "\n")
                out.flush()
        return 0


class McpError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _error(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
