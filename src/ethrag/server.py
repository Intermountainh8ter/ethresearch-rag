"""Tiny stdlib HTTP JSON API over a ``RagPipeline`` (no framework dependency).

    GET  /health                 -> {"ok": true, "docs": N, "chunks": M}
    GET  /ask?q=...&k=5          -> Answer.to_dict()
    POST /ask  {"question": "...", "k": 5}
    GET  /search?q=...&k=5       -> retrieved chunks only
    GET  /related?doc=eip-1559&depth=1
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .pipeline import RagPipeline

MAX_BODY = 1 << 20


class RagHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], pipeline: RagPipeline, quiet: bool = True):
        super().__init__(address, RagHandler)
        self.pipeline, self.quiet = pipeline, quiet


class RagHandler(BaseHTTPRequestHandler):
    server: RagHTTPServer

    def log_message(self, fmt, *args):  # silence default stderr logging unless verbose
        if not self.server.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _k(self, params: dict, default: int = 5) -> int:
        try:
            return max(1, min(50, int(params.get("k", [default])[0] if isinstance(params.get("k"), list)
                                      else params.get("k", default))))
        except (TypeError, ValueError):
            return default

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = parse_qs(url.query)
        p = self.server.pipeline
        if url.path == "/health":
            self._send(200, {"ok": True, "docs": len(p.index.graph.docs), "chunks": len(p.chunks)})
        elif url.path == "/ask":
            q = params.get("q", [""])[0].strip()
            if not q:
                return self._send(400, {"error": "missing q"})
            self._send(200, p.ask(q).to_dict())
        elif url.path == "/search":
            q = params.get("q", [""])[0].strip()
            if not q:
                return self._send(400, {"error": "missing q"})
            hits = p.index.search(q, self._k(params))
            self._send(200, {"question": q, "results": [
                {"id": c.id, "doc": c.doc, "title": c.title, "section": c.section,
                 "score": s, "text": c.body} for c, s in hits]})
        elif url.path == "/related":
            doc = params.get("doc", [""])[0].strip()
            if doc not in p.index.graph.docs:
                return self._send(404, {"error": f"unknown doc {doc!r}"})
            depth = self._k({"k": params.get("depth", ["1"])}, 1)
            self._send(200, {"doc": doc, "related": [
                {"doc": d, "distance": dist, "title": p.index.graph.docs[d].title}
                for d, dist in p.related(doc, depth)]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/ask":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_BODY:
            return self._send(400, {"error": "body required (max 1 MiB)"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._send(400, {"error": "invalid JSON"})
        q = str(payload.get("question", "")).strip() if isinstance(payload, dict) else ""
        if not q:
            return self._send(400, {"error": "missing question"})
        p = self.server.pipeline
        k = self._k(payload, p.k)
        old_k, p.k = p.k, k
        try:
            self._send(200, p.ask(q).to_dict())
        finally:
            p.k = old_k


def make_server(pipeline: RagPipeline, host: str = "127.0.0.1", port: int = 0,
                quiet: bool = True) -> RagHTTPServer:
    """Bind (port 0 = random free port) without serving; call ``serve_forever`` yourself."""
    return RagHTTPServer((host, port), pipeline, quiet)


def serve(pipeline: RagPipeline, host: str = "127.0.0.1", port: int = 8080) -> None:
    srv = make_server(pipeline, host, port, quiet=False)
    print(f"ethrag serving on http://{host}:{srv.server_port}  (GET /health, /ask?q=..., POST /ask)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
