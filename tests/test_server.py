import http.client
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from ethrag import RagPipeline
from ethrag.server import MAX_BODY, make_server, serve


@pytest.fixture()
def api(mini_chunks, mini_index, repeat_llm, cid):
    # A scripted generator keeps the answers deterministic; the verifier is the real one.
    reply = f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [{cid('eip-1559', 'BASE_FEE_MAX')}]."
    pipe = RagPipeline(mini_chunks, repeat_llm(reply), index=mini_index, k=5)
    srv = make_server(pipe, "127.0.0.1", 0)  # port 0 = a random free port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield base, srv
    srv.shutdown()
    srv.server_close()
    t.join(timeout=5)


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(base, path, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def q(s):
    return urllib.parse.quote(s)


def test_random_port_is_bound(api):
    base, srv = api
    assert srv.server_port > 0 and base.endswith(str(srv.server_port))


def test_health(api):
    status, body = get(api[0], "/health")
    assert status == 200 and body == {"ok": True, "docs": 7, "chunks": len(api[1].pipeline.chunks)}


def test_ask_get_returns_a_verified_answer_with_provenance(api):
    status, body = get(api[0], "/ask?q=" + q("What is the BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"))
    assert status == 200 and body["verified"] is True
    assert "BASE_FEE_MAX_CHANGE_DENOMINATOR" in body["answer"]
    assert body["provenance"] and body["retrieved"]
    assert body["verification"]["ok"] is True and body["verification"]["claims"]


def test_ask_get_honours_k_and_clamps_garbage(api):
    # k=2 / k=garbage change retrieval only; the draft stays fixed and may be withheld
    base = api[0]
    s, b = get(base, "/ask?q=" + q("blob transactions") + "&k=2")
    assert s == 200 and len(b["retrieved"]) <= 2
    s, b = get(base, "/ask?q=" + q("blob transactions") + "&k=notanumber")
    assert s == 200 and len(b["retrieved"]) <= 5


def test_ask_requires_q(api):
    assert get(api[0], "/ask") == (400, {"error": "missing q"})
    assert get(api[0], "/ask?q=%20%20")[0] == 400


def test_ask_post_json(api):
    s, b = post(api[0], "/ask", {"question": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "k": 3})
    assert s == 200 and len(b["retrieved"]) <= 3


def test_unverifiable_draft_is_withheld_not_errored(api):
    # The scripted draft cites an eip-1559 chunk that a Flashbots-style question never retrieves.
    s, b = get(api[0], "/ask?q=" + q("What is the airspeed velocity of an unladen swallow?"))
    assert s == 200 and b["verified"] is False and b["issues"]


def test_post_validation_errors(api):
    base = api[0]
    assert post(base, "/ask", None, raw=b"{not json") == (400, {"error": "invalid JSON"})
    assert post(base, "/ask", {"question": "  "}) == (400, {"error": "missing question"})
    assert post(base, "/ask", [1, 2]) == (400, {"error": "missing question"})
    assert post(base, "/nope", {"question": "x"}) == (404, {"error": "not found"})
    assert post(base, "/ask", None, raw=b"")[0] == 400  # empty body


def test_post_rejects_oversized_and_malformed_content_length(api):
    host, port = api[0].removeprefix("http://").split(":")
    for header, expected in ((str(MAX_BODY + 1), "body required"), ("abc", "bad Content-Length")):
        conn = http.client.HTTPConnection(host, int(port), timeout=10)
        conn.putrequest("POST", "/ask")
        conn.putheader("Content-Length", header)
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400 and expected in json.loads(resp.read())["error"]
        conn.close()


def test_search_endpoint(api):
    s, b = get(api[0], "/search?q=" + q("GAS_PER_BLOB blob") + "&k=3")
    assert s == 200 and 1 <= len(b["results"]) <= 3
    assert b["results"][0]["doc"] == "eip-4844" and {"id", "score", "text", "section"} <= set(b["results"][0])
    assert get(api[0], "/search")[0] == 400


def test_related_endpoint(api):
    s, b = get(api[0], "/related?doc=EIP-1559&depth=1")
    assert s == 200 and {r["doc"] for r in b["related"]} == {"eip-2718", "eip-2930", "eip-4844"}
    s, b = get(api[0], "/related?doc=eip-4844&depth=2")
    assert {"eip-2929"} <= {r["doc"] for r in b["related"]}
    assert get(api[0], "/related?doc=eip-0")[0] == 404


def test_unknown_path_is_json_404(api):
    assert get(api[0], "/nope") == (404, {"error": "not found"})


def test_concurrent_requests_with_different_k_do_not_interfere(api):
    base, srv = api
    results, errors = {}, []

    def work(k):
        try:
            s, b = post(base, "/ask", {"question": "blob transactions kzg commitments", "k": k})
            results[k] = (s, len(b["retrieved"]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in (1, 2, 3, 4) * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors
    assert all(s == 200 and n <= k for k, (s, n) in results.items())
    assert srv.pipeline.k == 5  # the shared pipeline was never mutated


def test_handler_exceptions_become_json_500(api):
    base, srv = api

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    srv.pipeline.ask = boom
    s, b = get(base, "/ask?q=hello")
    assert s == 500 and "kaboom" in b["error"]
    s, b = post(base, "/ask", {"question": "hello"})
    assert s == 500 and "kaboom" in b["error"]


def test_serve_prints_banner_and_closes_on_keyboard_interrupt(mini_chunks, mini_index, monkeypatch, capsys):
    pipe = RagPipeline(mini_chunks, None, index=mini_index)
    import ethrag.server as server_mod

    def interrupt(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(server_mod.RagHTTPServer, "serve_forever", interrupt)
    serve(pipe, "127.0.0.1", 0)
    assert "ethrag serving on http://127.0.0.1:" in capsys.readouterr().out


def test_verbose_server_logs_requests_to_stderr(mini_chunks, mini_index, capsys):
    srv = make_server(RagPipeline(mini_chunks, None, index=mini_index), "127.0.0.1", 0, quiet=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert get(f"http://127.0.0.1:{srv.server_port}", "/health")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)
    assert "GET /health" in capsys.readouterr().err
