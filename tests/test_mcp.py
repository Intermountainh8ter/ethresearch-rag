import io
import json
import subprocess
import sys

import pytest

from ethrag import RagPipeline
from ethrag.mcp_server import PROTOCOL_VERSION, SUPPORTED_VERSIONS, TOOLS, McpServer


@pytest.fixture()
def server(mini_chunks, mini_index, repeat_llm, cid):
    reply = f"ELASTICITY_MULTIPLIER is 2 [{cid('eip-1559', 'ELASTICITY_MULTIPLIER')}]."
    return McpServer(RagPipeline(mini_chunks, repeat_llm(reply), index=mini_index, k=5))


def rpc(method, params=None, id=1):
    m = {"jsonrpc": "2.0", "method": method}
    if id is not None:
        m["id"] = id
    if params is not None:
        m["params"] = params
    return m


def call(server, name, args, id=1):
    return server.handle(rpc("tools/call", {"name": name, "arguments": args}, id))


def payload(resp):
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize_handshake_sequence(server):
    r = server.handle(rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                         "clientInfo": {"name": "t", "version": "0"}}))
    res = r["result"]
    assert r["id"] == 1 and res["protocolVersion"] == "2025-03-26"
    assert res["serverInfo"]["name"] == "ethrag" and "tools" in res["capabilities"]
    assert not server.initialized
    assert server.handle(rpc("notifications/initialized", id=None)) is None
    assert server.initialized


def test_unknown_protocol_version_gets_servers_latest(server):
    for offered in ("1999-01-01", None, ""):
        params = {"protocolVersion": offered} if offered is not None else {}
        assert server.handle(rpc("initialize", params))["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert PROTOCOL_VERSION == SUPPORTED_VERSIONS[0]


def test_ping_and_tools_list(server):
    assert server.handle(rpc("ping"))["result"] == {}
    tools = server.handle(rpc("tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == ["ask_eips", "search_eips", "related_eips"] == [t["name"] for t in TOOLS]
    assert all(t["inputSchema"]["type"] == "object" and t["inputSchema"]["required"] for t in tools)


def test_ask_tool_returns_verified_answer(server):
    resp = call(server, "ask_eips", {"question": "What is ELASTICITY_MULTIPLIER in EIP-1559?", "k": 4})
    p = payload(resp)
    assert resp["result"]["isError"] is False and p["verified"] is True
    assert "ELASTICITY_MULTIPLIER" in p["answer"] and p["provenance"] and len(p["retrieved"]) <= 4


def test_ask_tool_reports_unverified_with_issues(server):
    p = payload(call(server, "ask_eips", {"question": "What is the airspeed velocity of an unladen swallow?"}))
    assert p["verified"] is False and p["issues"]


def test_search_and_related_tools(server):
    p = payload(call(server, "search_eips", {"query": "GAS_PER_BLOB", "k": 2}))
    assert p["results"][0]["doc"] == "eip-4844" and len(p["results"]) <= 2
    p = payload(call(server, "related_eips", {"doc": "EIP-1559"}))
    assert {r["doc"] for r in p["related"]} == {"eip-2718", "eip-2930", "eip-4844"}
    p = payload(call(server, "related_eips", {"doc": "eip-4844", "depth": 99}))
    assert "eip-2929" in {r["doc"] for r in p["related"]}  # depth clamped to 3, not rejected


def test_unknown_doc_is_a_tool_error_not_a_protocol_error(server):
    resp = call(server, "related_eips", {"doc": "eip-0"})
    assert resp["result"]["isError"] is True and "unknown doc" in payload(resp)["error"]


def test_protocol_errors(server):
    assert server.handle(rpc("nope"))["error"]["code"] == -32601
    assert call(server, "nope", {})["error"]["code"] == -32602
    assert call(server, "ask_eips", {})["error"]["code"] == -32602
    assert call(server, "ask_eips", {"question": "   "})["error"]["code"] == -32602
    assert call(server, "ask_eips", [1])["error"]["code"] == -32602
    assert server.handle({"jsonrpc": "2.0", "id": 5, "method": "ping", "params": [1]})["error"]["code"] == -32602
    for bad in (["batch"], "str", {"id": 3, "method": "ping"}, {"jsonrpc": "2.0", "id": 3}):
        assert server.handle(bad)["error"]["code"] == -32600


def test_request_without_id_is_a_notification_and_gets_no_reply(server):
    assert server.handle(rpc("ping", id=None)) is None
    assert server.handle(rpc("notifications/cancelled", {"requestId": 1}, id=None)) is None


def test_internal_errors_are_reported_not_raised(server):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    server.pipeline.ask = boom
    err = call(server, "ask_eips", {"question": "blob"})["error"]
    assert err["code"] == -32603 and "kaboom" in err["message"]


def test_handle_line_parse_errors_and_blank_lines(server):
    assert server.handle_line("   ") is None
    assert server.handle_line("{oops")["error"]["code"] == -32700
    assert server.handle_line(json.dumps(rpc("ping")))["result"] == {}


def test_serve_over_in_memory_pipes(server):
    lines = [rpc("initialize", {"protocolVersion": PROTOCOL_VERSION}), rpc("notifications/initialized", id=None),
             rpc("tools/list", id=2), "", "{bad json", rpc("ping", id=3)]
    inp = io.StringIO("\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n")
    out = io.StringIO()
    assert server.serve(inp, out) == 0
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, None, 3]
    assert replies[2]["error"]["code"] == -32700


def test_real_subprocess_handshake_over_stdio(mini_dir):
    """Drive `ethrag mcp` through real OS pipes exactly as an MCP client would."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "ethrag.cli", "mcp", "--data", str(mini_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        def send(msg):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        def recv():
            return json.loads(proc.stdout.readline())

        send(rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                "clientInfo": {"name": "pytest", "version": "0"}}))
        assert recv()["result"]["serverInfo"]["name"] == "ethrag"
        send(rpc("notifications/initialized", id=None))
        send(rpc("tools/list", id=2))
        assert len(recv()["result"]["tools"]) == 3
        send(rpc("tools/call", {"name": "ask_eips", "arguments": {
            "question": "What is the BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"}}, id=3))
        reply = recv()
        body = json.loads(reply["result"]["content"][0]["text"])
        assert reply["id"] == 3 and body["retrieved"] and "verified" in body
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
