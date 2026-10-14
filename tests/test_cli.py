import io
import json

import pytest

from ethrag import cli
from ethrag.cli import CliError, main


@pytest.fixture(scope="module")
def built_index(mini_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cli-index") / "idx"
    assert main(["build-index", "--data", str(mini_dir), "--out", str(out)]) == 0
    return out


def run(capsys, *argv):
    rc = main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_build_index_reports_counts(mini_dir, tmp_path, capsys):
    rc, out, _ = run(capsys, "build-index", "--data", str(mini_dir), "--out", str(tmp_path / "i"))
    assert rc == 0 and out.startswith("indexed 7 docs / ") and (tmp_path / "i" / "arrays.npz").is_file()


def test_build_index_errors(tmp_path, capsys):
    rc, _, err = run(capsys, "build-index", "--data", str(tmp_path / "missing"), "--out", str(tmp_path / "o"))
    assert rc == 1 and "not found" in err
    (tmp_path / "empty").mkdir()
    rc, _, err = run(capsys, "build-index", "--data", str(tmp_path / "empty"), "--out", str(tmp_path / "o"))
    assert rc == 1 and "no markdown documents" in err


def test_ask_from_index_prints_json_and_sets_exit_code(built_index, capsys):
    rc, out, _ = run(capsys, "ask", "--index", str(built_index), "What does EIP-2718 say about the transaction type byte?")
    body = json.loads(out)
    assert set(body) == {"verified", "answer", "retrieved", "issues", "warnings", "provenance"}
    assert rc == (0 if body["verified"] else 2)
    rc, out, _ = run(capsys, "ask", "--index", str(built_index), "--json", "-k", "3",
                     "What is the airspeed velocity of an unladen swallow?")
    full = json.loads(out)
    assert rc == 2 and full["verified"] is False and "verification" in full and len(full["retrieved"]) <= 3


def test_legacy_positional_question_form(built_index, capsys, monkeypatch):
    monkeypatch.setenv("ETHRAG_INDEX", str(built_index))
    cap_rc, out, _ = run(capsys, "What is the transaction type byte range in EIP-2718?")
    assert json.loads(out)["retrieved"] and cap_rc in (0, 2)


def test_ask_from_data_dir_and_options(mini_dir, capsys, tmp_path):
    from ethrag.rerank import LogisticReranker

    w = tmp_path / "w.json"
    LogisticReranker().save(w)
    rc, out, _ = run(capsys, "ask", "--data", str(mini_dir), "--rerank", str(w), "--expand-graph",
                     "--no-negation-check", "--min-coverage", "0.1", "--min-relevance", "0.0",
                     "What is GAS_PER_BLOB in EIP-4844?")
    assert json.loads(out)["retrieved"] and rc in (0, 2)


def test_related_and_stats(built_index, capsys):
    rc, out, _ = run(capsys, "related", "EIP-1559", "--index", str(built_index), "--depth", "2")
    docs = {r["doc"] for r in json.loads(out)["related"]}
    assert rc == 0 and {"eip-2718", "eip-2929"} <= docs
    rc, _, err = run(capsys, "related", "eip-0", "--index", str(built_index))
    assert rc == 1 and "unknown doc" in err
    rc, out, _ = run(capsys, "stats", "--index", str(built_index))
    stats = json.loads(out)
    assert rc == 0 and stats["docs"] == 7 and stats["graph"]["requires_edges"] == 7
    assert set(stats["kinds"]) <= {"prose", "code", "table"} and stats["status"]["Final"] == 6


def test_missing_sources_give_friendly_errors(tmp_path, capsys):
    rc, _, err = run(capsys, "ask", "--index", str(tmp_path / "nope"), "q")
    assert rc == 1 and err.startswith("error: ") and "build-index" in err
    rc, _, err = run(capsys, "ask", "--data", str(tmp_path / "nope"), "q")
    assert rc == 1 and "--data DIR" in err


def test_llm_flag_without_key_or_sdk(mini_dir, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc, _, err = run(capsys, "ask", "--data", str(mini_dir), "--llm", "q")
    assert rc == 1 and "ANTHROPIC_API_KEY" in err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)  # import raises ImportError
    rc, _, err = run(capsys, "ask", "--data", str(mini_dir), "--llm", "q")
    assert rc == 1 and "ethrag[llm]" in err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "ethrag " in capsys.readouterr().out


def test_serve_command_delegates_to_server(built_index, monkeypatch):
    seen = {}
    import ethrag.server

    monkeypatch.setattr(ethrag.server, "serve", lambda pipe, host, port: seen.update(
        chunks=len(pipe.chunks), host=host, port=port))
    assert main(["serve", "--index", str(built_index), "--host", "127.0.0.1", "--port", "0"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 0 and seen["chunks"] > 0


def test_mcp_command_reads_stdin(built_index, monkeypatch, capsys):
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    monkeypatch.setattr("sys.stdin", io.StringIO(msg + "\n"))
    assert main(["mcp", "--index", str(built_index)]) == 0
    assert json.loads(capsys.readouterr().out) == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_cli_error_type_is_exported():
    assert issubclass(CliError, Exception) and cli.SUBCOMMANDS[0] == "ask"


def test_ask_explain_prints_the_per_claim_table(built_index, capsys):
    rc, out, _ = run(capsys, "ask", "--index", str(built_index), "--explain",
                     "What is the ELASTICITY_MULTIPLIER in EIP-1559?")
    assert rc == 0
    assert "ELASTICITY_MULTIPLIER is 2" in out and "verdict: VERIFIED" in out and "[ok  ]" in out
    assert "elasticity_multiplier" in out and "(exact)" in out
    rc, out, _ = run(capsys, "ask", "--index", str(built_index), "--explain",
                     "What is the airspeed velocity of an unladen swallow?")
    assert rc == 2 and out.startswith("(withheld) ") and "model reported insufficient context" in out


def test_ask_explain_for_a_withheld_draft(mini_chunks, mini_index, repeat_llm, cid, monkeypatch, capsys):
    from ethrag import RagPipeline

    wrong = f"ELASTICITY_MULTIPLIER is 3 [{cid('eip-1559', 'ELASTICITY_MULTIPLIER')}]."
    monkeypatch.setattr(cli, "build_pipeline", lambda a: RagPipeline(mini_chunks, repeat_llm(wrong), index=mini_index))
    rc, out, _ = run(capsys, "ask", "--explain", "What is ELASTICITY_MULTIPLIER in EIP-1559?")
    assert rc == 2 and out.startswith("(withheld) ") and "verdict: WITHHELD" in out
    assert "[FAIL] s0 3" in out and "issue: unsupported claim '3'" in out


def test_default_verifier_thresholds_are_the_ones_chosen_on_train():
    from ethrag.cli import build_parser

    a = build_parser().parse_args(["ask", "q"])
    assert (a.min_coverage, a.min_relevance, a.min_novelty) == (0.7, 0.1, 0.0)
