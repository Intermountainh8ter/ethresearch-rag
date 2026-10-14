"""End-to-end check of evals/run_evals.py on the mini corpus (seconds, no network)."""
import importlib.util
import json

import pytest

QUESTIONS = [
    ("t1", "train", "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "eip-1559", "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8"),
    ("t2", "train", "What is GAS_PER_BLOB for blob transactions?", "eip-4844", "GAS_PER_BLOB"),
    ("t3", "train", "What does a cold storage read cost?", "eip-2929", "COLD_SLOAD_COST"),
    ("t4", "train", "What is the access list address cost in EIP-2930?", "eip-2930", "ACCESS_LIST_ADDRESS_COST"),
    ("t5", "train", "How many slots are in an epoch?", "cl-phase0-beacon-chain", "SLOTS_PER_EPOCH"),
    ("e1", "test", "What is ELASTICITY_MULTIPLIER in EIP-1559?", "eip-1559", "ELASTICITY_MULTIPLIER = 2"),
    ("e2", "test", "What is MAX_BLOB_GAS_PER_BLOCK?", "eip-4844", "MAX_BLOB_GAS_PER_BLOCK"),
    ("e3", "test", "What is WARM_STORAGE_READ_COST?", "eip-2929", "WARM_STORAGE_READ_COST"),
]


@pytest.fixture(scope="module")
def script(root):
    spec = importlib.util.spec_from_file_location("run_evals_script", root / "evals" / "run_evals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qfile(tmp_path_factory):
    recs = [{"id": i, "type": "constant", "origin": "seed", "split": sp, "q": q, "gold_doc": d,
             "must_contain": n, "answer": "", "wrong": ""} for i, sp, q, d, n in QUESTIONS]
    recs.append({"id": "m1", "type": "multihop", "origin": "seed", "split": "test", "tier": "both-named",
                 "q": "What are GAS_PER_BLOB in EIP-4844 and COLD_SLOAD_COST in EIP-2929?",
                 "gold": [["eip-4844", "GAS_PER_BLOB"], ["eip-2929", "COLD_SLOAD_COST"]],
                 "answer": "", "wrong": ""})
    p = tmp_path_factory.mktemp("q") / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def run(script, mini_dir, qfile, tmp_path, *extra):
    out = tmp_path / "res.json"
    rc = script.main(["--data", str(mini_dir), "--questions", str(qfile), "--out", str(out),
                      "--reranker-out", str(tmp_path / "rr.json"), "--bootstrap", "50", *extra])
    return rc, out


def test_prints_train_and_test_tables_and_writes_results(script, mini_dir, qfile, tmp_path, capsys):
    rc, out = run(script, mini_dir, qfile, tmp_path)
    text = capsys.readouterr().out
    assert rc == 0
    for needle in ("retrieval, TRAIN split", "retrieval, TEST split", "fusion sweep", "ablations",
                   "doc-set recall", "paired bootstrap", "by tier"):
        assert needle in text
    res = json.loads(out.read_text())
    systems = [r["system"] for r in res["retrieval"]["test"]]
    assert systems == ["bm25", "dense(hashed)", "hybrid", "hybrid+rerank", "hybrid+graph"]
    assert {"R@1", "R@3", "R@5", "R@10", "MRR@10", "nDCG@10"} <= set(res["retrieval"]["train"][0])
    assert res["questions"] == {"total": 9, "train": 5, "test": 4, "multihop_train": 0, "multihop_test": 1}
    assert res["fusion"]["w"] in [i / 10 for i in range(11)]
    assert (tmp_path / "rr.json").is_file() and res["reranker"]["train_positives"] > 0


def test_gate_passes_below_and_fails_above_the_measured_test_recall(script, mini_dir, qfile, tmp_path, capsys):
    rc, out = run(script, mini_dir, qfile, tmp_path)
    r5 = json.loads(out.read_text())["gate"]["test_r5"]
    capsys.readouterr()
    rc, _ = run(script, mini_dir, qfile, tmp_path, "--min-test-recall5", str(max(0.0, r5 - 0.01)))
    assert rc == 0 and "gate ok" in capsys.readouterr().out
    if r5 < 1.0:
        rc, _ = run(script, mini_dir, qfile, tmp_path, "--min-test-hybrid-recall5", str(r5 + 0.01))
        assert rc == 1 and "GATE FAILED" in capsys.readouterr().out


def test_rejects_questions_without_gold_chunks_and_missing_splits(script, mini_dir, tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "type": "constant", "split": "train", "q": "q?", "gold_doc": "eip-1559",
                               "must_contain": "NOT THERE", "answer": "", "wrong": ""}) + "\n")
    assert script.main(["--data", str(mini_dir), "--questions", str(bad), "--out", str(tmp_path / "o.json")]) == 1
    assert "no gold chunk" in capsys.readouterr().out
    only_train = tmp_path / "t.jsonl"
    only_train.write_text(json.dumps({"id": "x", "type": "constant", "split": "train", "q": "q?", "gold_doc": "eip-1559",
                                      "must_contain": "ELASTICITY_MULTIPLIER", "answer": "", "wrong": ""}) + "\n")
    assert script.main(["--data", str(mini_dir), "--questions", str(only_train), "--out", str(tmp_path / "o.json")]) == 1
    assert "at least one train and one test" in capsys.readouterr().out


def test_can_run_from_a_saved_index_and_show_misses(script, mini_index, qfile, tmp_path, capsys):
    from ethrag.store import save_index

    save_index(mini_index, tmp_path / "ix")
    out = tmp_path / "r.json"
    rc = script.main(["--index", str(tmp_path / "ix"), "--questions", str(qfile), "--out", str(out),
                      "--reranker-out", str(tmp_path / "rr.json"), "--bootstrap", "0", "--show-misses"])
    assert rc == 0 and json.loads(out.read_text())["bootstrap_test"] == {}
    capsys.readouterr()
