import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

SPEC = importlib.util.spec_from_file_location(
    "run_parametric", Path(__file__).resolve().parents[1] / "evals" / "run_parametric.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def ch(i, doc, text):
    return NS(id=i, doc=doc, text=text)


def test_mine_picks_other_chunk_without_line():
    a, b = ch("d#1", "d", "intro\nA = 1"), ch("d#2", "d", "other text")
    out = mod.mine([a, b])
    assert [(x[1], x[2], x[3].id) for x in out] == [("A", "1", "d#2")]


def test_mine_skips_chunks_holding_line_and_other_docs():
    a, b = ch("d#1", "d", "A = 1"), ch("d#2", "d", "A = 1")
    c = ch("e#1", "e", "zzz")
    assert mod.mine([a, b, c]) == []


def test_mine_dedupes_per_doc():
    a, b, c = ch("d#1", "d", "A = 1"), ch("d#2", "d", "A = 1"), ch("d#3", "d", "x")
    assert len(mod.mine([a, b, c])) == 1


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
