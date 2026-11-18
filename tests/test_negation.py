import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_negation", Path(__file__).resolve().parents[1] / "evals" / "run_negation.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0


def test_interval_counts():
    r = mod.interval(3, 4)
    assert r["k"] == 3 and r["n"] == 4 and r["rate"] == 0.75


def test_line_matches_integer_constants_only():
    assert mod.LINE.match("MAX = 8")
    assert not mod.LINE.match("MAX = 0x08")
