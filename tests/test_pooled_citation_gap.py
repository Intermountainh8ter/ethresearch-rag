import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_pooled_citation_gap", Path(__file__).resolve().parents[1] / "evals" / "run_pooled_citation_gap.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_strip_line_removes_value_line_only():
    assert mod.strip_line("A = 8\nB = 16", "A = 8") == "B = 16"


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
