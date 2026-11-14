import importlib.util
import random
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_stale_version_blind_spot", Path(__file__).resolve().parents[1] / "evals" / "run_stale_version_blind_spot.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_make_stale_keeps_value_line_and_shifts_others():
    out = mod.make_stale("A = 8\nB = 16", "A = 8", random.Random(1))
    lines = out.split("\n")
    assert lines[0] == "Status: Superseded"
    assert "A = 8" in lines
    assert "B = 16" not in lines


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
