import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_unit_swap", Path(__file__).resolve().parents[1] / "evals" / "run_unit_swap.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_pick_unit_skips_present_words():
    assert mod.pick_unit("a value in bytes", 0) == "gwei"
    assert mod.pick_unit("nothing here", 1) == "gwei"


def test_pick_unit_none_when_all_present():
    assert mod.pick_unit("bytes gwei seconds slots", 7) is None


def test_pick_unit_word_boundary():
    assert mod.pick_unit("megabytes", 0) == "bytes"


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
