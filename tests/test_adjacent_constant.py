import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_adjacent_constant", Path(__file__).resolve().parents[1] / "evals" / "run_adjacent_constant.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_pick_other_first_in_file_order():
    t = "A = 1\nB = 2\nC = 3"
    assert mod.pick_other(t, "A", "1") == ("B", "2")


def test_pick_other_skips_same_value_and_name():
    assert mod.pick_other("A = 1\nB = 1\nC = 5", "A", "1") == ("C", "5")


def test_pick_other_skips_value_next_to_name():
    t = "A = 1\nA is also 2 here\nB = 2\nC = 9"
    assert mod.pick_other(t, "A", "1") == ("C", "9")


def test_pick_other_none():
    assert mod.pick_other("A = 1", "A", "1") is None


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
