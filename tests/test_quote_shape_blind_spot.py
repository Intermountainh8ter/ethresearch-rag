import importlib.util
import random
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_quote_shape_blind_spot", Path(__file__).resolve().parents[1] / "evals" / "run_quote_shape_blind_spot.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_truncate_cuts_before_value():
    assert mod.truncate("BASE_FEE = 8 here", "8") == "BASE_FEE"


def test_ellipsis_join():
    assert mod.ellipsis_join("a b ", " c = 1") == "a b ... c = 1"


def test_reorder_same_words_different_order():
    out = mod.reorder("one two three four", random.Random(1))
    assert sorted(out.split()) == ["four", "one", "three", "two"]
    assert out != "one two three four"


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
