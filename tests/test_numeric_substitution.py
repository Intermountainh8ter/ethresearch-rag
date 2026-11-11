import importlib.util
import random
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_numeric_substitution", Path(__file__).resolve().parents[1] / "evals" / "run_numeric_substitution.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_other_numbers_excludes_value_and_dedups():
    assert mod.other_numbers("A = 8, B 16, C 16, D 3.5, v8", "8") == ["16"]


def test_other_numbers_empty():
    assert mod.other_numbers("A = 8", "8") == []


def test_absent_number_not_in_text():
    assert mod.absent_number("12345", random.Random(1)) != "12345"


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
