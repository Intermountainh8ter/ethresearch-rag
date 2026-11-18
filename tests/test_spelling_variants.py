import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_spelling_variants", Path(__file__).resolve().parents[1] / "evals" / "run_spelling_variants.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_spellings_power_of_two_large():
    assert mod.spellings(4096) == {"exact": "4096", "hex": "0x1000", "pow2": "2**12", "comma": "4,096"}


def test_spellings_small_non_power():
    assert mod.spellings(3) == {"exact": "3", "hex": "0x3"}


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
