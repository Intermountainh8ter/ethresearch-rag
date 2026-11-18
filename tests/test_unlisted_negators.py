import importlib.util
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "run_unlisted_negators", Path(__file__).resolve().parents[1] / "evals" / "run_unlisted_negators.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0


def test_mine_dedups_and_caps():
    c = SimpleNamespace(id="a#1", text="A = 1\nA = 1\nB = 2\nC = 0x3")
    out = mod.mine([c], 7, 10)
    assert sorted((n, v) for n, v, _ in out) == [("A", "1"), ("B", "2")]
    assert len(mod.mine([c], 7, 1)) == 1


def test_mine_is_seeded():
    c = SimpleNamespace(id="a#1", text="\n".join(f"N{i} = {i}" for i in range(30)))
    assert mod.mine([c], 3, 5) == mod.mine([c], 3, 5)


def test_variants_mention_name_and_value():
    for tpl in mod.VARIANTS.values():
        s = tpl.format(n="X", v="9")
        assert "X" in s and "9" in s
