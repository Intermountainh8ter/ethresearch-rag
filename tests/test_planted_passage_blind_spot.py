import importlib.util
from pathlib import Path

from ethrag.ingest import Chunk

SPEC = importlib.util.spec_from_file_location(
    "run_planted_passage_blind_spot", Path(__file__).resolve().parents[1] / "evals" / "run_planted_passage_blind_spot.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _chunk():
    return Chunk(id="eip-1#000", doc="eip-1", title="T", section="S", text="X_MAX = 8 here")


def test_plant_replaces_value():
    p = mod.plant(_chunk(), "X_MAX = 8", "X_MAX is 8", "X_MAX is 16")
    assert p is not None and "X_MAX = 16" in p.text and p.id == "planted#000"


def test_plant_rejects_mismatch():
    assert mod.plant(_chunk(), "X_MAX = 8", "X_MAX is 8", "Y is 16") is None
    assert mod.plant(_chunk(), "X_MAX = 8", "X_MAX is 8", "X_MAX is 8") is None
    assert mod.plant(_chunk(), "X_MAX = 9", "X_MAX is 9", "X_MAX is 16") is None


def test_interval_empty():
    assert mod.interval(0, 0)["rate"] == 0.0
