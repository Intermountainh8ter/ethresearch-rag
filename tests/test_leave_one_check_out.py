import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_leave_one_check_out", Path(__file__).resolve().parents[1] / "evals" / "run_leave_one_check_out.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_issue_type_buckets():
    assert mod.issue_type("unbound claim '8' found in the cited chunks") == "binding"
    assert mod.issue_type("unsupported claim '16' not found") == "unsupported_claim"
    assert mod.issue_type("range mismatch: answer says 1 to 2") == "range"
    assert mod.issue_type("sentence 0 coverage 0.10 < 0.70: x") == "coverage"
    assert mod.issue_type("mystery") == "other"


def test_ablate_counts_only_answers_blocked_solely_by_check():
    sets = [{"unit"}, {"unit", "range"}, set(), {"binding"}]
    assert mod.ablate(sets, None) == 3
    assert mod.ablate(sets, "unit") == 2
    assert mod.ablate(sets, "range") == 3


def test_arms_cover_known_types():
    known = {n for _, n in mod.PREFIXES} | {"coverage", "other"}
    for checks in mod.ARMS.values():
        assert set(checks) <= known
