"""Verifier false-accept rate for denials worded outside its negation pattern.

    python3 evals/run_unlisted_negators.py --seed 7 --max-lines 200 --out results/427-unlisted-negators.json

``NAME = <decimal integer>`` lines are mined from every corpus chunk, sampled with the seed. The
control answer is ``NAME is <value>``; each variant denies the value ("differs from", "is other
than", "is anything but"). Default verifier, offline, simulated answers citing the unaltered
chunk. Release = verified; variant rates count only lines whose control was released.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.ingest import load_corpus
from ethrag.verify import verify_answer
from ethrag.verifyeval import wilson

LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*) = (\d+)$", re.MULTILINE)
MIN_N = 20
VARIANTS = {
    "differs_from": "{n} differs from {v}",
    "other_than": "{n} is other than {v}",
    "anything_but": "{n} is anything but {v}",
}


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def mine(chunks, seed: int, max_lines: int) -> list[tuple]:
    """Unique (name, value, chunk) triples, one per name-value pair, sampled by seed."""
    seen: set[tuple[str, str]] = set()
    found = []
    for c in chunks:
        for m in LINE.finditer(c.text):
            key = (m.group(1), m.group(2))
            if key not in seen:
                seen.add(key)
                found.append((m.group(1), m.group(2), c))
    random.Random(seed).shuffle(found)
    return found[:max_lines]


def release(text: str, chunk) -> bool:
    return bool(verify_answer(f"{text} [{chunk.id}]", {chunk.id: chunk}, [chunk.id]).ok)


def run(data: str, seed: int, max_lines: int) -> dict:
    lines = mine(list(load_corpus(data)), seed, max_lines)
    ctl = [0, 0]
    counts = {k: [0, 0] for k in VARIANTS}
    for name, val, chunk in lines:
        ok = release(f"{name} is {val}", chunk)
        ctl[0] += ok
        ctl[1] += 1
        if not ok:
            continue
        for key, tpl in VARIANTS.items():
            counts[key][0] += release(tpl.format(n=name, v=val), chunk)
            counts[key][1] += 1
    return {
        "lines_sampled": len(lines),
        "control_exact": interval(*ctl),
        "unlisted_negators": {
            k: {**interval(*c), "min_n": MIN_N, "evaluable": c[1] >= MIN_N}
            for k, c in counts.items()},
        "note": "constructed, simulated answers citing the unaltered chunk; NAME = <integer> lines mined from the whole corpus",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-lines", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    res = run(a.data, a.seed, a.max_lines)
    res["provenance"] = {
        "started_at": started, "seeds": [a.seed],
        "command": "python3 evals/run_unlisted_negators.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_exact", "unlisted_negators")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
