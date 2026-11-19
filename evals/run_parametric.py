"""Verifier release rate on true answers whose cited chunk does not hold the line (parametric).

    python3 evals/run_parametric.py --seed 7 --out results/542-parametric.json

Every ``NAME = <decimal integer>`` line is mined from corpus chunks (one per document, name and
value). The control answer ``NAME is <value>`` cites the chunk holding the line; the parametric
answer cites the first other chunk of the same document that does not contain the line. Default
verifier, offline, simulated answers. Release = verified. Only lines whose control is released
count for the parametric rate.
"""
from __future__ import annotations

import argparse
import json
import platform
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

LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*) = (\d+)$")
MIN_N = 20


def mine(chunks: list) -> list[tuple]:
    """(chunk, name, value, other_chunk) per unique (doc, name, value); other has no such line."""
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    seen: set[tuple] = set()
    out = []
    for c in chunks:
        for ln in c.text.splitlines():
            m = LINE.match(ln.strip())
            if not m or (c.doc, m.group(1), m.group(2)) in seen:
                continue
            seen.add((c.doc, m.group(1), m.group(2)))
            other = next((o for o in by_doc[c.doc]
                          if o.id != c.id and not any(x.strip() == ln.strip() for x in o.text.splitlines())), None)
            if other is not None:
                out.append((c, m.group(1), m.group(2), other))
    return out


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, seed: int) -> dict:
    items = mine(list(load_corpus(data)))
    ctl = [0, 0]
    par = [0, 0]
    par_noname = [0, 0]
    for gold, name, val, other in items:
        q = f"What is {name}?"
        r0 = verify_answer(f"{name} is {val} [{gold.id}]", {gold.id: gold}, [gold.id], question=q)
        ctl[0] += bool(r0.ok)
        ctl[1] += 1
        if not r0.ok:
            continue
        r1 = verify_answer(f"{name} is {val} [{other.id}]", {other.id: other}, [other.id], question=q)
        par[0] += bool(r1.ok)
        par[1] += 1
        if name not in other.text:
            par_noname[0] += bool(r1.ok)
            par_noname[1] += 1
    return {
        "lines_mined": len(items),
        "control_supported": interval(*ctl),
        "parametric": {"false_accept_rate": interval(*par)["rate"], **interval(*par),
                       "min_n": MIN_N, "evaluable": par[1] >= MIN_N},
        "parametric_name_absent": interval(*par_noname),
        "note": "constructed, simulated true answers; cited chunk is another chunk of the same document without the line",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    res = run(a.data, a.seed)
    res["provenance"] = {
        "started_at": started, "seeds": [a.seed],
        "command": "python3 evals/run_parametric.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_supported", "parametric")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
