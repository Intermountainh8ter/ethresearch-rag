#!/usr/bin/env python3
"""Reproducibly (re)build the bundled corpus from raw.githubusercontent.com.

Sources (all CC0 / permissively licensed, see data/LICENSE):

* ethereum/EIPs      -> data/eips/eip-N.md
* ethereum/ERCs      -> data/eips/erc-N.md   (canonical home of ERCs; wins over an EIPs-repo copy)
* ethereum/consensus-specs -> data/consensus-specs/cl-<fork>-<file>.md
* flashbots/flashbots-docs -> data/flashbots/fb-<path>.md  (a curated list of pages; .mdx kept raw)

Usage:
    python scripts/fetch_corpus.py [--out data] [--workers 8] [--numbers 1559,4844] [--dry-run]

Rules: keep only HTTP 200 bodies larger than ``MIN_BYTES`` whose front matter is not ``status:
Moved``; 404s are expected for most of the number list. Every kept file is recorded in
``data/MANIFEST.json`` with its sha256 so the corpus can be audited offline. Only the standard
library is used so the script runs anywhere Python does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

RAW = "https://raw.githubusercontent.com"
MIN_BYTES = 300
TIMEOUT = 30
RETRIES = 2

# Curated EIP/ERC numbers. Many do not exist (gaps in numbering) and will 404; that is expected.
_EXPLICIT = [
    1, 2, 5, 7, 8, 20, 55, 86, 100, 101, 137, 140, 141, 145, 150, 152, 155, 158, 160, 161, 165, 170,
    190, 196, 197, 198, 211, 214, 225, 234, 609, 615, 616, 649, 658, 663, 665, 681, 689, 695, 698,
    706, 721, 747, 777, 778, 779, 820, 868, 875, 884, 897, 900, 902, 918, 998, 999, 1001, 1010,
    1011, 1013, 1014, 1015, 1046, 1052, 1057, 1062, 1066, 1077, 1078, 1080, 1081, 1087, 1102, 1108,
    1109, 1123, 1129, 1153, 1154, 1155, 1167, 1175, 1191, 1193, 1202, 1207, 1227, 1234, 1240, 1261,
    1271, 1276, 1283, 1285, 1295, 1319, 1328, 1337, 1344, 1352, 1355, 1363, 1380, 1386, 1387, 1388,
    1417, 1438, 1444, 1450, 1459, 1462, 1470, 1474, 1482, 1485, 1491, 1504, 1523, 1538, 1559, 1571,
    1577, 1581, 1588, 1592, 1613, 1616, 1620, 1633, 1679, 1681, 1682, 1702, 1706, 1710, 1716, 1753,
    1761, 1767, 1775, 1803, 1812, 1820, 1822, 1829, 1844, 1884, 1890, 1895, 1898, 1900, 1901, 1921,
    1922, 1923, 1930, 1948, 1959, 1962, 1965, 1967, 1973, 1985, 1994, 1995, 1996, 2003, 2009, 2014,
    2015, 2019, 2020, 2021, 2025, 2026, 2028, 2029, 2031, 2035, 2045, 2046, 2069, 2070, 2098, 2124,
    2135, 2157, 2159, 2200, 2228, 2242, 2255, 2256, 2266, 2294, 2304, 2309, 2315, 2327, 2330, 2333,
    2334, 2335, 2364, 2378, 2384, 2386, 2387, 2390, 2400, 2464, 2470, 2477, 2481, 2488, 2515, 2520,
    2525, 2535, 2537, 2539, 2542, 2565, 2566, 2569, 2583, 2584, 2593, 2612, 2615, 2645, 2657, 2666,
    2677, 2678, 2681, 2696, 2718, 2733, 2746, 2767, 2770, 2771, 2780, 2786, 2831, 2848, 2876, 2917,
    2926, 2929, 2930, 2935, 2936, 2937, 2938, 2970, 2976, 2980, 2981, 2982, 2997, 3000, 3005, 3009,
    3014, 3030, 3041, 3045, 3046, 3068, 3074, 3085, 3091, 3102, 3135, 3143, 3155, 3156, 3198, 3220,
    3224, 3234, 3238, 3267, 3298, 3300, 3322, 3326, 3336, 3337, 3338, 3368, 3372, 3386, 3403, 3416,
    3436, 3440, 3448, 3450, 3475, 3508, 3520, 3521, 3525, 3529, 3534, 3540, 3541, 3554, 3561, 3569,
    3583, 3589, 3607, 3643, 3651, 3668, 3670, 3675, 3690, 3709, 3712, 3722, 3754, 3756, 3770, 3772,
    3779, 3788, 3855, 3860, 3864, 4200, 4337, 4341, 4345, 4353, 4361, 4393, 4396, 4399, 4400, 4444,
    4488, 4494, 4519, 4521, 4524, 4527, 4573, 4626, 4675, 4747, 4750, 4758, 4762, 4788, 4803, 4804,
    4824, 4844, 4863, 4881, 4883, 4884, 4885, 4886, 4895, 4906, 4907, 4910, 4931, 4944, 4950, 4955,
    4973, 4974, 4987, 5000, 5003, 5006, 5007, 5008, 5018, 5023, 5027, 5050, 5058, 5081, 5095, 5114,
    5131, 5133, 5139, 5143, 5164, 5169, 5173, 5185, 5187, 5189, 5192, 5202, 5216, 5218, 5219, 5247,
    5252, 5267, 5269, 5283, 5289, 5298, 5313, 5334, 5345, 5375, 5380, 5409, 5437, 5450, 5453, 5484,
    5489, 5496, 5501, 5505, 5507, 5516, 5521, 5528, 5539, 5553, 5554, 5560, 5564, 5568, 5570, 5573,
    5585, 5604, 5606, 5615, 5625, 5630, 5633, 5639, 5643, 5646, 5656, 5660, 5679, 5700, 5719, 5725,
    5727, 5732, 5744, 5748, 5750, 5773, 5791, 5792, 5793, 5805, 5806, 5827, 5843, 5850, 5851, 5883,
    5920, 5988, 6038, 6046, 6049, 6059, 6065, 6066, 6093, 6105, 6110, 6120, 6123, 6147, 6150, 6170,
    6188, 6206, 6212, 6220, 6224, 6239, 6268, 6315, 6327, 6353, 6357, 6358, 6366, 6372, 6381, 6384,
    6404, 6454, 6464, 6475, 6493, 6506, 6551, 6596, 6604, 6617, 6672, 6690, 6780, 6785, 6786, 6787,
    6808, 6809, 6810, 6821, 6823, 6860, 6864, 6865, 6888, 6909, 6913, 6916, 6956, 6960, 6963, 6969,
    6982, 7002, 7007, 7015, 7039, 7044, 7045, 7053, 7066, 7069, 7085, 7087, 7092, 7105, 7160, 7196,
    7201, 7204, 7208, 7212, 7231, 7246, 7251, 7254, 7266, 7281, 7291, 7303, 7329, 7377, 7390, 7399,
    7401, 7405, 7409, 7410, 7412, 7425, 7432, 7439, 7444, 7480, 7484, 7495, 7496, 7498, 7503, 7506,
    7507, 7508, 7509, 7510, 7511, 7512, 7514, 7516, 7518, 7519, 7520,
    7917, 7918, 7928, 7934, 7935, 7939, 7942, 7951, 7954,
]
KNOWN_NUMBERS: list[int] = sorted(set(_EXPLICIT) | set(range(7521, 7901)))

# ethereum/consensus-specs (master). Missing files 404 and are skipped.
SPEC_FORKS = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"]
SPEC_FILES = ["beacon-chain.md", "fork-choice.md", "validator.md", "p2p-interface.md",
              "das-core.md", "light-client/sync-protocol.md", "fork.md"]

# flashbots/flashbots-docs (main). The GitHub tree API is not always reachable, so this is a
# curated page list; unknown pages simply 404.
FLASHBOTS_PAGES = [
    "flashbots-auction/overview.mdx", "flashbots-auction/quick-start.mdx",
    "flashbots-auction/advanced/rpc-endpoint.mdx", "flashbots-auction/advanced/bundle-pricing.mdx",
    "flashbots-auction/advanced/coinbase-payment.mdx",
    "flashbots-auction/advanced/eth_sendPrivateTransaction.mdx",
    "flashbots-auction/advanced/troubleshooting.mdx",
    "flashbots-auction/advanced/understanding-bundles.mdx",
    "flashbots-auction/advanced/reputation.mdx", "flashbots-auction/advanced/gas-fee-refunds.mdx",
    "flashbots-auction/advanced/bundle-relay.mdx", "flashbots-auction/advanced/private-tx.mdx",
    "flashbots-auction/searchers/faq.mdx", "flashbots-auction/searchers/advanced/rpc-endpoint.mdx",
    "flashbots-auction/searchers/advanced/bundle-pricing.mdx",
    "flashbots-auction/searchers/advanced/coinbase-payment.mdx",
    "flashbots-auction/searchers/advanced/understanding-bundles.mdx",
    "flashbots-auction/searchers/advanced/reputation.mdx",
    "flashbots-auction/searchers/advanced/troubleshooting.mdx",
    "flashbots-auction/searchers/advanced/eth_sendPrivateTransaction.mdx",
    "flashbots-auction/searchers/advanced/gas-fee-refunds.mdx",
    "flashbots-auction/searchers/libraries/ethers-js-provider.mdx",
    "flashbots-auction/searchers/libraries/web3py-provider.mdx",
    "flashbots-auction/libraries/ethers-js-provider.mdx",
    "flashbots-auction/libraries/web3py-provider.mdx",
    "flashbots-auction/libraries/golang.mdx", "flashbots-auction/libraries/rust.mdx",
    "flashbots-auction/miners/mev-geth-introduction.mdx",
    "flashbots-auction/miners/mev-geth-spec/v01.mdx",
    "flashbots-auction/miners/mev-geth-spec/v02.mdx",
    "flashbots-auction/miners/mev-geth-spec/v03.mdx",
    "flashbots-auction/miners/mev-geth-spec/v04.mdx",
    "flashbots-auction/miners/mev-geth-spec/v05.mdx",
    "flashbots-protect/overview.mdx", "flashbots-protect/quick-start.mdx",
    "flashbots-protect/mev-refunds.mdx", "flashbots-protect/settings-guide.mdx",
    "flashbots-protect/nonce-management.mdx", "flashbots-protect/rpc/quick-start.mdx",
    "flashbots-protect/rpc/mev-share.mdx", "flashbots-protect/rpc/fast-mode.mdx",
    "flashbots-protect/rpc/failed-transactions.mdx",
    "flashbots-protect/additional-documentation/eth-sendPrivateTransaction.mdx",
    "flashbots-protect/additional-documentation/ethereum-tx-privacy.mdx",
    "flashbots-protect/additional-documentation/status-api.mdx",
    "flashbots-protect/additional-documentation/bundle-cache.mdx",
    "flashbots-protect/additional-documentation/batch-cache.mdx",
    "flashbots-protect/additional-documentation/gas-fee-refunds.mdx",
    "flashbots-protect/gas-fee-refunds.mdx", "flashbots-protect/fast-mode.mdx",
    "flashbots-mev-boost/introduction.md", "flashbots-mev-boost/flashbots-builders.mdx",
    "flashbots-mev-boost/architecture-overview/block-proposal.md",
    "flashbots-mev-boost/architecture-overview/relays.md",
    "flashbots-mev-boost/architecture-overview/specifications.md",
    "flashbots-mev-boost/architecture-overview/system-design.md",
    "flashbots-mev-boost/architecture-overview/builders.md",
    "flashbots-mev-boost/getting-started/installing.md",
    "flashbots-mev-boost/getting-started/run.md",
    "flashbots-mev-boost/getting-started/testnet.md",
    "flashbots-mev-boost/getting-started/usage.md",
    "flashbots-mev-boost/getting-started/mev-boost-relay.mdx",
    "flashbots-mev-boost/getting-started/operations.md",
    "flashbots-mev-boost/faq.md", "flashbots-mev-boost/searchers.md",
    "flashbots-mev-boost/block-builders.mdx", "flashbots-mev-boost/searchers.mdx",
    "flashbots-mev-boost/faq.mdx", "flashbots-mev-boost/relay.mdx",
    "flashbots-mev-boost/block-proposal.mdx", "flashbots-mev-boost/circuit-breaker.mdx",
    "flashbots-mev-share/introduction.mdx", "flashbots-mev-share/overview.mdx",
    "flashbots-mev-share/release-notes.mdx",
    "flashbots-mev-share/searchers/getting-started.mdx",
    "flashbots-mev-share/searchers/understanding-bundles.mdx",
    "flashbots-mev-share/searchers/sending-bundles.mdx",
    "flashbots-mev-share/searchers/event-stream.mdx",
    "flashbots-mev-share/searchers/debugging.mdx",
    "flashbots-mev-share/searchers/tutorials/flash-loan-arbitrage/simple-blind-arbitrage.mdx",
    "flashbots-mev-share/searchers/tutorials/limit-order/introduction.mdx",
    "flashbots-mev-share/searchers/tutorials/limit-order/sending-bundles.mdx",
    "flashbots-mev-share/searchers/tutorials/limit-order/simulating-bundles.mdx",
    "flashbots-mev-share/searchers/libraries/golang.mdx",
    "flashbots-mev-share/searchers/libraries/rust.mdx",
    "flashbots-mev-share/searchers/libraries/typescript.mdx",
    "flashbots-mev-share/searchers/libraries/python.mdx",
    "flashbots-mev-share/searchers/faq.mdx",
    "flashbots-mev-share/searchers/sending-bundles-to-mev-share.mdx",
    "flashbots-mev-share/builders/implementation.mdx",
    "flashbots-mev-share/builders/matchmaker.mdx",
    "flashbots-data/dashboard.mdx", "flashbots-data/blocks-api.mdx",
    "flashbots-data/blocks-api/quick-start.mdx", "flashbots-data/transparency-dashboard.mdx",
    "flashbots-data/mempool-dumpster.mdx", "flashbots-data/mev-explore/overview.mdx",
    "flashbots-data/mev-explore/quick-start.mdx", "flashbots-data/mev-inspect.mdx",
    "flashbots-data/mev-share-history.mdx", "flashbots-data/mev-blocks-api.mdx",
    "flashbots-data/mev-explore-dashboard.mdx", "flashbots-data/relayscan.mdx",
    "flashbots-data/mev-share-history/overview.mdx",
    "flashbots-mev-share/searchers/event-stream-endpoints.mdx",
    "flashbots-auction/searchers/advanced/bundle-relay.mdx",
    "flashbots-auction/searchers/advanced/private-transaction.mdx",
    "flashbots-protect/additional-documentation/eth-sendPrivateTransaction.md",
    "flashbots-protect/additional-documentation/ethereum-tx-privacy.md",
    "flashbots-protect/additional-documentation/status-api.md",
    "flashbots-protect/additional-documentation/bundle-cache.md",
    "flashbots-protect/additional-documentation/batch-cache.md",
    "flashbots-protect/overview.md", "flashbots-protect/quick-start.md",
    "flashbots-protect/mev-refunds.md", "flashbots-protect/settings-guide.md",
    "flashbots-protect/nonce-management.md", "flashbots-protect/fast-mode.md",
    "flashbots-auction/overview.md", "flashbots-auction/quick-start.md",
    "flashbots-mev-share/introduction.md", "flashbots-mev-share/overview.md",
    "flashbots-mev-boost/introduction.mdx",
    "buildernet/introduction.mdx", "buildernet/system-overview.mdx",
    "buildernet/architecture.mdx", "buildernet/faq.mdx", "buildernet/order-flow.mdx",
    "buildernet/refunds.mdx", "buildernet/privacy.mdx", "buildernet/tee.mdx",
    "buildernet/configuration.mdx", "buildernet/system-design.mdx",
    "buildernet/node-operators/overview.mdx", "buildernet/node-operators/requirements.mdx",
    "buildernet/node-operators/setup.mdx", "buildernet/node-operators/attestation.mdx",
    "buildernet/node-operators/onboarding.mdx", "buildernet/releases.mdx",
    "buildernet/security.mdx", "buildernet/roadmap.mdx", "buildernet/order-flow-sharing.mdx",
    "buildernet/system/overview.mdx", "buildernet/system/architecture.mdx",
    "buildernet/system/system-design.mdx", "buildernet/system/refunds.mdx",
    "buildernet/system/order-flow.mdx", "buildernet/system/privacy.mdx",
    "buildernet/system/attestation.mdx", "buildernet/system/configuration.mdx",
    "buildernet/system/tee.mdx", "buildernet/system/faq.mdx",
    "flashbots-protect/additional-documentation/gas-fee-refunds.md",
    "flashbots-protect/protect-rpc-api.mdx", "flashbots-protect/protect-rpc-api.md",
    "flashbots-protect/refund-rule.mdx", "flashbots-protect/mev-share.mdx",
    "flashbots-protect/fast-and-private.mdx", "flashbots-protect/transaction-status-api.mdx",
    "flashbots-mev-boost/rollup-boost.mdx", "flashbots-mev-boost/relays.mdx",
    "flashbots-mev-boost/specifications.mdx", "flashbots-mev-boost/system-design.mdx",
    "rbuilder/introduction.mdx", "rbuilder/overview.mdx", "rollup-boost/introduction.mdx",
    "rollup-boost/overview.mdx", "rollup-boost/architecture.mdx",
    "flashbots-suave/introduction.mdx", "flashbots-suave/overview.mdx",
    "mev-share/introduction.mdx", "mev-share/overview.mdx",
    "flashbots-mev-share/searchers/event-stream-example.mdx",
    "flashbots-mev-share/introduction.md",
]


@dataclass
class Entry:
    id: str
    path: str
    source: str
    url: str
    number: int | None
    title: str
    status: str
    category: str
    type: str
    created: str
    requires: list[int]
    sha256: str
    bytes: int


def fetch(url: str) -> tuple[int, bytes]:
    """GET ``url``; returns (status, body). Status 0 means a transport error after retries."""
    last_err = ""
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
        except (urllib.error.URLError, TimeoutError, OSError) as e:  # transient
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    print(f"  ! {url}: {last_err}", file=sys.stderr)
    return 0, b""


def front_matter(raw: str) -> dict[str, str]:
    m = re.match(r"^---\r?\n(.*?)\r?\n---", raw, re.DOTALL)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip().strip('"')
    return meta


def parse_requires(value: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\d+", value or "")})


def first_heading(raw: str) -> str:
    m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    return m.group(1).strip() if m else ""


def acceptable(raw: bytes) -> bool:
    if len(raw) < MIN_BYTES:
        return False
    head = raw[:400].decode("utf-8", "replace")
    if "<!DOCTYPE html" in head or "<html" in head:
        return False
    meta = front_matter(head)
    return meta.get("status", "").lower() != "moved"


def build_entry(doc_id: str, rel_path: str, source: str, url: str, number: int | None,
                raw: bytes) -> Entry:
    text = raw.decode("utf-8", "replace")
    meta = front_matter(text)
    title = meta.get("title") or first_heading(text) or doc_id
    if source == "consensus-specs":
        status, category, typ = "Spec", "Consensus", "Specification"
    elif source == "flashbots-docs":
        status, category, typ = "Docs", "MEV", "Documentation"
    else:
        status = meta.get("status", "Unknown")
        category = meta.get("category", "Core" if meta.get("type", "") == "Standards Track" else "")
        typ = meta.get("type", "")
    return Entry(
        id=doc_id, path=rel_path, source=source, url=url, number=number, title=title,
        status=status, category=category or "", type=typ, created=meta.get("created", ""),
        requires=parse_requires(meta.get("requires", "")),
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
    )


def jobs(numbers: list[int]) -> list[tuple[str, str, str, int | None, str]]:
    """(doc_id, rel_path, source, number, url) for every candidate file."""
    out = []
    for n in numbers:
        out.append((f"eip-{n}", f"eips/eip-{n}.md", "EIPs", n,
                    f"{RAW}/ethereum/EIPs/master/EIPS/eip-{n}.md"))
        out.append((f"erc-{n}", f"eips/erc-{n}.md", "ERCs", n,
                    f"{RAW}/ethereum/ERCs/master/ERCS/erc-{n}.md"))
    for fork in SPEC_FORKS:
        for f in SPEC_FILES:
            stem = f.replace("/", "-").removesuffix(".md")
            out.append((f"cl-{fork}-{stem}", f"consensus-specs/cl-{fork}-{stem}.md",
                        "consensus-specs", None,
                        f"{RAW}/ethereum/consensus-specs/master/specs/{fork}/{f}"))
    for page in FLASHBOTS_PAGES:
        stem = re.sub(r"\.(mdx?|md)$", "", page)
        parts = [p.removeprefix("flashbots-") for p in stem.split("/")]
        doc_id = re.sub(r"[^a-z0-9-]", "-", ("fb-" + "-".join(parts)).lower())
        out.append((doc_id, f"flashbots/{doc_id}.md", "flashbots-docs", None,
                    f"{RAW}/flashbots/flashbots-docs/main/docs/{page}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--numbers", default="", help="comma-separated EIP numbers (default: curated list)")
    ap.add_argument("--no-specs", action="store_true", help="skip consensus-specs and flashbots docs")
    ap.add_argument("--dry-run", action="store_true", help="list candidate URLs and exit")
    a = ap.parse_args(argv)

    numbers = [int(x) for x in a.numbers.split(",") if x.strip()] or KNOWN_NUMBERS
    work = jobs(numbers)
    if a.no_specs:
        work = [w for w in work if w[2] in ("EIPs", "ERCs")]
    if a.dry_run:
        for w in work:
            print(w[4])
        return 0

    out = Path(a.out)
    t0 = time.time()
    print(f"fetching {len(work)} candidate urls with {a.workers} workers -> {out}")

    def run(job):
        code, body = fetch(job[4])
        return job, code, body

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (job, code, body) in enumerate(ex.map(run, work), 1):
            results.append((job, code, body))
            if i % 200 == 0:
                print(f"  {i}/{len(work)} ({time.time() - t0:.0f}s)")

    kept: dict[int | str, tuple[tuple, bytes]] = {}
    counts = {"ok": 0, "404": 0, "other": 0, "rejected": 0, "superseded": 0}
    for job, code, body in results:
        doc_id, rel, source, number, url = job
        if code == 404:
            counts["404"] += 1
            continue
        if code != 200:
            counts["other"] += 1
            continue
        if not acceptable(body):
            counts["rejected"] += 1
            continue
        counts["ok"] += 1
        key: int | str = number if number is not None else doc_id
        prev = kept.get(key)
        # The ERCs repo is canonical for ERCs: it wins over a copy still in the EIPs repo.
        if prev is None or (source == "ERCs" and prev[0][2] == "EIPs"):
            if prev is not None:
                counts["superseded"] += 1
            kept[key] = (job, body)
        else:
            counts["superseded"] += 1

    # A full run owns the whole tree: files that are no longer part of the corpus are removed.
    # A partial run (--numbers / --no-specs) only adds or replaces the files it fetched and merges
    # them into the existing manifest, so `--numbers 9999` can never delete the other documents.
    partial = bool(a.numbers.strip()) or a.no_specs
    for sub in ("eips", "consensus-specs", "flashbots"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if not partial:
        live_paths = {job[1] for job, _ in kept.values()}
        for sub in ("eips", "consensus-specs", "flashbots"):
            for old in (out / sub).glob("*.md"):
                if f"{sub}/{old.name}" not in live_paths:
                    old.unlink()
    entries: list[Entry] = []
    total = 0
    for job, body in kept.values():
        doc_id, rel, source, number, url = job
        (out / rel).write_bytes(body)
        total += len(body)
        entries.append(build_entry(doc_id, rel, source, url, number, body))
    if partial and (out / "MANIFEST.json").is_file():
        have = {e["id"]: e for e in json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["entries"]}
        have.update({e.id: asdict(e) for e in entries})
        entries = [Entry(**v) for v in have.values()]
        total = sum(e.bytes for e in entries)
    entries.sort(key=lambda e: (e.source, e.number if e.number is not None else -1, e.id))

    manifest = {
        "generated_by": "scripts/fetch_corpus.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": len(entries),
        "bytes": total,
        "sources": {
            "EIPs": "https://github.com/ethereum/EIPs (CC0-1.0)",
            "ERCs": "https://github.com/ethereum/ERCs (CC0-1.0)",
            "consensus-specs": "https://github.com/ethereum/consensus-specs (CC0-1.0)",
            "flashbots-docs": "https://github.com/flashbots/flashbots-docs (MIT)",
        },
        "entries": [asdict(e) for e in entries],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
    print(f"done in {time.time() - t0:.0f}s: kept {len(entries)} docs ({total / 1e6:.1f} MB); "
          f"http: {counts}")
    print("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
