from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verification.cost_model import _DEFAULT_ERROR_WEIGHTS


LAYER_ORDER = (1, 2, 3, 4)


def measure_layer_costs(records: list[dict]) -> dict[str, float]:
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for r in records:
        for lid, tok in (r.get("layer_tokens") or {}).items():
            totals[int(lid)] += float(tok)
            counts[int(lid)] += 1
    return {str(lid): round(totals[lid] / counts[lid], 3)
            for lid in sorted(counts) if counts[lid]}


def fit_efficacy(records: list[dict]) -> dict[str, float]:
    errors = [r for r in records if r.get("is_error")]
    efficacy: dict[str, float] = {}
    for i, L in enumerate(LAYER_ORDER):
        earlier = set(LAYER_ORDER[:i])
        prior_missed = [r for r in errors
                        if not (set(r.get("caught_by", [])) & earlier)]
        if not prior_missed:
            efficacy[str(L)] = 0.0
            continue
        caught = sum(1 for r in prior_missed if L in set(r.get("caught_by", [])))
        efficacy[str(L)] = round(caught / len(prior_missed), 6)
    return efficacy


def _demo_records() -> list[dict]:
    rec = []
    caught_plan = (
        [[1]] * 4 +
        [[3]] * 6 +
        [[4]] * 5 +
        [[3, 4]] * 2 +
        [[]] * 3
    )
    for cb in caught_plan:
        rec.append({"is_error": True, "caught_by": cb,
                    "layer_tokens": {"1": 0, "2": 120, "3": 305, "4": 790}})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure layer costs + fit efficacy on dev split (frozen before test).")
    ap.add_argument("--records", help="dev-split JSONL of per-step catch outcomes")
    ap.add_argument("--out", default="configs/cost_model.json")
    ap.add_argument("--demo", action="store_true", help="use synthetic dev data")
    args = ap.parse_args()

    if not args.records and not args.demo:
        ap.error("provide --records <dev jsonl> or --demo")

    print("[fit_efficacy] DEV SPLIT ONLY -- costs/efficacy frozen before test.")
    if args.demo:
        records = _demo_records()
    else:
        with open(args.records, "r", encoding="utf-8") as f:
            records = [json.loads(ln) for ln in f if ln.strip()]

    layer_costs = measure_layer_costs(records)
    efficacy = fit_efficacy(records)

    config = {
        "error_weights": dict(_DEFAULT_ERROR_WEIGHTS),
        "layer_costs": layer_costs,
        "efficacy": efficacy,
    }
    print(f"  measured layer_costs = {layer_costs}")
    print(f"  dev-fit efficacy     = {efficacy}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"[fit_efficacy] wrote {args.out}")


if __name__ == "__main__":
    main()
