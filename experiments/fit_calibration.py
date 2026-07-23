from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verification.calibration import (
    TemperatureScaling, expected_calibration_error, fit_temperature,
)


def load_records(path: str) -> dict[int, tuple[list[float], list[int]]]:
    by_layer: dict[int, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lid = int(obj["layer_id"])
            confs, correct = by_layer[lid]
            confs.append(float(obj["raw_confidence"]))
            correct.append(int(obj["correct"]))
    return dict(by_layer)


def fit_all(by_layer: dict[int, tuple[list[float], list[int]]]) -> dict[str, float]:
    temps: dict[str, float] = {}
    for lid in sorted(by_layer):
        confs, correct = by_layer[lid]
        T = fit_temperature(confs, correct)
        cal = TemperatureScaling(T)
        ece_before = expected_calibration_error(confs, correct, n_bins=10)
        ece_after = expected_calibration_error([cal.apply(p) for p in confs],
                                               correct, n_bins=10)
        temps[str(lid)] = round(T, 6)
        print(f"  Layer {lid}: n={len(confs):4d}  T={T:.3f}  "
              f"ECE {ece_before:.3f} -> {ece_after:.3f}")
    return temps


def _demo_records() -> dict[int, tuple[list[float], list[int]]]:
    l4_conf = [0.9] * 10 + [0.8] * 10
    l4_correct = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    l3_conf = [0.7] * 10 + [0.6] * 10
    l3_correct = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    return {3: (l3_conf, l3_correct), 4: (l4_conf, l4_correct)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit per-layer temperature on dev split (frozen before test).")
    ap.add_argument("--records", help="dev-split JSONL of layer confidences + correctness")
    ap.add_argument("--out", default="configs/calibration.json")
    ap.add_argument("--demo", action="store_true", help="use synthetic dev data")
    args = ap.parse_args()

    if not args.records and not args.demo:
        ap.error("provide --records <dev jsonl> or --demo")

    print("[fit_calibration] DEV SPLIT ONLY -- temperatures are frozen before test.")
    by_layer = _demo_records() if args.demo else load_records(args.records)
    temps = fit_all(by_layer)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(temps, f, indent=2, sort_keys=True)
    print(f"[fit_calibration] wrote {args.out}: {temps}")


if __name__ == "__main__":
    main()
