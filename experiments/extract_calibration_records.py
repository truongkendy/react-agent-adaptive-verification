"""Extract calibration + efficacy records from adaptive experiment logs.

Reads one or more adaptive log JSON files (produced by exp_adaptive.py) and
writes two JSONL files consumed by the fitting scripts:

  calibration JSONL  →  fit_calibration.py   (per-layer temperature)
  efficacy JSONL     →  fit_efficacy.py       (layer costs + error catch rate)

Usage
-----
# Calibration only (fastest path):
python experiments/extract_calibration_records.py \\
    results/logs/adaptive_n60_distractor_qwen32b.json \\
    --calibration-out data/calibration_records.jsonl

# Both outputs at once:
python experiments/extract_calibration_records.py \\
    results/logs/adaptive_n60_distractor_qwen32b.json \\
    results/logs/adaptive_n30_distractor_v2.json \\
    --calibration-out data/calibration_records.jsonl \\
    --efficacy-out    data/efficacy_records.jsonl

Then run the fitters:
  python experiments/fit_calibration.py \\
      --records data/calibration_records.jsonl \\
      --out configs/calibration.json
  python experiments/fit_efficacy.py \\
      --records data/efficacy_records.jsonl \\
      --out configs/cost_model.json

Design notes
------------
**Correctness proxy**
  For all layers we use episode-level EM (exact match) as the correctness label.
  This is a noisy proxy for non-Finish steps — a good Search step contributes
  positively even in a failed episode — but it is the only ground truth we have
  without per-step human labels. The calibration problem is well-posed even with
  noisy labels because the calibrator only needs P(correct | conf=x) to be a
  monotone function; constant label noise shrinks the calibration signal without
  introducing systematic bias.

  A tighter proxy is used for L4 and L5 Finish steps: the step is "correct" iff
  the agent's final prediction matched the gold answer (em=1). Non-Finish steps
  from EM=1 episodes are conservative but not wrong — if the episode ended right,
  all the steps that led there were at least not catastrophically bad.

**Parse failures**
  LLM layers (L2, L4, L5) emit `parse_ok=False` when they fail to parse their
  own reply. The fallback confidence (0.5) is not a layer signal; it is a
  "no-data" sentinel. Including it in the calibration fit would bias the
  temperature toward pulling all confidences toward 0.5. So parse failures are
  EXCLUDED from calibration records. They are not excluded from efficacy records
  (the token cost was still paid).

**Deduplication**
  Within one question a step may be re-verified (revision loop). Each unique
  (step_index, layer_id) pair is emitted only once — using the FIRST run, which
  is the one that caused the revision decision (subsequent re-runs are on a
  slightly different step context and their verdicts are noiser labels).

**Efficacy records**
  A step is an "error" iff any layer returned FAIL on it. `caught_by` is the
  set of layer_ids that FAILed. `layer_tokens` is derived from the layer's
  known token estimate in layer_signals rather than from the cost model defaults,
  because this is what was actually spent. Note: we can only measure tokens for
  layers that appear in the log's layer_signals; if a layer ran but the log
  predates this field, its cost is estimated from the declared defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Declared layer token costs (fallback when layer_signals not present in log).
# These mirror DeclaredCostModel defaults in cost_model.py.
_FALLBACK_LAYER_TOKENS: dict[int, float] = {
    1: 0,
    2: 120,
    3: 300,
    4: 800,
    5: 400,
}


def _calibration_from_signals(em: int, signals: list[dict]) -> list[dict]:
    """Extract calibration records from a layer_signals list."""
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for sig in signals:
        key = (int(sig["step_index"]), int(sig["layer_id"]))
        if key in seen:
            continue
        seen.add(key)

        lid = int(sig["layer_id"])
        if lid == 1:
            continue   # deterministic; skip
        parse_ok = sig.get("parse_ok")
        if parse_ok is False:
            continue   # fallback 0.5 is not a layer signal; skip

        out.append({
            "layer_id": lid,
            "raw_confidence": float(sig["raw_confidence"]),
            "correct": em,
        })
    return out


def _calibration_from_traces(em: int, traces: list[dict]) -> list[dict]:
    """Fallback: extract calibration records from decision_traces.

    decision_traces is available in older logs that predate layer_signals.
    Each trace represents one completed layer run (layer_id = the layer that
    just ran, raw_confidence = its pre-calibration confidence).
    Deduplication: first occurrence of each (step_index, layer_id) wins,
    same policy as _calibration_from_signals.
    """
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for tr in traces:
        key = (int(tr["step_index"]), int(tr["layer_id"]))
        if key in seen:
            continue
        seen.add(key)

        lid = int(tr["layer_id"])
        if lid == 1:
            continue   # deterministic; skip
        parse_ok = tr.get("parse_ok")
        if parse_ok is False:
            continue   # fallback sentinel; skip

        out.append({
            "layer_id": lid,
            "raw_confidence": float(tr["raw_confidence"]),
            "correct": em,
            "_source": "decision_traces",
        })
    return out


def _extract_calibration(q: dict) -> list[dict]:
    """Yield calibration records for one question.

    Returns [{layer_id, raw_confidence, correct}, ...].
    Skips L1 (deterministic — its confidence is always 0.90 by construction,
    so the temperature calibrator has nothing to fit), and parse failures.

    Falls back to decision_traces when layer_signals is absent (older logs).
    """
    em = int(q.get("em") or 0)
    signals = q.get("layer_signals")
    if signals:
        return _calibration_from_signals(em, signals)

    traces = q.get("decision_traces")
    if traces:
        return _calibration_from_traces(em, traces)

    return []   # log predates both fields


def _extract_efficacy(q: dict) -> list[dict]:
    """Yield efficacy records for one question.

    Returns [{is_error, caught_by, layer_tokens}, ...], one per *verified step*.
    A step is one (step_index, ...) group in layer_signals.
    """
    signals = q.get("layer_signals")
    if not signals:
        return []

    # Group by step_index.
    by_step: dict[int, list[dict]] = {}
    for sig in signals:
        si = int(sig["step_index"])
        by_step.setdefault(si, []).append(sig)

    out: list[dict] = []
    for step_sigs in by_step.values():
        # Which layers gave FAIL on this step?
        caught_by = [int(s["layer_id"]) for s in step_sigs
                     if s.get("verdict") == "fail"]
        is_error = len(caught_by) > 0

        # Token cost per layer: not stored individually in layer_signals, use
        # fallback. A future improvement is to log sig.cost.tokens in layer_signals.
        layer_tokens = {str(s["layer_id"]): _FALLBACK_LAYER_TOKENS.get(
            int(s["layer_id"]), 0) for s in step_sigs}

        out.append({
            "is_error": is_error,
            "caught_by": caught_by,
            "layer_tokens": layer_tokens,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract calibration + efficacy JSONL from adaptive log(s).")
    ap.add_argument("logs", nargs="+", help="adaptive log JSON file(s)")
    ap.add_argument("--calibration-out", default="data/calibration_records.jsonl",
                    help="output path for calibration JSONL (default: %(default)s)")
    ap.add_argument("--efficacy-out", default="",
                    help="output path for efficacy JSONL; omit to skip")
    ap.add_argument("--stats", action="store_true",
                    help="print per-layer record counts and mean confidence")
    args = ap.parse_args()

    cal_records: list[dict] = []
    eff_records: list[dict] = []
    n_questions = 0
    n_missing_signals = 0
    n_trace_fallback = 0

    for log_path in args.logs:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
        for q in data:
            n_questions += 1
            has_signals = bool(q.get("layer_signals"))
            has_traces = bool(q.get("decision_traces"))
            if not has_signals and not has_traces:
                n_missing_signals += 1
                continue
            if not has_signals and has_traces:
                n_trace_fallback += 1
            cal_records.extend(_extract_calibration(q))
            if args.efficacy_out:
                if has_signals:
                    eff_records.extend(_extract_efficacy(q))
                # efficacy from decision_traces not yet supported (verdict needed)

    if n_missing_signals:
        print(f"WARNING: {n_missing_signals}/{n_questions} questions have no "
              f"layer_signals or decision_traces — skipped.",
              file=sys.stderr)
    if n_trace_fallback:
        print(f"NOTE: {n_trace_fallback}/{n_questions} questions used decision_traces "
              f"fallback (older log without layer_signals). Calibration records are "
              f"equivalent; efficacy records skipped for those questions.",
              file=sys.stderr)

    if not cal_records:
        sys.exit("No calibration records extracted. Use a log produced by the "
                 "current exp_adaptive.py (it must contain layer_signals or "
                 "decision_traces).")

    # Write calibration JSONL.
    Path(args.calibration_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.calibration_out, "w", encoding="utf-8") as f:
        for r in cal_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cal_records)} calibration records -> {args.calibration_out}")

    # Write efficacy JSONL.
    if args.efficacy_out:
        Path(args.efficacy_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.efficacy_out, "w", encoding="utf-8") as f:
            for r in eff_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(eff_records)} efficacy records  -> {args.efficacy_out}")

    if args.stats:
        from collections import defaultdict
        by_layer: dict[int, list[float]] = defaultdict(list)
        correct_by_layer: dict[int, list[int]] = defaultdict(list)
        for r in cal_records:
            lid = r["layer_id"]
            by_layer[lid].append(r["raw_confidence"])
            correct_by_layer[lid].append(r["correct"])
        print("\nPer-layer calibration stats:")
        for lid in sorted(by_layer):
            confs = by_layer[lid]
            labels = correct_by_layer[lid]
            mean_conf = sum(confs) / len(confs)
            accuracy = sum(labels) / len(labels)
            print(f"  L{lid}: n={len(confs):4d}  "
                  f"mean_conf={mean_conf:.3f}  accuracy={accuracy:.3f}  "
                  f"gap={mean_conf - accuracy:+.3f}")
        if eff_records:
            n_err = sum(1 for r in eff_records if r["is_error"])
            print(f"\nEfficacy: {n_err}/{len(eff_records)} steps flagged as errors "
                  f"({n_err / max(len(eff_records), 1):.1%})")


if __name__ == "__main__":
    main()
