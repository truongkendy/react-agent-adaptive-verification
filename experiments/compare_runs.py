"""Paired comparison of a baseline log against an adaptive log.

Two runs are comparable only if they scored the *same questions* with the *same
model*. The earlier n=5 logs could not be compared because the baseline recorded
no model at all, so this script refuses by default when the runs' `.meta.json`
disagree — an unattributable EM delta is worse than no number.

Reports accuracy and cost together (repo convention: matching baseline EM at 2.5x
the LLM calls is a negative result, not a neutral one), plus a two-sided sign
test over the discordant questions so a delta of one or two answers is not read
as a trend.

    python experiments/compare_runs.py \
        --baseline results/logs/baseline_n30.json \
        --adaptive results/logs/adaptive_n30.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fields that must agree for a paired comparison to mean anything. `retrieval`
# matters even when `data` matches: the same distractor file run with and without
# `--distractor` gives the agent completely different evidence. Runs predating
# the flag have no `retrieval` key, so both sides read None and still pair.
PAIRING_KEYS = ("model", "backend", "data", "seed", "n_requested", "retrieval")


def _f1(rec: dict) -> float:
    """F1 of one record, recomputed from `prediction`/`gold` so that logs written
    before F1 was scored still compare."""
    from src.evaluation.metrics import f1_score
    if rec.get("f1") is not None:
        return float(rec["f1"])
    return f1_score(rec.get("prediction") or "", rec.get("gold") or "")


def load_run(path: str | Path) -> tuple[list[dict], dict | None]:
    p = Path(path)
    records = json.loads(p.read_text(encoding="utf-8"))
    meta_p = p.with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else None
    return records, meta


def check_pairing(b_meta: dict | None, a_meta: dict | None) -> list[str]:
    """Reasons the two runs are not safely comparable."""
    problems: list[str] = []
    for name, meta in (("baseline", b_meta), ("adaptive", a_meta)):
        if meta is None:
            problems.append(f"{name} log has no .meta.json -> model/seed unknown, "
                            f"cannot attribute the EM delta to anything")
    if b_meta and a_meta:
        for key in PAIRING_KEYS:
            bv, av = b_meta.get(key), a_meta.get(key)
            if bv != av:
                problems.append(f"{key} differs: baseline={bv!r} adaptive={av!r}")
    return problems


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test over discordant pairs (McNemar, exact form).
    Ties carry no information about direction, so only discordants count."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def totals(records: list[dict]) -> dict:
    def s(key: str) -> int:
        return sum(r.get(key, 0) or 0 for r in records)

    layer_runs: Counter = Counter()
    for r in records:
        for lid, c in (r.get("layer_runs") or {}).items():
            layer_runs[int(lid)] += c
    return {
        "n": len(records),
        "em": s("em"),
        "unfinished": sum(1 for r in records if not r.get("finished")),
        "steps": s("n_steps"),
        "llm_calls": s("n_llm_calls"),
        "react_calls": s("n_react_calls") or s("n_llm_calls"),
        "verify_calls": s("n_verify_calls"),
        "revisions": s("n_revisions"),
        "tokens": s("prompt_tokens") + s("completion_tokens"),
        "verify_est_tokens": s("verify_est_tokens"),
        "layer_runs": dict(layer_runs),
    }


def trace_summary(records: list[dict]) -> dict | None:
    traces = [t for r in records for t in (r.get("decision_traces") or [])]
    if not traces:
        return None
    n = len(traces)
    avg = lambda k: sum(t.get(k, 0) for t in traces) / n
    return {
        "n_decisions": n,
        "actions": dict(Counter(t["action"] for t in traces)),
        "mean_tau": avg("tau"),
        "mean_tau_accept": avg("tau_accept"),
        "mean_conf": avg("calibrated_confidence"),
        "mean_raw_conf": avg("raw_confidence"),
        # Decisions this close to the accept threshold are effectively coin
        # flips: the run is inert with respect to its own routing rule.
        "knife_edge": sum(1 for t in traces
                          if abs(t.get("calibrated_confidence", 0)
                                 - t.get("tau_accept", 0)) < 0.05),
        # T=1 is only *approximately* the identity: the logit/sigmoid roundtrip
        # clamps to [1e-6, 1-1e-6], so a raw 1.0 comes back as 0.999999. Judge
        # by the largest shift, well above that floor — a real fitted
        # temperature moves confidences by far more than 0.01.
        "max_calibration_shift": max(
            abs(t.get("calibrated_confidence", 0) - t.get("raw_confidence", 0))
            for t in traces),
    }


def _ratio(a: float, b: float) -> str:
    return f"{a / b:.2f}x" if b else "n/a"


def compare(b_path: str, a_path: str, force: bool = False) -> dict:
    b_rec, b_meta = load_run(b_path)
    a_rec, a_meta = load_run(a_path)

    problems = check_pairing(b_meta, a_meta)
    print("=" * 72)
    print(f"baseline : {b_path}")
    print(f"adaptive : {a_path}")
    if b_meta:
        print(f"model    : {b_meta.get('model')} ({b_meta.get('backend')})  "
              f"seed={b_meta.get('seed')} n={b_meta.get('n_requested')}")
    if a_meta:
        print(f"adaptive : tag={a_meta.get('tag')} budget={a_meta.get('budget')} "
              f"revise_bias={a_meta.get('revise_bias')}")
        print(f"routing  : calibration={a_meta.get('calibration_config') or 'DECLARED'}"
              f"  cost_model={a_meta.get('cost_model_config') or 'DECLARED'}"
              f"  identity_calibration={a_meta.get('identity_calibration')}")
    if problems:
        print("\n!! NOT SAFELY COMPARABLE:")
        for p in problems:
            print(f"   - {p}")
        if not force:
            print("\nRefusing. Re-run the pair with matching settings, or pass "
                  "--force to compare anyway (the delta will be uninterpretable).")
            raise SystemExit(1)
        print("   (--force given: continuing, treat every number below as suspect)")

    B = {r["id"]: r for r in b_rec}
    A = {r["id"]: r for r in a_rec}
    shared = [i for i in B if i in A]
    only_b, only_a = len(B) - len(shared), len(A) - len(shared)
    if only_b or only_a:
        print(f"\nNOTE: {len(shared)} shared questions "
              f"({only_b} baseline-only, {only_a} adaptive-only were dropped)")

    wins = [i for i in shared if A[i]["em"] > B[i]["em"]]
    losses = [i for i in shared if A[i]["em"] < B[i]["em"]]
    # EM alone is not enough to call a winner, and on the n30_ollama pair the two
    # metrics pointed opposite ways: EM +2 while mean F1 was -0.041. EM gives the
    # baseline zero credit for near-misses (`Major John André` vs `John André`,
    # `Stone Brewing Co.` vs `Stone Brewing`) that it then cannot lose, so
    # turning one into a step-exhausted non-answer registers as no loss at all.
    # On the distractor pair, EM was an exact 11-11 tie with 0 wins and 0 losses
    # while F1 fell 0.494 -> 0.452 on two questions that went to `None`.
    bf = {i: _f1(B[i]) for i in shared}
    af = {i: _f1(A[i]) for i in shared}
    f1_wins = [i for i in shared if af[i] > bf[i] + 1e-9]
    f1_losses = [i for i in shared if af[i] < bf[i] - 1e-9]
    p_f1 = sign_test_p(len(f1_wins), len(f1_losses))
    bt = totals([B[i] for i in shared])
    at = totals([A[i] for i in shared])
    n = len(shared)
    p = sign_test_p(len(wins), len(losses))

    print(f"\n----- accuracy (paired over {n} questions) -----")
    print(f"  baseline EM : {bt['em']}/{n} = {bt['em']/max(n,1):.3f}")
    print(f"  adaptive EM : {at['em']}/{n} = {at['em']/max(n,1):.3f}"
          f"   (delta {at['em']-bt['em']:+d})")
    print(f"  per-question: same={n-len(wins)-len(losses)} "
          f"adaptive_win={len(wins)} adaptive_lose={len(losses)}")
    print(f"  sign test   : p = {p:.3f} "
          f"({'not significant' if p >= 0.05 else 'significant'} at 0.05)")
    mb, ma = sum(bf.values()) / max(n, 1), sum(af.values()) / max(n, 1)
    print(f"  baseline F1 : {mb:.3f}   (mean)")
    print(f"  adaptive F1 : {ma:.3f}   (delta {ma - mb:+.3f})")
    print(f"  per-question: same={n-len(f1_wins)-len(f1_losses)} "
          f"adaptive_win={len(f1_wins)} adaptive_lose={len(f1_losses)}   "
          f"sign test p = {p_f1:.3f}")
    for i in f1_losses:
        print(f"    F1 loss {bf[i]:.2f} -> {af[i]:.2f}  gold={B[i].get('gold')!r} "
              f"base={B[i].get('prediction')!r} adap={A[i].get('prediction')!r}")
    print(f"  unfinished  : baseline={bt['unfinished']} adaptive={at['unfinished']}"
          f"   (step exhaustion; revisions charge against max_steps)")

    print(f"\n----- cost -----")
    print(f"  LLM calls   : {bt['llm_calls']} -> {at['llm_calls']}  "
          f"({_ratio(at['llm_calls'], bt['llm_calls'])})   "
          f"react={at['react_calls']} verify={at['verify_calls']}")
    print(f"  tokens      : {bt['tokens']} -> {at['tokens']}  "
          f"({_ratio(at['tokens'], bt['tokens'])})")
    print(f"  agent steps : {bt['steps']} -> {at['steps']}   "
          f"revisions={at['revisions']}")
    print(f"  verify est. : {at['verify_est_tokens']} tokens (orchestrator's own)")
    if at["layer_runs"]:
        print(f"  layer runs  : " +
              " ".join(f"L{l}={at['layer_runs'].get(l, 0)}" for l in (1, 2, 3, 4, 5)))
    if bt["em"] and at["em"]:
        print(f"  calls per EM: {bt['llm_calls']/bt['em']:.1f} -> "
              f"{at['llm_calls']/at['em']:.1f}")

    ts = trace_summary([A[i] for i in shared])
    if ts:
        print(f"\n----- router ({ts['n_decisions']} decisions) -----")
        for act, c in sorted(ts["actions"].items(), key=lambda kv: -kv[1]):
            print(f"  {act:<18}: {c:4d}  ({c/ts['n_decisions']:.1%})")
        print(f"  mean tau / tau_accept : {ts['mean_tau']:.3f} / "
              f"{ts['mean_tau_accept']:.3f}")
        print(f"  mean conf (raw)       : {ts['mean_conf']:.3f} "
              f"({ts['mean_raw_conf']:.3f})")
        shift = ts["max_calibration_shift"]
        print(f"  calibration           : "
              f"{'ACTIVE' if shift > 0.01 else 'identity (no-op)'} "
              f"(max shift {shift:.4f})")
        print(f"  knife-edge decisions  : {ts['knife_edge']}/{ts['n_decisions']} "
              f"(|conf - tau_accept| < 0.05)")

    verdict = ("adaptive improves EM" if at["em"] > bt["em"] and p < 0.05 else
               "no significant EM difference" if at["em"] == bt["em"] or p >= 0.05 else
               "adaptive hurts EM")
    print(f"\n  VERDICT: {verdict}, at "
          f"{_ratio(at['llm_calls'], bt['llm_calls'])} the LLM calls.")
    if at["em"] <= bt["em"]:
        print("  Cost went up without accuracy going up -> negative result.")
    # Said explicitly, because an EM tie that hides an F1 loss reads as neutral.
    if ma < mb - 1e-9 and at["em"] >= bt["em"]:
        print(f"  WARNING: EM did not fall but mean F1 did ({mb:.3f} -> {ma:.3f}). "
              f"Adaptive is turning partial answers into non-answers; EM cannot "
              f"see that. Believe F1 here.")

    return {
        "baseline": {"path": str(b_path), "meta": b_meta, "totals": bt},
        "adaptive": {"path": str(a_path), "meta": a_meta, "totals": at,
                     "traces": ts},
        "paired": {"n": n, "wins": wins, "losses": losses,
                   "em_delta": at["em"] - bt["em"], "sign_test_p": p,
                   "f1_baseline": mb, "f1_adaptive": ma,
                   "f1_wins": f1_wins, "f1_losses": f1_losses,
                   "f1_sign_test_p": p_f1},
        "pairing_problems": problems,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adaptive", required=True)
    ap.add_argument("--force", action="store_true",
                    help="compare even when the runs' meta disagree")
    ap.add_argument("--out", help="also write the comparison as JSON")
    args = ap.parse_args()

    report = compare(args.baseline, args.adaptive, force=args.force)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved comparison -> {out}")


if __name__ == "__main__":
    main()
