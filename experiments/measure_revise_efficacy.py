"""Measure `revise_efficacy` from adaptive run logs.

DEV SPLIT ONLY — freeze before test.

`revise_efficacy` = P(the agent actually repairs the step | we asked it to
revise). It is the single most consequential number in the router: `tau_accept`
is derived from it, and every accept/revise decision compares against that. It
has been DECLARED at 0.50 since the cost model was written, and never measured —
so the threshold that decides whether to spend an agent step rests on a guess.

This script measures it from what the logs already record. Three estimators,
because "repaired" is not one thing:

  answer_rate   of the blocks whose trajectory went on to produce a correct
                (EM=1) answer, over all blocks. The outcome the thesis cares
                about, and the most conservative.
  finish_rate   of the blocks whose trajectory went on to produce *any* answer,
                over all blocks. Separates "the block redirected the agent" from
                "the block burned the last step".
  change_rate   of the blocks whose next action differed from the blocked one.
                An upper bound: behaviour changed, outcome may not have.

`answer_rate` is the one to put in configs/cost_model.json. It is a per-episode
attribution (a block is credited with the trajectory's outcome), which
over-credits when a trajectory had several blocks and only one mattered — stated
here rather than hidden, because there is no per-step ground truth in the logs.

Usage:
    python experiments/measure_revise_efficacy.py results/logs/adaptive_n30*.json
    python experiments/measure_revise_efficacy.py <logs...> --out configs/cost_model.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verification.cost_model import default_cost_model

_ACTION = re.compile(r"^Action \d+:\s*(.*)$")
_BLOCK = re.compile(r"^Observation \d+:\s*\[Verify(?::(\w+))?\]\s*(.*)$")


def _canon(action: str) -> str:
    m = re.match(r"^(\w+)\[(.*)\]$", (action or "").strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        return re.sub(r"\s+", " ", (action or "").strip().lower())
    return f"{m.group(1).lower()}[{re.sub(r'\s+', ' ', m.group(2).strip().lower())}]"


def blocks_of(record: dict) -> list[dict]:
    """One entry per injected `[Verify...]` observation in the trajectory."""
    lines = (record.get("trajectory") or "").split("\n")
    actions: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _ACTION.match(line)
        if m:
            actions.append((i, m.group(1).strip()))

    out = []
    for i, line in enumerate(lines):
        m = _BLOCK.match(line)
        if not m:
            continue
        blocked = next((a for j, a in reversed(actions) if j < i), "")
        nxt = next((a for j, a in actions if j > i), None)
        out.append({
            "error_type": m.group(1) or "none",
            "feedback": m.group(2),
            "blocked_action": blocked,
            "next_action": nxt,
            "changed": nxt is not None and _canon(nxt) != _canon(blocked),
            "em": int(record.get("em") or 0),
            "finished": bool(record.get("finished")),
            "question_id": record.get("id"),
        })
    return out


def measure(paths: list[str]) -> dict:
    blocks, n_questions, n_with_blocks = [], 0, 0
    for path in paths:
        records = json.loads(open(path, encoding="utf-8").read())
        for rec in records:
            n_questions += 1
            b = blocks_of(rec)
            if b:
                n_with_blocks += 1
            blocks.extend(b)

    n = len(blocks)
    if not n:
        return {"n_blocks": 0, "n_questions": n_questions}

    by_type: dict[str, dict] = {}
    for b in blocks:
        d = by_type.setdefault(b["error_type"], {"n": 0, "em": 0, "fin": 0, "chg": 0})
        d["n"] += 1
        d["em"] += b["em"]
        d["fin"] += int(b["finished"])
        d["chg"] += int(b["changed"])

    return {
        "sources": paths,
        "n_questions": n_questions,
        "n_questions_with_blocks": n_with_blocks,
        "n_blocks": n,
        "answer_rate": sum(b["em"] for b in blocks) / n,
        "finish_rate": sum(b["finished"] for b in blocks) / n,
        "change_rate": sum(b["changed"] for b in blocks) / n,
        "by_error_type": {
            k: {"n": v["n"],
                "answer_rate": v["em"] / v["n"],
                "finish_rate": v["fin"] / v["n"],
                "change_rate": v["chg"] / v["n"]}
            for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["n"])
        },
        "repeat_blocks": sum(
            1 for b in blocks
            if sum(1 for o in blocks
                   if o["question_id"] == b["question_id"]
                   and _canon(o["blocked_action"]) == _canon(b["blocked_action"])) > 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="adaptive run log(s) with trajectories")
    ap.add_argument("--out", default="",
                    help="write a cost_model.json carrying the measured "
                         "revise_efficacy (declared values for everything else)")
    args = ap.parse_args()

    print("=" * 72)
    print("DEV SPLIT ONLY — freeze this before touching the test split.")
    print("=" * 72)

    m = measure(args.logs)
    if not m.get("n_blocks"):
        sys.exit(f"No revisions found in {args.logs} — nothing to measure. "
                 f"(A log with 0 revisions cannot estimate revise_efficacy.)")

    print(f"\nquestions            : {m['n_questions']} "
          f"({m['n_questions_with_blocks']} had at least one block)")
    print(f"blocks (revisions)   : {m['n_blocks']}  "
          f"({m['repeat_blocks']} were a repeat block on the same action)")
    print(f"\n  answer_rate  : {m['answer_rate']:.3f}   "
          f"<- use this as revise_efficacy")
    print(f"  finish_rate  : {m['finish_rate']:.3f}   "
          f"(block led to any answer at all)")
    print(f"  change_rate  : {m['change_rate']:.3f}   "
          f"(retry differed from the blocked action; upper bound)")
    print(f"\n  declared     : {default_cost_model().revise_efficacy():.3f}   "
          f"(the unmeasured constant currently in use)")

    print("\n  by suspected error type:")
    for k, v in m["by_error_type"].items():
        print(f"    {k:<10} n={v['n']:<4} answer={v['answer_rate']:.3f} "
              f"finish={v['finish_rate']:.3f} change={v['change_rate']:.3f}")

    if m["answer_rate"] == 0.0:
        print("\n  answer_rate is 0: no block in these logs led to a correct "
              "answer. Fitting revise_efficacy to 0 makes tau_accept 0, i.e. "
              "the router would never revise and the cascade would reduce to "
              "the baseline. That is the honest reading of these logs, and the "
              "reason to fix the repair loop rather than the threshold.")

    if args.out:
        cm = default_cost_model()
        cfg = {
            "error_weights": cm.error_weights,
            "layer_costs": {str(k): v for k, v in cm.layer_costs.items()},
            "efficacy": {str(k): v for k, v in cm.efficacy.items()},
            "revision_cost": cm.revision_cost(0),
            "revise_efficacy": m["answer_rate"],
            "revision_repeat_penalty": cm._revision_repeat_penalty,
            "_provenance": {
                "revise_efficacy": "MEASURED (answer_rate)",
                "everything_else": "DECLARED",
                "sources": m["sources"],
                "n_blocks": m["n_blocks"],
            },
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nWrote {args.out} (revise_efficacy={m['answer_rate']:.4f}, "
              f"everything else declared)")


if __name__ == "__main__":
    main()
