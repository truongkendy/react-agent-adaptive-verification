from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_baseline import (
    add_distractor_arg, build_llm, check_distractor_data, load_dotenv,
)
from src.verification.routing_config import add_routing_args, load_routing_config


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to hotpot_dev_fullwiki.json")
    ap.add_argument("--n", type=int, default=10, help="number of questions (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--key-env", default="GROQ_API_KEY")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--llm-timeout", type=int, default=120,
                    help="per-call HTTP timeout in seconds (ollama). Raise it for "
                         "a large local model: 600 for 32B, 900 for 70B")
    ap.add_argument("--budget", type=int, default=1620,
                    help="per-step verification token budget for the orchestrator. "
                         "Default 1620 ≈ max one-step layer spend (L1+L2+L4 ≈ 1620 tok) "
                         "so the budget-pressure term in the cost model is active. "
                         "Use a higher value only to deliberately suppress that term.")
    ap.add_argument("--no-layer4", action="store_true",
                    help="disable Layer 4 (LLM judge)")
    ap.add_argument("--no-layer3", action="store_true",
                    help="disable Layer 3 (retrieval + NLI); skips the model load")
    ap.add_argument("--no-layer5", action="store_true",
                    help="disable Layer 5 (post-hoc answer gate on Finish[...])")
    ap.add_argument("--revise-bias", type=float, default=1.0,
                    help="multiplier on the derived revision cost: >1 revises "
                         "less readily, <1 more readily (sweep this)")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--tpm", type=int, default=0,
                    help="account tokens-per-minute limit; paces requests to stay "
                         "under it instead of discovering it via 429s "
                         "(Groq free tier: 12000). 0 = no pacing")
    ap.add_argument("--out", default="results/logs/adaptive.json")
    ap.add_argument("--resume", action="store_true",
                    help="reuse questions already present in --out and only run "
                         "the missing ones (results are flushed per question)")
    ap.add_argument("--verbose", action="store_true")
    add_routing_args(ap)
    add_distractor_arg(ap)
    args = ap.parse_args()

    from src.agents.adaptive_react import AdaptiveReActAgent
    from src.tools.wikipedia import MediaWikiBackend, WikiEnv
    from src.benchmarks.hotpotqa import load_examples
    from src.evaluation.metrics import exact_match, f1_score

    # Resolved before anything expensive runs: a bad --calibration path should
    # fail here, not after the NLI download and the first API call.
    routing = load_routing_config(args.calibration, args.cost_model)

    examples = load_examples(args.data, args.n, args.seed)
    check_distractor_data(args, examples)
    use_l3, use_l4 = not args.no_layer3, not args.no_layer4
    use_l5 = not args.no_layer5
    tag = ("adaptive[L1+L2" + ("+L3" if use_l3 else "")
           + ("+L4" if use_l4 else "") + ("+L5" if use_l5 else "") + "]")
    print(f"Running {tag} ReAct on {len(examples)} questions | "
          f"backend={args.backend} model={args.model} budget={args.budget} "
          f"revise_bias={args.revise_bias} "
          f"retrieval={'distractor' if args.distractor else 'fullwiki'}")
    # The budget-pressure arm of the cost model (scarcity raises c_next, which
    # lowers tau and suppresses escalation) only does anything if the budget can
    # actually be exhausted. On the n=30 distractor run --budget 5000 left
    # ~4300 unspent at every decision, so that whole term was inert and the run
    # was not testing the adaptive part of the router at all.
    max_spend = sum(routing.cost_model.layer_cost(l) for l in (2, 3, 4, 5))
    if args.budget > 1.5 * max_spend:
        print(f"WARNING: --budget {args.budget} exceeds 1.5x the most the "
              f"registered layers can spend on one step ({max_spend:.0f} tokens), "
              f"so the budget-pressure term in the cost model is inert. "
              f"Try --budget {int(max_spend)} to make it bind.")
    print(routing.describe() + "\n")

    llm = build_llm(args)
    dbackend = None
    if args.distractor:
        from src.tools.wikipedia import DistractorBackend
        dbackend = DistractorBackend()
        env = WikiEnv(dbackend)
    else:
        env = WikiEnv(MediaWikiBackend())

    if use_l3:
        print("Loading Layer 3 NLI model (DeBERTa)... uses cache if already fetched.")
    agent = AdaptiveReActAgent(llm, env, max_steps=args.max_steps,
                               verbose=args.verbose, use_layer4=use_l4,
                               use_layer3=use_l3, use_layer5=use_l5,
                               routing=routing,
                               budget=args.budget, revise_bias=args.revise_bias)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, n_em, sum_f1 = [], 0, 0.0
    layer_totals: Counter = Counter()
    action_totals: Counter = Counter()
    # Parse failures per LLM layer. A layer that cannot parse its own reply is
    # billed in full and then abstains, so this has to be reported next to the
    # layer-run counts or the cascade looks healthy when it is mute.
    parse_fail_totals: Counter = Counter()
    parse_total_totals: Counter = Counter()
    all_traces: list[dict] = []
    # An adaptive run is long enough to be interrupted (~20 min at n=15 on a
    # local model), and the log used to be written only at the very end, so a
    # kill lost every completed question. Results are flushed after each one and
    # --resume picks up where the previous attempt stopped.
    if args.resume and out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        n_em = sum(r["em"] for r in results)
        sum_f1 = sum(r.get("f1") or 0.0 for r in results)
        for r in results:
            for lid, c in (r.get("layer_runs") or {}).items():
                layer_totals[int(lid)] += c
            for lid, c in (r.get("layer_parse_fail") or {}).items():
                parse_fail_totals[int(lid)] += c
            for lid, c in (r.get("layer_parse_total") or {}).items():
                parse_total_totals[int(lid)] += c
            for t in (r.get("decision_traces") or []):
                action_totals[t["action"]] += 1
                all_traces.append({"id": r["id"], **t})
        print(f"--resume: {len(results)} question(s) already in {out_path}, "
              f"skipping those\n")
    done_ids = {r["id"] for r in results}

    def flush() -> None:
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    t0 = time.time()
    for k, ex in enumerate(examples, 1):
        if ex["_id"] in done_ids:
            continue
        if args.sleep and k > 1:
            time.sleep(args.sleep)
        # Swap in this question's paragraph set before the agent resets the env.
        if dbackend is not None:
            dbackend.load(ex.get("context") or [])
        try:
            res = agent.run(ex["question"], gold=ex.get("answer"))
        except Exception as e:
            print(f"[{k}] ERROR: {e}")
            continue
        em = exact_match(res.prediction, ex.get("answer", ""))
        n_em += em
        # Scored alongside EM because the two disagree, and EM is the more
        # flattering of the pair: on the n30_ollama comparison EM read +2 while
        # mean F1 read -0.041, because EM gives no credit for `Major John André`
        # against gold `John André` and so registers no loss when that answer is
        # replaced by nothing at all.
        f1 = f1_score(res.prediction or "", ex.get("answer", "") or "")
        sum_f1 += f1
        for lid, c in res.layer_runs.items():
            layer_totals[lid] += c
        for lid, c in res.layer_parse_fail.items():
            parse_fail_totals[lid] += c
        for lid, c in res.layer_parse_total.items():
            parse_total_totals[lid] += c
        # Every router decision, with both derived thresholds. Without this the
        # run is undiagnosable: you cannot tell a step that passed cheaply from
        # one that escalated and then passed anyway.
        traces = [asdict(t) for t in res.decision_traces]
        for t in traces:
            action_totals[t["action"]] += 1
            all_traces.append({"id": ex["_id"], **t})
        results.append({
            "id": ex["_id"],
            "question": ex["question"],
            "gold": ex.get("answer"),
            "prediction": res.prediction,
            "em": em,
            "f1": f1,
            "n_steps": res.n_steps,
            "n_react_calls": res.n_react_calls,
            "n_verify_calls": res.n_verify_calls,
            "n_llm_calls": res.n_llm_calls,
            "n_revisions": res.n_revisions,
            "finished": res.finished,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "layer_runs": res.layer_runs,
            "layer_parse_fail": res.layer_parse_fail,
            "layer_parse_total": res.layer_parse_total,
            "verify_est_tokens": res.verify_est_tokens,
            "trajectory": res.trajectory,
            "decision_traces": traces,
            # Per-step, per-layer signals for calibration fitting.
            # Each entry: {step_index, layer_id, verdict, raw_confidence, parse_ok}
            # Feed to experiments/extract_calibration_records.py.
            "layer_signals": res.layer_signals,
        })
        flush()
        lr = " ".join(f"L{l}:{res.layer_runs.get(l, 0)}" for l in (1, 2, 3, 4, 5))
        print(f"[{k}/{len(examples)}] EM={em} pred={res.prediction!r} "
              f"gold={ex.get('answer')!r} steps={res.n_steps} "
              f"calls={res.n_llm_calls}(react={res.n_react_calls}/verify={res.n_verify_calls}) "
              f"rev={res.n_revisions} [{lr}]")

    dt = time.time() - t0
    # Same guard as exp_baseline: an all-errored run must not leave an empty log
    # that later reads as a legitimate EM 0.000.
    if not results:
        sys.exit(f"\nAll {len(examples)} questions errored — nothing scored, no log "
                 f"written. See the errors above (a quota wall is the usual cause).")
    if len(results) < len(examples):
        print(f"\nWARNING: only {len(results)}/{len(examples)} questions scored; "
              f"the rest errored. This log is NOT comparable to a complete run.")

    n = len(results)
    tot_in = sum(r["prompt_tokens"] for r in results)
    tot_out = sum(r["completion_tokens"] for r in results)
    print(f"\n================= SUMMARY [{tag.upper()}] =================")
    print(f"Questions scored      : {n}")
    print(f"Exact Match           : {n_em}/{n} = {n_em / max(n, 1):.3f}")
    print(f"Mean F1               : {sum_f1 / max(n, 1):.3f}")
    print(f"Avg LLM calls         : {sum(r['n_llm_calls'] for r in results)/max(n,1):.2f}")
    print(f"  - react calls       : {sum(r['n_react_calls'] for r in results)/max(n,1):.2f}")
    print(f"  - verify calls      : {sum(r['n_verify_calls'] for r in results)/max(n,1):.2f}")
    print(f"Avg steps             : {sum(r['n_steps'] for r in results)/max(n,1):.2f}")
    print(f"Avg revisions/question: {sum(r['n_revisions'] for r in results)/max(n,1):.2f}")
    print(f"Layer runs (total)    : " +
          " ".join(f"L{l}={layer_totals.get(l, 0)}" for l in (1, 2, 3, 4, 5)))
    # A layer that cannot parse its own reply degrades to UNSURE at confidence
    # 0.5, which fuse() treats as non-decisive — full price, no signal. Printed
    # unconditionally: a 0% rate is the reassurance, and a high one is the first
    # thing that explains a run where the cascade never decided anything.
    if parse_total_totals:
        parts = []
        for l in sorted(parse_total_totals):
            tot = parse_total_totals[l]
            bad = parse_fail_totals.get(l, 0)
            parts.append(f"L{l}={bad}/{tot} ({bad / max(tot, 1):.0%})")
        print(f"Parse failures        : " + "  ".join(parts))
    print(f"Total tokens (in/out) : {tot_in}/{tot_out}")
    print(f"Time                  : {dt:.1f}s")

    n_dec = len(all_traces)
    if n_dec:
        print(f"\n----- router decisions ({n_dec}) -----")
        for act, c in action_totals.most_common():
            print(f"  {act:<18}: {c:4d}  ({c / n_dec:.1%})")
        _avg = lambda k: sum(t[k] for t in all_traces) / n_dec
        print(f"  mean tau (escalate): {_avg('tau'):.3f}")
        print(f"  mean tau_accept    : {_avg('tau_accept'):.3f}")
        print(f"  mean confidence    : {_avg('calibrated_confidence'):.3f} "
              f"(raw {_avg('raw_confidence'):.3f})")
        # If confidence never crosses either threshold the run is inert, so say
        # so here rather than leaving it to be discovered in the logs.
        near = sum(1 for t in all_traces
                   if abs(t["calibrated_confidence"] - t["tau_accept"]) < 0.05)
        print(f"  |conf - tau_accept| < 0.05 on {near}/{n_dec} decisions")

    flush()
    # Sibling file rather than a wrapper object, so the log stays a flat list
    # of per-question records like exp_baseline's. Without this a log cannot be
    # attributed to a model, which makes cross-run EM comparisons meaningless.
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "experiment": "adaptive", "tag": tag,
        "backend": args.backend, "model": args.model,
        "base_url": args.base_url, "data": args.data,
        "n_requested": args.n, "n_scored": n, "seed": args.seed,
        "max_steps": args.max_steps, "budget": args.budget,
        "retrieval": "distractor" if args.distractor else "fullwiki",
        "revise_bias": args.revise_bias,
        "use_layer3": use_l3, "use_layer4": use_l4, "use_layer5": use_l5,
        "exact_match": n_em / max(n, 1), "n_em": n_em,
        "mean_f1": sum_f1 / max(n, 1),
        "elapsed_s": round(dt, 1),
        "router": type(agent.orchestrator.router).__name__,
        "cost_model": repr(agent.orchestrator.router.cost_model),
        "calibrator": repr(agent.orchestrator.router.calibrator),
        # Which half of the routing parameters was fitted vs. declared. Without
        # this a log cannot say whether its thresholds meant anything.
        "calibration_config": routing.calibration_path,
        "cost_model_config": routing.cost_model_path,
        "identity_calibration": routing.is_identity_calibration,
        "routing_warnings": routing.warnings,
    }, indent=2))
    print(f"\nSaved details -> {args.out}")
    print(f"Saved run meta -> {meta_path}")


if __name__ == "__main__":
    main()
