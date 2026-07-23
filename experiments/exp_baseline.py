from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv() -> None:
    import os
    here = Path(__file__).resolve().parent
    for env_path in (here.parent / ".env", here / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def add_distractor_arg(ap) -> None:
    """Shared by both experiments so the two sides of a comparison cannot end up
    on different retrieval settings."""
    ap.add_argument("--distractor", action="store_true",
                    help="serve the paragraphs HotpotQA's distractor setting "
                         "ships with each question (2 gold + 8 distractors) "
                         "instead of live Wikipedia. Removes the retrieval "
                         "bottleneck: in fullwiki, 43%% of the n30 questions "
                         "never retrieved the gold answer at all, which no "
                         "pre-hoc step verifier can fix. Requires a "
                         "hotpot_dev_distractor.json --data file.")


def check_distractor_data(args, examples) -> None:
    """Fail loudly rather than quietly running a half-broken retrieval setting.

    Presence of a `context` field is *not* sufficient: the fullwiki dev file also
    carries one, holding paragraphs an IR system retrieved, and only 28% of those
    contain both gold titles. Pointing `--distractor` at it would serve evidence
    that usually cannot answer the question and read as an agent failure. What
    identifies the distractor set is that the gold titles are always present, so
    that is what gets checked.
    """
    if not getattr(args, "distractor", False):
        return
    if not examples:
        return
    ok = 0
    for ex in examples:
        gold = {t for t, _ in (ex.get("supporting_facts") or [])}
        titles = {t for t, _ in (ex.get("context") or [])}
        if gold and gold <= titles:
            ok += 1
    frac = ok / len(examples)
    if frac < 0.95:
        sys.exit(
            f"--distractor expects the distractor dev set, but only {ok}/"
            f"{len(examples)} examples in {args.data} have all gold titles in "
            f"their 'context' ({frac:.0%}; the distractor set is 100%, fullwiki "
            f"is ~28%). Run: python experiments/download_data.py --split dev "
            f"--config distractor  ->  data/hotpotqa/hotpot_dev_distractor.json")


def build_llm(args):
    import os
    if args.backend == "ollama":
        from src.llm import OllamaLLM
        # The default 120s is comfortable for an 8B model (~2.5s/call measured)
        # and not for a 32B/70B one: the FIRST call also pays for loading ~20-40GB
        # of weights, and a 512-token completion at ~10 tok/s is another ~50s. A
        # timeout mid-run reads as a question that errored, which the summary then
        # reports as a NOT-comparable partial run.
        return OllamaLLM(model=args.model,
                         base_url=args.base_url or "http://localhost:11434",
                         max_tokens=512, temperature=0.0,
                         timeout=getattr(args, "llm_timeout", 120))
    if args.backend == "openai":
        from src.llm import OpenAICompatLLM
        if not args.base_url:
            sys.exit("--backend openai requires --base-url (e.g. https://api.groq.com/openai/v1)")
        return OpenAICompatLLM(base_url=args.base_url, model=args.model,
                               api_key=os.environ.get(args.key_env, ""),
                               max_tokens=512, temperature=0.0,
                               tpm_limit=getattr(args, "tpm", None) or None)
    if args.backend == "anthropic":
        from src.llm import AnthropicLLM
        return AnthropicLLM(model=args.model, max_tokens=512, temperature=0.0)
    sys.exit(f"unsupported backend: {args.backend}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="path to hotpot_dev_fullwiki.json")
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
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to sleep between questions (avoid rate-limit)")
    ap.add_argument("--tpm", type=int, default=0,
                    help="account tokens-per-minute limit; paces requests to stay "
                         "under it instead of discovering it via 429s "
                         "(Groq free tier: 12000). 0 = no pacing")
    ap.add_argument("--out", default="results/logs/baseline.json")
    ap.add_argument("--resume", action="store_true",
                    help="reuse questions already present in --out and only run "
                         "the missing ones (results are flushed per question)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--layer1", action="store_true",
                    help="enable Layer 1 rule-based verification (~0 cost)")
    add_distractor_arg(ap)
    args = ap.parse_args()

    from src.agents.base_react import ReActAgent
    from src.tools.wikipedia import MediaWikiBackend, WikiEnv
    from src.benchmarks.hotpotqa import load_examples
    from src.evaluation.metrics import exact_match, f1_score

    examples = load_examples(args.data, args.n, args.seed)
    check_distractor_data(args, examples)
    layers = "baseline" + ("+layer1" if args.layer1 else "")
    print(f"Running Vanilla ReAct [{layers}] on {len(examples)} questions | "
          f"backend={args.backend} model={args.model} "
          f"retrieval={'distractor' if args.distractor else 'fullwiki'}\n")

    llm = build_llm(args)
    dbackend = None
    if args.distractor:
        from src.tools.wikipedia import DistractorBackend
        dbackend = DistractorBackend()
        env = WikiEnv(dbackend)
    else:
        env = WikiEnv(MediaWikiBackend())

    verifier = None
    if args.layer1:
        from src.verification.layer1_rule import RuleBasedVerifier
        verifier = RuleBasedVerifier()

    agent = ReActAgent(llm, env, max_steps=args.max_steps, verbose=args.verbose,
                       verifier=verifier)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, n_em, sum_f1 = [], 0, 0.0
    # Flushed per question so an interrupted run keeps what it scored; --resume
    # then runs only the missing questions. See exp_adaptive for the same wiring.
    if args.resume and out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        n_em = sum(r["em"] for r in results)
        sum_f1 = sum(r.get("f1") or 0.0 for r in results)
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
        results.append({
            "id": ex["_id"],
            "question": ex["question"],
            "gold": ex.get("answer"),
            "prediction": res.prediction,
            "em": em,
            "f1": f1,
            "n_steps": res.n_steps,
            "n_llm_calls": res.n_llm_calls,
            "finished": res.finished,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "layer1_violations": res.layer1_violations,
            "trajectory": res.trajectory,
        })
        flush()
        viol_str = f" viol={res.layer1_violations}" if args.layer1 else ""
        print(f"[{k}/{len(examples)}] EM={em} pred={res.prediction!r} "
              f"gold={ex.get('answer')!r} steps={res.n_steps} "
              f"calls={res.n_llm_calls}{viol_str}")

    dt = time.time() - t0
    # A run where every question errored used to exit 0 and write `[]`, which
    # reads downstream as "EM 0.000 over 0 questions" rather than "the run never
    # happened". Fail loudly instead of leaving an empty log behind.
    if not results:
        sys.exit(f"\nAll {len(examples)} questions errored — nothing scored, no log "
                 f"written. See the errors above (a quota wall is the usual cause).")
    if len(results) < len(examples):
        print(f"\nWARNING: only {len(results)}/{len(examples)} questions scored; "
              f"the rest errored. This log is NOT comparable to a complete run.")

    tot_in   = sum(r["prompt_tokens"] for r in results)
    tot_out  = sum(r["completion_tokens"] for r in results)
    tot_viol = sum(r["layer1_violations"] for r in results)
    n = len(results)
    print(f"\n================= SUMMARY [{layers.upper()}] =================")
    print(f"Questions scored      : {n}")
    print(f"Exact Match           : {n_em}/{n} = {n_em / max(n, 1):.3f}")
    print(f"Mean F1               : {sum_f1 / max(n, 1):.3f}")
    print(f"Average LLM calls     : {sum(r['n_llm_calls'] for r in results)/max(n,1):.2f}")
    print(f"Average steps         : {sum(r['n_steps'] for r in results)/max(n,1):.2f}")
    print(f"Total tokens (in/out) : {tot_in}/{tot_out}")
    if args.layer1:
        print(f"Layer1 violations     : {tot_viol} total | "
              f"{tot_viol/max(n,1):.2f} average/question")
    print(f"Time                  : {dt:.1f}s")

    flush()
    # Sibling meta, same shape as exp_adaptive's. A baseline log with no model
    # recorded cannot be paired with an adaptive log — that is what made the
    # earlier n=5/n=15 comparisons unattributable.
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "experiment": "baseline", "tag": layers,
        "backend": args.backend, "model": args.model,
        "base_url": args.base_url, "data": args.data,
        "n_requested": args.n, "n_scored": n, "seed": args.seed,
        "max_steps": args.max_steps, "layer1": args.layer1,
        "retrieval": "distractor" if args.distractor else "fullwiki",
        "exact_match": n_em / max(n, 1), "n_em": n_em,
        "mean_f1": sum_f1 / max(n, 1),
        "layer1_violations": tot_viol,
        "elapsed_s": round(dt, 1),
    }, indent=2))
    print(f"\nSaved details -> {args.out}")
    print(f"Saved run meta -> {meta_path}")
    print("(EM here is raw answer-EM; the official eval also includes supporting-facts/joint.)")


if __name__ == "__main__":
    main()
