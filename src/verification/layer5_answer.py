from __future__ import annotations

import json
import re
import time
from typing import Callable

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.verification.json_extract import first_json_object
from src.verification.orchestrator import (
    Context, Cost, ErrorType, Layer, Signal, Step, Verdict,
)

GateFn = Callable[[str], str]

# Same reason as Layer 2 / Layer 4: resending an identical prompt to a
# temperature-0 backend returns an identical reply, so a retry has to vary its
# input to be a retry at all.
RETRY_SUFFIX = (
    "\n\nYour previous reply could not be parsed. Reply with EXACTLY one JSON "
    "object and nothing else: no prose, no repeated instructions, no second "
    "object."
)

_FINISH_PAT = re.compile(r"^\s*finish\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)

# The evidence pool is the whole trajectory's observations, which is far larger
# than one step's prompt. Truncated per observation (keeping the head, where the
# intro extract puts the answer) and capped in count, so the gate stays roughly
# one Layer-4-sized call rather than growing with trajectory length.
MAX_EVIDENCE = 6
MAX_EVIDENCE_CHARS = 700

# Answers that concede failure. These are the ones actually observed being
# committed: `Finish['unknown']`, `Finish['none']`, `Finish['I do not know']`
# across results/logs/*_n30_*. They need no LLM to reject, and rejecting them
# costs nothing, so they are checked before the gate call is made.
_NULL_ANSWERS = frozenset({
    "", "none", "unknown", "null", "n/a", "na", "nothing", "no answer",
    "not found", "not available", "not specified", "not mentioned",
    "i do not know", "i don't know", "dont know", "cannot determine",
    "can not determine", "cannot be determined", "undetermined", "unclear",
    "insufficient information", "no information",
})
# Yes/no questions are the one case where the answer legitimately does not
# appear in the retrieved text, so the grounding check below must skip them.
_BOOLEAN_ANSWERS = frozenset({"yes", "no", "true", "false"})


def finish_argument(action: str) -> str | None:
    """The proposed answer inside `Finish[...]`, or None if this is not a Finish."""
    m = _FINISH_PAT.match(action or "")
    return m.group(1).strip() if m else None


def _evidence(step: Step, context: Context, trajectory_key: str) -> list[str]:
    pool = list(context.scratch.get(trajectory_key) or [])
    if step.prev_observation.strip() and step.prev_observation not in pool:
        pool.append(step.prev_observation)
    # Keep the most recent observations: the answer-bearing page is normally the
    # one just retrieved, and the early ones are search-result listings.
    pool = [p for p in pool if p and p.strip()][-MAX_EVIDENCE:]
    return [p.strip()[:MAX_EVIDENCE_CHARS] for p in pool]


def is_grounded(answer: str, evidence: list[str]) -> bool:
    """Does the proposed answer actually occur in the evidence the agent has?

    HotpotQA answers are extractive — 29 of the 30 questions in the seed-42
    distractor subset contain their gold answer verbatim in the served
    paragraphs, the 30th being a yes/no — so an answer that appears nowhere in
    any observation was not read off the evidence, it was invented. That is
    checkable without an LLM, and it has to be, because llama3.1 replied
    `supported: true, confidence: 1.0` to `Finish[none]` on question 5ac3e0f7:
    the model's own boolean is not the thing to trust here."""
    a = _norm(answer)
    if not a or a in _BOOLEAN_ANSWERS:
        return True          # nothing to ground, or legitimately not in the text
    pool = _norm(" ".join(evidence))
    return a in pool


def build_prompt(step: Step, evidence: list[str], answer: str) -> str:
    ev = "\n".join(f"[{i}] {e}" for i, e in enumerate(evidence, 1)) or "(none)"
    return "\n".join([
        "An agent is about to commit a FINAL ANSWER. Decide whether the evidence",
        "gathered so far actually supports it.",
        "",
        f"Question: {step.goal or '(unspecified)'}",
        f"Proposed final answer: {answer or '(empty)'}",
        "",
        "Evidence gathered so far:",
        ev,
        "",
        "Judge ONLY against the evidence above. Do not use outside knowledge.",
        "If the evidence supports a DIFFERENT answer, put that answer in the",
        '"answer" field, copied verbatim from the evidence. If the evidence does',
        'not answer the question at all, leave "answer" empty.',
        "",
        "Respond with ONLY a JSON object, no prose:",
        '{"supported": <true|false>, '
        '"confidence": <float 0..1>, '
        '"quote": "<the exact sentence from the evidence that settles it>", '
        '"answer": "<answer the evidence supports, or empty>"}',
        "confidence = how sure you are that your supported judgement is correct.",
        'The "quote" must be copied verbatim from the evidence above.',
    ])


def parse_gate(text: str) -> dict | None:
    data = first_json_object(text, ("supported",))
    if data is None:
        return None

    sup = data["supported"]
    if isinstance(sup, str):
        sup = sup.strip().lower() in ("true", "yes", "1", "supported", "pass")
    else:
        sup = bool(sup)

    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    ans = data.get("answer") or ""
    if not isinstance(ans, str):
        ans = str(ans)
    ans = ans.strip()
    if ans.lower() in _NULL_ANSWERS or ans.lower() == "empty":
        ans = ""

    quote = data.get("quote") or ""
    if not isinstance(quote, str):
        quote = str(quote)

    return {"supported": sup, "confidence": conf, "answer": ans,
            "quote": quote.strip()}


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def repair_feedback(goal: str, proposed: str, supported_answer: str) -> str:
    """The whole point of this layer: hand the agent something it can act on.

    Measured on `adaptive_n30_ollama_steps12.json`, verification blocked a
    premature `Finish[...]` 12 times, every one of them a wrong answer — and not
    one of the 12 turned into a correct answer, because the injected feedback was
    the generic "The Thought does not follow from the previous Observation". The
    agent was told the step was bad, never what to do instead. A block that names
    the answer the evidence actually supports, or names the missing fact, is the
    only kind that can change an outcome."""
    if supported_answer and _norm(supported_answer) != _norm(proposed):
        return (f"The evidence gathered so far supports \"{supported_answer}\", "
                f"not \"{proposed}\". If \"{supported_answer}\" answers the "
                f"question, emit Finish[{supported_answer}] now.")
    return ("The evidence gathered so far does not answer the question "
            f"\"{goal}\". Do not finish yet — the missing fact is not in any "
            "observation. Retrieve it first: Search a different entity named in "
            "the question, or Lookup a keyword inside the page already open.")


def null_answer_feedback(goal: str) -> str:
    return ("You are about to answer that you do not know, which scores zero. "
            f"The question is \"{goal}\". Do not finish — find the fact first: "
            "Search[<an entity named in the question>], then "
            "Lookup[<the attribute the question asks for>] inside that page.")


def ungrounded_feedback(goal: str, proposed: str) -> str:
    return (f"\"{proposed}\" does not appear anywhere in the evidence you have "
            "retrieved, so it is not supported — do not commit it. Either "
            "Lookup[<keyword>] to find the answer in the page already open, or "
            "Search[<the other entity named in the question>].")


class Layer5AnswerGate(Layer):
    """Post-hoc answer-sufficiency gate on `Finish[...]`.

    Every other layer is *pre-hoc*: it judges a step before that step's own
    observation exists, so none of them can catch a wrong answer read off
    evidence that is already in hand. Measured on the `n30_ollama` fullwiki pair,
    adaptive's 20 failures split into 13 where the gold string never reached any
    observation (unreachable by step verification) and 7 where the evidence was
    present and went unused; on the distractor set — where all 30 questions have
    their gold paragraphs — baseline still scored only 11/30, so ~19 failures are
    of that second kind. This layer is the gate for them: it compares the
    accumulated evidence pool against the goal and the proposed answer, which is
    what Layer 3 does *not* do (L3 verifies claims inside the Thought).

    `confidence` is P(the proposed answer is supported) — pass_score-compatible
    with every other layer — and RAW: calibration belongs to the orchestrator.
    """

    def __init__(self, gate_fn: GateFn, pass_above: float = 0.75,
                 fail_below: float = 0.25, max_parse_retries: int = 1,
                 fallback_confidence: float = 0.5,
                 null_confidence: float = 0.02,
                 ungrounded_confidence: float = 0.10,
                 trajectory_key: str = "observations"):
        assert 0.0 <= fail_below <= pass_above <= 1.0
        super().__init__(layer_id=5)
        self.gate_fn = gate_fn
        self.pass_above = pass_above
        self.fail_below = fail_below
        self.max_parse_retries = max_parse_retries
        self.fallback_confidence = fallback_confidence
        # Deterministic branches, so these are near-certain rather than a
        # judgement: a self-declared non-answer cannot be right, and an answer
        # absent from every observation was not read off the evidence. Not 0.0 —
        # `is_grounded` normalizes away punctuation, so a legitimate answer
        # phrased differently from the source is possible.
        self.null_confidence = null_confidence
        self.ungrounded_confidence = ungrounded_confidence
        self.trajectory_key = trajectory_key

    def _deterministic(self, step: Step, answer: str,
                       evidence: list[str]) -> Signal | None:
        """Zero-token rejections. Returns None when the gate call is still needed."""
        if answer.strip().lower() in _NULL_ANSWERS:
            return Signal(Verdict.FAIL, self.null_confidence, ErrorType.ANSWER,
                          Cost(0, 0.0),
                          raw={"parse_ok": None, "deterministic": True,
                               "rule": "null_answer",
                               "proposed_answer": answer,
                               "n_evidence": len(evidence),
                               "feedback": null_answer_feedback(step.goal)})
        if not is_grounded(answer, evidence):
            return Signal(Verdict.FAIL, self.ungrounded_confidence,
                          ErrorType.ANSWER, Cost(0, 0.0),
                          raw={"parse_ok": None, "deterministic": True,
                               "rule": "ungrounded_answer",
                               "proposed_answer": answer,
                               "n_evidence": len(evidence),
                               "feedback": ungrounded_feedback(step.goal, answer)})
        return None

    def applicable(self, step: Step, context: Context) -> bool:
        """Only on a Finish, and only with something to check it against. On a
        Search/Lookup step there is no answer to gate, and with an empty evidence
        pool the honest verdict is unavailable — running anyway would block the
        agent's first answer on no information."""
        context = context or Context()
        if finish_argument(step.action) is None:
            return False
        return bool(_evidence(step, context, self.trajectory_key))

    def run(self, step: Step, context: Context) -> Signal:
        context = context or Context()
        answer = finish_argument(step.action) or ""
        evidence = _evidence(step, context, self.trajectory_key)

        # Two rejections that need no LLM, so they are made before paying for
        # one. Both were observed being committed in results/logs/*_n30_*, and
        # asking llama3.1 about them is worse than not asking: it answered
        # `supported: true, confidence: 1.0` to `Finish[none]`.
        det = self._deterministic(step, answer, evidence)
        if det is not None:
            return det

        prompt = build_prompt(step, evidence, answer)

        t0 = time.perf_counter()
        tokens, n_calls, parsed = 0, 0, None
        for attempt in range(self.max_parse_retries + 1):
            sent = prompt if attempt == 0 else prompt + RETRY_SUFFIX
            raw_text = self.gate_fn(sent)
            n_calls += 1
            tokens += _est_tokens(sent) + _est_tokens(raw_text)
            parsed = parse_gate(raw_text)
            if parsed is not None:
                break
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if parsed is None:
            return Signal(Verdict.UNSURE, self.fallback_confidence, ErrorType.NONE,
                          Cost(tokens, latency_ms),
                          raw={"parse_ok": False, "n_calls": n_calls,
                               "n_evidence": len(evidence)})

        c = parsed["confidence"]
        p_supported = c if parsed["supported"] else (1.0 - c)

        # A "supported" verdict whose quote is not in the evidence is not a
        # verdict about the evidence. Downgrade rather than reject outright: the
        # model may have paraphrased, so this is a loss of confidence, not proof
        # of a bad answer.
        quote_ok = True
        if parsed["supported"] and parsed["quote"]:
            quote_ok = is_grounded(parsed["quote"], evidence)
            if not quote_ok:
                p_supported = min(p_supported, self.pass_above - 1e-9)

        if p_supported >= self.pass_above:
            verdict, suspected = Verdict.PASS, ErrorType.NONE
        elif p_supported <= self.fail_below:
            verdict, suspected = Verdict.FAIL, ErrorType.ANSWER
        else:
            verdict, suspected = Verdict.UNSURE, ErrorType.ANSWER

        raw = {
            "parse_ok": True,
            "supported": parsed["supported"],
            "gate_confidence": c,
            "supported_answer": parsed["answer"],
            "proposed_answer": answer,
            "quote_grounded": quote_ok,
            "n_evidence": len(evidence),
            "n_calls": n_calls,
        }
        if verdict is not Verdict.PASS:
            raw["feedback"] = repair_feedback(step.goal, answer, parsed["answer"])
        return Signal(verdict, p_supported, suspected, Cost(tokens, latency_ms), raw)


def make_mock_gate_fn(supported: bool, confidence: float = 0.9,
                      answer: str = "", *, wrap: bool = False) -> GateFn:
    payload = json.dumps({"supported": supported, "confidence": confidence,
                          "answer": answer})
    body = (f"Here is my judgement:\n```json\n{payload}\n```\nDone."
            if wrap else payload)

    def _fn(_prompt: str) -> str:
        return body

    return _fn


def _show(title: str, sig: Signal) -> None:
    print(f"----- {title} -----")
    print(f"  verdict={sig.verdict.value}  confidence(raw)={sig.confidence:.3f}  "
          f"suspect={sig.suspected_error_type.value}  tokens={sig.cost.tokens}")
    if sig.raw.get("feedback"):
        print(f"  feedback: {sig.raw['feedback']}")
    print()


def _demo() -> None:
    ctx = Context(scratch={"observations": [
        "Stone Brewing is an American brewery headquartered in Escondido, "
        "California. It was founded in 1996.",
        "Escondido is a city in San Diego County, California.",
    ]})
    step = Step(action="Finish[Ballast Point]",
                thought="The brewery in Escondido must be Ballast Point.",
                goal="Which brewery is headquartered in Escondido, California?",
                prev_observation="Escondido is a city in San Diego County, California.",
                step_index=5, reversibility=0.2, task_stakes=0.9)

    _show("(1) answer contradicted, evidence names the right one",
          Layer5AnswerGate(make_mock_gate_fn(
              False, 0.9, "Stone Brewing")).run(step, ctx))

    _show("(2) evidence does not answer the question at all",
          Layer5AnswerGate(make_mock_gate_fn(False, 0.85, "", wrap=True)).run(step, ctx))

    ok = Step(**{**vars(step), "action": "Finish[Stone Brewing]"})
    _show("(3) supported answer passes",
          Layer5AnswerGate(make_mock_gate_fn(True, 0.92, "Stone Brewing")).run(ok, ctx))

    gate = Layer5AnswerGate(make_mock_gate_fn(True))
    print("applicable on Search step  :", gate.applicable(
        Step(action="Search[Escondido]", goal="q"), ctx))
    # prev_observation counts as evidence, so "no pool" means a step 1 Finish.
    print("applicable, no evidence at all:", gate.applicable(
        Step(action="Finish[Ballast Point]", goal=step.goal, step_index=1), Context()))
    print("applicable on Finish+pool  :", gate.applicable(step, ctx))


if __name__ == "__main__":
    _demo()
