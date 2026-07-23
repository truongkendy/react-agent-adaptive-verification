from __future__ import annotations

import json
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


JudgeFn = Callable[[str], str]

# Appended on a parse retry. The retry used to resend the prompt byte for byte,
# and decoding is deterministic (Ollama's temperature defaults to 0.0), so the
# reply came back byte-identical: every parse failure was billed twice for a
# second attempt that could not possibly differ. Varying the input is what makes
# the retry a retry.
RETRY_SUFFIX = (
    "\n\nYour previous reply could not be parsed. Reply with EXACTLY one JSON "
    "object and nothing else: no prose, no repeated instructions, no second "
    "object."
)


def build_prompt(step: Step) -> str:
    return "\n".join([
        "You are a careful judge of the REASONING in ONE step of a ReAct agent.",
        "Ignore surface format and factual lookups; judge only whether the",
        "Thought follows logically from the previous Observation and whether the",
        "Action is a logically sound next move toward the goal.",
        "",
        f"Goal: {step.goal or '(unspecified)'}",
        f"Previous Observation: {step.prev_observation or '(none)'}",
        f"Thought: {step.thought or '(none)'}",
        f"Action: {step.action}",
        "",
        "Flag a reasoning error if the Thought is a non-sequitur, draws an",
        "invalid inference, contradicts itself, or the Action does not follow",
        "from the Thought / drifts away from the goal.",
        "",
        "Respond with ONLY a JSON object, no prose:",
        '{"reasoning_sound": <true|false>, '
        '"confidence": <float 0..1>, '
        '"rationale": "<one short sentence explaining your verdict>", '
        '"feedback": "<if not sound: one imperative sentence telling the agent '
        'what concrete action to take instead (e.g. Search[X] or Lookup[Y]); '
        'leave empty string if reasoning is sound>"}',
        "confidence = how sure you are that your reasoning_sound judgement is correct.",
    ])


def parse_judgement(text: str) -> dict | None:
    data = first_json_object(text, ("reasoning_sound",))
    if data is None:
        return None

    sound = data["reasoning_sound"]
    if isinstance(sound, str):
        sound = sound.strip().lower() in ("true", "yes", "1", "sound", "pass")
    else:
        sound = bool(sound)

    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    rationale = data.get("rationale", "")
    feedback = data.get("feedback", "")
    return {"reasoning_sound": sound, "confidence": conf,
            "rationale": str(rationale) if rationale is not None else "",
            "feedback": str(feedback).strip() if feedback else ""}


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Layer4LLMJudge(Layer):

    def __init__(self, judge_fn: JudgeFn, pass_above: float = 0.75,
                 fail_below: float = 0.25, max_parse_retries: int = 1,
                 fallback_confidence: float = 0.5):
        assert 0.0 <= fail_below <= pass_above <= 1.0
        super().__init__(layer_id=4)
        self.judge_fn = judge_fn
        self.pass_above = pass_above
        self.fail_below = fail_below
        self.max_parse_retries = max_parse_retries
        self.fallback_confidence = fallback_confidence

    def applicable(self, step: Step, context: Context) -> bool:
        """The judge is asked whether the Thought *follows from the previous
        Observation*. On step 1 there is no previous observation, so the honest
        answer is unavailable — but the model answers anyway, and reliably
        complains that the Thought is unsupported ("Reason only from the actual
        Observation; do not fabricate facts."). That is an artefact of the prompt,
        not a property of the step."""
        return bool(step.prev_observation.strip())

    def run(self, step: Step, context: Context) -> Signal:
        prompt = build_prompt(step)

        t0 = time.perf_counter()
        tokens, n_calls, parsed = 0, 0, None
        for attempt in range(self.max_parse_retries + 1):
            sent = prompt if attempt == 0 else prompt + RETRY_SUFFIX
            raw_text = self.judge_fn(sent)
            n_calls += 1
            tokens += _est_tokens(sent) + _est_tokens(raw_text)
            parsed = parse_judgement(raw_text)
            if parsed is not None:
                break
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if parsed is None:
            return Signal(Verdict.UNSURE, self.fallback_confidence, ErrorType.NONE,
                          Cost(tokens, latency_ms),
                          raw={"parse_ok": False, "n_calls": n_calls})

        c = parsed["confidence"]
        p_sound = c if parsed["reasoning_sound"] else (1.0 - c)

        if p_sound >= self.pass_above:
            verdict, suspected = Verdict.PASS, ErrorType.NONE
        elif p_sound <= self.fail_below:
            verdict, suspected = Verdict.FAIL, ErrorType.REASONING
        else:
            verdict, suspected = Verdict.UNSURE, ErrorType.REASONING

        raw = {
            "parse_ok": True,
            "reasoning_sound": parsed["reasoning_sound"],
            "judge_confidence": c,
            "rationale": parsed["rationale"],
            "n_calls": n_calls,
        }
        # Only surface `feedback` when the step is not passing — a feedback
        # string on a PASS would be picked up by `_feedback()` even though the
        # step was accepted, confusing the complainer selection logic.
        if verdict is not Verdict.PASS and parsed.get("feedback"):
            raw["feedback"] = parsed["feedback"]
        return Signal(verdict, p_sound, suspected, Cost(tokens, latency_ms), raw)


def make_mock_judge_fn(reasoning_sound: bool, confidence: float = 0.9,
                       rationale: str = "", *, wrap: bool = False) -> JudgeFn:
    payload = json.dumps({"reasoning_sound": reasoning_sound,
                          "confidence": confidence, "rationale": rationale})
    body = (f"Here is my judgement:\n```json\n{payload}\n```\nDone."
            if wrap else payload)

    def _fn(_prompt: str) -> str:
        return body

    return _fn


def _show(title: str, sig: Signal) -> None:
    print(f"----- {title} -----")
    print(f"  verdict={sig.verdict.value}  confidence(raw)={sig.confidence:.3f}  "
          f"suspect={sig.suspected_error_type.value}  tokens={sig.cost.tokens}")
    if sig.raw.get("parse_ok") and sig.raw.get("rationale"):
        print(f"  rationale: {sig.raw['rationale']}")
    print()


def _demo() -> None:
    step = Step(
        action="Finish[Chef]",
        thought="The passage lists his jobs but never says which he held longest, "
                "so the answer must be Chef.",
        goal="Determine which job the person held for the longest time.",
        prev_observation="He worked as a teacher, then a chef, then a writer.",
        step_index=6,
    )

    sound = make_mock_judge_fn(True, confidence=0.88,
                               rationale="Action follows from the Thought and serves the goal.")
    _show("(1) sound reasoning", Layer4LLMJudge(sound).run(step, Context()))

    bad = make_mock_judge_fn(False, confidence=0.82,
                             rationale="'so the answer must be Chef' is a non-sequitur.",
                             wrap=True)
    _show("(2) non-sequitur", Layer4LLMJudge(bad).run(step, Context()))

    broken: JudgeFn = lambda _p: "I think it's fine, honestly."
    _show("(3) unparseable reply", Layer4LLMJudge(broken).run(step, Context()))

    from src.verification.orchestrator import (
        Cost, ErrorType, FixedThresholdRouter, MockLayer, Orchestrator,
        Signal, Verdict,
    )
    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.UNSURE, 0.5, ErrorType.REASONING, Cost(120, 20))),
        3: MockLayer(3, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90))),
        4: Layer4LLMJudge(bad),
    }
    res = Orchestrator(layers, FixedThresholdRouter(0.3, 0.8)).verify(step)
    print("----- (4) end-to-end via Orchestrator -----")
    for rec in res.signal_history:
        s = rec.signal
        print(f"  L{rec.layer_id}: {s.verdict.value:<6} conf={s.confidence:.2f} "
              f"suspect={s.suspected_error_type.value}")
    print(f"  -> decision={res.decision.value}  layers_run={res.layers_run}  "
          f"trigger={res.triggering_error_type.value if res.triggering_error_type else '-'}  "
          f"fused(src=L{res.fused.source_layer_id}, ps={res.fused.pass_score:.2f})  "
          f"cost={res.total_cost.tokens} tok")


if __name__ == "__main__":
    _demo()
