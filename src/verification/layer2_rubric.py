from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.verification.json_extract import first_json_object
from src.verification.orchestrator import (
    Context, Cost, ErrorType, Layer, Signal, Step, Verdict,
)


ScoreFn = Callable[[str], str]

# See `layer4_llm_judge.RETRY_SUFFIX`: a byte-identical retry under deterministic
# decoding cannot produce a different reply, so the retry varies the prompt.
RETRY_SUFFIX = (
    "\n\nYour previous reply could not be parsed. Reply with EXACTLY one JSON "
    "object and nothing else: no prose, no repeated instructions, no second "
    "object."
)


GROUNDING = "observation_grounding"
GOAL      = "goal_relevance"
PROGRESS  = "progress"


@dataclass(frozen=True)
class Criterion:
    name: str
    max_score: int
    weight: float
    anchors: dict[int, str]


RUBRIC: tuple[Criterion, ...] = (
    Criterion(
        GOAL, 1, 1.0,
        {0: "step does not serve the question's goal",
         1: "step directly serves the goal"},
    ),
    Criterion(
        "thought_action_consistency", 3, 1.0,
        {1: "action contradicts the Thought",
         2: "action only loosely relates to the Thought",
         3: "action closely follows what the Thought just reasoned"},
    ),
    Criterion(
        GROUNDING, 3, 1.5,
        {1: "Thought fabricates/contradicts the previous Observation",
         2: "Thought is only partly based on the Observation",
         3: "Thought stays close to the facts in the Observation"},
    ),
    Criterion(
        PROGRESS, 1, 0.5,
        {0: "stalls/repeats a previous step",
         1: "moves the problem forward one step"},
    ),
    Criterion(
        "tool_appropriateness", 1, 0.5,
        {0: "chose the wrong tool for the situation",
         1: "chose the right tool (Search/Lookup/Finish)"},
    ),
)


FEEDBACK_TEMPLATES: dict[str, str] = {
    "goal_relevance":             "Tie this step directly to the question's goal.",
    "thought_action_consistency": "Choose an action that matches what the Thought just reasoned.",
    "observation_grounding":      "Reason only from the actual Observation; do not fabricate facts.",
    "progress":                   "Avoid repetition; choose a step that moves the problem forward.",
    "tool_appropriateness":       "Use the right tool (Search/Lookup/Finish) for the situation.",
}


def _build_feedback(weakest_name: str, step: Step) -> str:
    """Context-aware feedback for the weakest rubric criterion.

    The static `FEEDBACK_TEMPLATES` told the agent *that* it was wrong but not
    *what* to do — "Reason only from the actual Observation" with no hint of
    what the Observation actually says is not actionable. Including a snippet of
    the relevant context gives the agent something to correct toward.
    """
    if weakest_name == "observation_grounding":
        obs = (step.prev_observation or "").strip()
        if obs:
            snippet = obs[:150].rstrip()
            if len(obs) > 150:
                snippet += "…"
            return (f"Your Thought does not stay close to the Observation. "
                    f"The Observation says: \"{snippet}\". "
                    "Reason only from what the Observation actually states; "
                    "do not add facts that are not in it.")
        return FEEDBACK_TEMPLATES["observation_grounding"]

    if weakest_name == "goal_relevance":
        goal = (step.goal or "").strip()
        if goal:
            snippet = goal[:120].rstrip()
            return (f"This step does not serve the question: \"{snippet}\". "
                    "Redirect your action toward answering that question directly.")
        return FEEDBACK_TEMPLATES["goal_relevance"]

    if weakest_name == "progress":
        action = (step.action or "").strip()
        if action:
            return (f"This step stalls or repeats previous work (action: \"{action}\"). "
                    "Choose a different entity, keyword, or tool to move toward the answer.")
        return FEEDBACK_TEMPLATES["progress"]

    if weakest_name == "thought_action_consistency":
        thought = (step.thought or "").strip()[:100]
        action = (step.action or "").strip()
        if thought and action:
            return (f"The Action \"{action}\" does not match what the Thought reasoned "
                    f"(\"{thought}…\"). Emit an action that directly follows from your Thought.")
        return FEEDBACK_TEMPLATES["thought_action_consistency"]

    return FEEDBACK_TEMPLATES.get(weakest_name,
                                   "Review this step and choose a better action.")


def build_prompt(step: Step, rubric: tuple[Criterion, ...]) -> str:
    lines = [
        "You are a strict grader for ONE step of a ReAct agent.",
        "Grade the step on each criterion using the exact integer anchors below.",
        "",
        f"Goal: {step.goal or '(unspecified)'}",
        f"Previous Observation: {step.prev_observation or '(none)'}",
        f"Thought: {step.thought or '(none)'}",
        f"Action: {step.action}",
        "",
        "Criteria (score = integer within the shown range):",
    ]
    for c in rubric:
        anchors = "; ".join(f"{lvl}={desc}" for lvl, desc in sorted(c.anchors.items()))
        lines.append(f"- {c.name} [{min(c.anchors)}..{c.max_score}]: {anchors}")
    keys = ", ".join(f'"{c.name}": <int>' for c in rubric)
    lines += [
        "",
        "Respond with ONLY a JSON object, no prose:",
        "{" + keys + "}",
    ]
    return "\n".join(lines)


def parse_scores(text: str, rubric: tuple[Criterion, ...]) -> dict[str, int] | None:
    # Filtering on the rubric's own criterion names is what lets the right object
    # be picked out of a reply that contains several.
    data = first_json_object(text, tuple(c.name for c in rubric))
    if data is None:
        return None

    scores: dict[str, int] = {}
    for c in rubric:
        if c.name not in data:
            return None
        try:
            v = int(round(float(data[c.name])))
        except (TypeError, ValueError):
            return None
        scores[c.name] = max(min(c.anchors), min(c.max_score, v))
    return scores


def aggregate(norms: dict[str, float], weights: dict[str, float],
              mode: str = "weighted_min") -> float:
    names = list(norms)
    if mode == "weighted_sum":
        wsum = sum(weights[n] for n in names)
        return sum(weights[n] * norms[n] for n in names) / wsum if wsum else 0.0
    if mode != "weighted_min":
        raise ValueError(f"unsupported aggregation: {mode}")
    max_w = max(weights[n] for n in names) or 1.0
    worst_pull = max((weights[n] / max_w) * (1.0 - norms[n]) for n in names)
    return 1.0 - worst_pull


def diagnose(norms: dict[str, float], concern: float = 0.5) -> ErrorType:
    if norms.get(GROUNDING, 1.0) < concern:
        return ErrorType.FACTUAL
    if norms.get(GOAL, 1.0) < concern or norms.get(PROGRESS, 1.0) < concern:
        return ErrorType.REASONING
    return ErrorType.NONE


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Layer2RubricVerifier(Layer):

    def __init__(self, score_fn: ScoreFn, rubric: tuple[Criterion, ...] = RUBRIC,
                 aggregation: str = "weighted_min", pass_above: float = 0.75,
                 fail_below: float = 0.25, max_parse_retries: int = 1,
                 fallback_confidence: float = 0.5):
        assert 0.0 <= fail_below <= pass_above <= 1.0
        super().__init__(layer_id=2)
        self.score_fn = score_fn
        self.rubric = rubric
        self.aggregation = aggregation
        self.pass_above = pass_above
        self.fail_below = fail_below
        self.max_parse_retries = max_parse_retries
        self.fallback_confidence = fallback_confidence

    def run(self, step: Step, context: Context) -> Signal:
        prompt = build_prompt(step, self.rubric)

        t0 = time.perf_counter()
        tokens, n_calls, scores = 0, 0, None
        for attempt in range(self.max_parse_retries + 1):
            sent = prompt if attempt == 0 else prompt + RETRY_SUFFIX
            raw_text = self.score_fn(sent)
            n_calls += 1
            tokens += _est_tokens(sent) + _est_tokens(raw_text)
            scores = parse_scores(raw_text, self.rubric)
            if scores is not None:
                break
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if scores is None:
            return Signal(Verdict.UNSURE, self.fallback_confidence, ErrorType.NONE,
                          Cost(tokens, latency_ms),
                          raw={"parse_ok": False, "n_calls": n_calls})

        norms = {c.name: scores[c.name] / c.max_score for c in self.rubric}
        weights = {c.name: c.weight for c in self.rubric}
        q = aggregate(norms, weights, self.aggregation)

        confidence = q
        if q >= self.pass_above:
            verdict = Verdict.PASS
        elif q <= self.fail_below:
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.UNSURE

        suspected = diagnose(norms)
        weakest = min(self.rubric, key=lambda c: norms[c.name])

        raw = {
            "parse_ok": True,
            "scores": scores,
            "normalized": norms,
            "aggregate": q,
            "aggregation": self.aggregation,
            "weakest": weakest.name,
            "n_calls": n_calls,
            "feedback": (_build_feedback(weakest.name, step)
                         if verdict != Verdict.PASS else None),
        }
        return Signal(verdict, confidence, suspected, Cost(tokens, latency_ms), raw)


def make_mock_score_fn(scores: dict[str, int], *, wrap: bool = False) -> ScoreFn:
    payload = json.dumps(scores)
    body = (f"Here is my grading:\n```json\n{payload}\n```\nDone."
            if wrap else payload)

    def _fn(_prompt: str) -> str:
        return body

    return _fn


def _show(title: str, sig: Signal) -> None:
    print(f"----- {title} -----")
    print(f"  verdict={sig.verdict.value}  confidence={sig.confidence:.3f}  "
          f"suspect={sig.suspected_error_type.value}  tokens={sig.cost.tokens}")
    if sig.raw.get("parse_ok"):
        print(f"  aggregate(q)={sig.raw['aggregate']:.3f} "
              f"[{sig.raw['aggregation']}]  weakest={sig.raw['weakest']}")
    if sig.raw.get("feedback"):
        print(f"  feedback: {sig.raw['feedback']}")
    print()


def _demo() -> None:
    step = Step(
        action="Search[High Plains (United States)]",
        thought="High Plains rise from around 1,800 to 7,000 ft.",
        goal="Find the elevation range of the area the eastern sector extends into.",
        prev_observation="High Plains refers to one of two distinct land regions.",
        step_index=4,
    )

    good = make_mock_score_fn({"goal_relevance": 1, "thought_action_consistency": 3,
                               "observation_grounding": 3, "progress": 1,
                               "tool_appropriateness": 1})
    _show("(1) good step → PASS", Layer2RubricVerifier(good).run(step, Context()))

    weak_scores = {"goal_relevance": 1, "thought_action_consistency": 3,
                   "observation_grounding": 1, "progress": 1, "tool_appropriateness": 1}
    weak = make_mock_score_fn(weak_scores, wrap=True)
    _show("(2) weak grounding → UNSURE + FACTUAL",
          Layer2RubricVerifier(weak).run(step, Context()))

    _show("(2b) same scores, weighted_sum",
          Layer2RubricVerifier(make_mock_score_fn(weak_scores),
                               aggregation="weighted_sum").run(step, Context()))

    broken: ScoreFn = lambda _p: "Sorry, I can't produce JSON."
    _show("(3) parse error → UNSURE",
          Layer2RubricVerifier(broken).run(step, Context()))

    from src.verification.orchestrator import (
        FixedThresholdRouter, MockLayer, Orchestrator,
    )
    from src.verification.layer1_rule import Layer1Adapter
    layers = {
        1: Layer1Adapter(pass_confidence=0.6),
        2: Layer2RubricVerifier(weak),
        3: MockLayer(3, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90))),
        4: MockLayer(4, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200))),
    }
    res = Orchestrator(layers, FixedThresholdRouter(0.3, 0.8)).verify(step)
    print("----- (4) end-to-end via Orchestrator -----")
    for rec in res.signal_history:
        s = rec.signal
        print(f"  L{rec.layer_id}: {s.verdict.value:<6} conf={s.confidence:.2f} "
              f"suspect={s.suspected_error_type.value}")
    print(f"  -> decision={res.decision.value}  layers_run={res.layers_run}  "
          f"fused(src=L{res.fused.source_layer_id}, ps={res.fused.pass_score:.2f})  "
          f"cost={res.total_cost.tokens} tok")


if __name__ == "__main__":
    _demo()
