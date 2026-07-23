import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.layer4_llm_judge import (
    Layer4LLMJudge, build_prompt, make_mock_judge_fn, parse_judgement,
)
from src.verification.orchestrator import (
    Context, Cost, ErrorType, FixedThresholdRouter, MockLayer, Orchestrator,
    Signal, Step, Verdict,
)


STEP = Step(action="Finish[Chef]",
            thought="It never says which job was longest, so it must be Chef.",
            goal="Which job did he hold longest?",
            prev_observation="He was a teacher, then a chef, then a writer.",
            step_index=6)


def test_parse() -> None:
    ok = parse_judgement('{"reasoning_sound": false, "confidence": 0.8, "rationale": "x"}')
    assert ok is not None
    assert ok["reasoning_sound"] is False
    assert ok["confidence"] == 0.8
    assert ok["rationale"] == "x"
    assert ok["feedback"] == ""   # no feedback key in input → empty string
    # With feedback field populated
    with_fb = parse_judgement('{"reasoning_sound": false, "confidence": 0.8, '
                               '"rationale": "x", "feedback": "Search[Y]"}')
    assert with_fb is not None and with_fb["feedback"] == "Search[Y]"
    wrapped = parse_judgement('sure:\n```json\n{"reasoning_sound":"yes","confidence":1.7}\n```')
    assert wrapped["reasoning_sound"] is True and wrapped["confidence"] == 1.0
    assert parse_judgement("no json here") is None
    assert parse_judgement('{"confidence": 0.5}') is None
    print("  [A] parse_judgement OK")


def test_sound_pass() -> None:
    sig = Layer4LLMJudge(make_mock_judge_fn(True, 0.88)).run(STEP, Context())
    assert sig.verdict is Verdict.PASS
    assert sig.suspected_error_type is ErrorType.NONE
    assert abs(sig.confidence - 0.88) < 1e-9
    assert sig.raw["parse_ok"]
    print("  [B] sound reasoning → PASS, confidence = P(sound) OK")


def test_broken_fail() -> None:
    sig = Layer4LLMJudge(make_mock_judge_fn(False, 0.9, wrap=True)).run(STEP, Context())
    assert sig.verdict is Verdict.FAIL
    assert sig.suspected_error_type is ErrorType.REASONING
    assert abs(sig.confidence - 0.10) < 1e-9
    print("  [C] confident 'broken' → FAIL, low P(sound) OK")


def test_ambiguous_unsure() -> None:
    sig = Layer4LLMJudge(make_mock_judge_fn(False, 0.55)).run(STEP, Context())
    assert sig.verdict is Verdict.UNSURE
    assert sig.suspected_error_type is ErrorType.REASONING
    print("  [D] borderline judgement → UNSURE + REASONING OK")


def test_parse_failure_unsure() -> None:
    sig = Layer4LLMJudge(lambda _p: "no json", fallback_confidence=0.4).run(STEP, Context())
    assert sig.verdict is Verdict.UNSURE and sig.confidence == 0.4
    assert sig.raw["parse_ok"] is False and sig.raw["n_calls"] == 2
    print("  [E] unparseable reply → UNSURE fallback + retry OK")


def test_prompt_contains_step() -> None:
    p = build_prompt(STEP)
    assert "Finish[Chef]" in p and "longest" in p and "reasoning_sound" in p
    print("  [F] build_prompt includes step + schema OK")


def test_end_to_end_escalation() -> None:
    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.UNSURE, 0.5, ErrorType.REASONING, Cost(120, 20))),
        3: MockLayer(3, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90))),
        4: Layer4LLMJudge(make_mock_judge_fn(False, 0.9)),
    }
    res = Orchestrator(layers, FixedThresholdRouter(0.3, 0.8)).verify(STEP)
    assert 4 in res.layers_run
    assert res.decision.value == "fail_revise"
    assert res.triggering_error_type is ErrorType.REASONING
    l4 = next(r.signal for r in res.signal_history if r.layer_id == 4)
    assert abs(l4.raw["raw_confidence"] - 0.10) < 1e-9
    print("  [G] end-to-end escalation to L4 → fail_revise/reasoning OK")


if __name__ == "__main__":
    test_parse()
    test_sound_pass()
    test_broken_fail()
    test_ambiguous_unsure()
    test_parse_failure_unsure()
    test_prompt_contains_step()
    test_end_to_end_escalation()
    print("ALL CHECKS PASSED ✓")
