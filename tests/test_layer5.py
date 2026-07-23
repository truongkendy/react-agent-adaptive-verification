"""Layer 5 (post-hoc answer gate) + the mandatory-gate wiring + the revision
repeat penalty.

Every check here corresponds to something measured in `results/logs/`:
  - the gate must run on a Finish step even when the cheap layers already
    returned a decisive PASS (43/303 decisions on the distractor run were an L2
    PASS at confidence 1.0, which stopped the cascade before any specialist);
  - a block must carry an actionable repair, not a verdict (12/12 blocked
    Finishes in adaptive_n30_ollama_steps12.json were wrong answers and 0/12
    became correct answers);
  - repeated revisions must get progressively more expensive (one trajectory
    absorbed five consecutive blocks and finished with no answer).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verification.cost_model import default_cost_model
from src.verification.layer5_answer import (
    Layer5AnswerGate, finish_argument, is_grounded, make_mock_gate_fn,
    parse_gate, repair_feedback,
)
from src.verification.orchestrator import (
    Action, AdaptiveThresholdRouter, Context, Cost, Decision, ErrorType,
    MockLayer, Orchestrator, Signal, SignalRecord, Step, Verdict, _State, fuse,
)

GOAL = "Which brewery is headquartered in Escondido, California?"
POOL = ["Stone Brewing is an American brewery headquartered in Escondido, California.",
        "Escondido is a city in San Diego County, California."]


def _step(action: str) -> Step:
    return Step(action=action, thought="The brewery must be the one I saw.",
                goal=GOAL, prev_observation=POOL[-1], step_index=5,
                reversibility=0.2, task_stakes=0.9)


def _ctx(pool=None) -> Context:
    return Context(scratch={"observations": list(POOL if pool is None else pool)})


def test_finish_argument() -> None:
    assert finish_argument("Finish[Stone Brewing]") == "Stone Brewing"
    assert finish_argument("  finish[ 42 ] ") == "42"
    assert finish_argument("Search[Escondido]") is None
    assert finish_argument("") is None
    print("  [A] finish_argument extracts the proposed answer OK")


def test_applicability() -> None:
    g = Layer5AnswerGate(make_mock_gate_fn(True))
    assert not g.applicable(_step("Search[Escondido]"), _ctx()), "not a Finish"
    assert not g.applicable(
        Step(action="Finish[X]", goal=GOAL, step_index=1), Context()), "no evidence"
    assert g.applicable(_step("Finish[Stone Brewing]"), _ctx())
    print("  [B] gate applies only to Finish with a non-empty evidence pool OK")


def test_parse_and_banding() -> None:
    assert parse_gate("not json at all") is None
    assert parse_gate('{"supported": "yes", "confidence": 2, "answer": "unknown"}') == {
        "supported": True, "confidence": 1.0, "answer": "", "quote": ""}

    ok = Layer5AnswerGate(make_mock_gate_fn(True, 0.92, "Stone Brewing"))
    sig = ok.run(_step("Finish[Stone Brewing]"), _ctx())
    assert sig.verdict is Verdict.PASS and abs(sig.confidence - 0.92) < 1e-9
    assert sig.suspected_error_type is ErrorType.NONE
    assert "feedback" not in (sig.raw or {}), "a PASS must not carry repair text"

    # "San Diego County" IS in the pool, so it reaches the LLM gate.
    bad = Layer5AnswerGate(make_mock_gate_fn(False, 0.9, "Stone Brewing", wrap=True))
    sig = bad.run(_step("Finish[San Diego County]"), _ctx())
    assert sig.verdict is Verdict.FAIL, sig
    assert abs(sig.confidence - 0.1) < 1e-9, "confidence is P(answer supported)"
    assert sig.suspected_error_type is ErrorType.ANSWER
    assert (sig.raw or {}).get("parse_ok") is True and sig.cost.tokens > 0

    broken = Layer5AnswerGate(lambda _p: "honestly it looks fine")
    sig = broken.run(_step("Finish[San Diego County]"), _ctx())
    assert sig.verdict is Verdict.UNSURE and sig.confidence == 0.5
    assert (sig.raw or {}).get("parse_ok") is False, "parse failure must be logged"
    print("  [C] parse + PASS/FAIL/UNSURE banding, raw confidence = P(supported) OK")


def test_feedback_is_actionable() -> None:
    named = repair_feedback(GOAL, "Ballast Point", "Stone Brewing")
    assert "Stone Brewing" in named and "Finish[Stone Brewing]" in named, named
    missing = repair_feedback(GOAL, "Ballast Point", "")
    assert "Search" in missing and "Lookup" in missing, missing
    # An answer that only differs in surface form is not a different answer.
    same = repair_feedback(GOAL, "Stone Brewing", "  stone   brewing ")
    assert "Finish[" not in same, same
    print("  [D] a block names the supported answer, or the missing retrieval OK")


def test_deterministic_rejections_are_free() -> None:
    """A conceded non-answer and an answer absent from every observation both
    need no LLM. Both were being committed: `Finish['unknown']`,
    `Finish['none']`, `Finish['I do not know']` across results/logs/*_n30_*, and
    llama3.1 replied `supported: true, confidence: 1.0` to `Finish[none]` on
    question 5ac3e0f7 of the n=5 smoke run — the model's boolean is not what to
    trust for this."""
    calls = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return '{"supported": true, "confidence": 1.0, "quote": "", "answer": ""}'

    g = Layer5AnswerGate(counting)

    for action, rule in (("Finish[none]", "null_answer"),
                         ("Finish[unknown]", "null_answer"),
                         ("Finish[I do not know]", "null_answer"),
                         ("Finish[Ballast Point]", "ungrounded_answer")):
        sig = g.run(_step(action), _ctx())
        assert sig.verdict is Verdict.FAIL, (action, sig)
        assert sig.cost.tokens == 0, (action, "must not pay for an LLM call")
        assert (sig.raw or {}).get("rule") == rule, (action, sig.raw)
        assert (sig.raw or {}).get("feedback"), action
    assert not calls, "the gate called the LLM on a deterministic rejection"

    # Grounded, non-null answers still go to the LLM.
    ok = g.run(_step("Finish[Stone Brewing]"), _ctx())
    assert calls and ok.verdict is Verdict.PASS, ok

    # Yes/no answers legitimately do not appear in the retrieved text.
    assert is_grounded("yes", POOL) and is_grounded("No", POOL)
    assert is_grounded("stone   brewing!", POOL), "normalized substring match"
    assert not is_grounded("Ballast Point", POOL)
    print("  [E0] null and ungrounded answers rejected at zero token cost OK")


def test_unquoted_pass_is_downgraded() -> None:
    """`supported: true` with a quote that is not in the evidence is not a
    judgement about the evidence, so it must not clear pass_above."""
    g = Layer5AnswerGate(make_mock_gate_fn(True, 0.95, "Stone Brewing"))
    good = g.run(_step("Finish[Stone Brewing]"), _ctx())
    assert good.verdict is Verdict.PASS and (good.raw or {})["quote_grounded"]

    fake = Layer5AnswerGate(
        lambda _p: '{"supported": true, "confidence": 0.99, '
                   '"quote": "Ballast Point is headquartered in Escondido.", '
                   '"answer": "Stone Brewing"}')
    sig = fake.run(_step("Finish[Stone Brewing]"), _ctx())
    assert sig.verdict is not Verdict.PASS, sig
    assert (sig.raw or {})["quote_grounded"] is False
    print("  [E1] a PASS whose quote is not in the evidence is downgraded OK")


def test_gate_runs_even_when_cheap_layers_pass() -> None:
    """The regression this layer exists for: L2 returns a decisive PASS at 1.0 on
    a Finish step, the router stops, and the irreversible step is committed
    without any specialist ever looking at it."""
    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.PASS, 1.0, ErrorType.NONE, Cost(120, 20))),
        5: Layer5AnswerGate(make_mock_gate_fn(False, 0.95, "Stone Brewing")),
    }
    step = _step("Finish[Ballast Point]")

    without = Orchestrator(layers, AdaptiveThresholdRouter(),
                           mandatory_layers=()).verify(step, _ctx())
    assert 5 not in without.layers_run, without.layers_run
    assert without.decision is Decision.PASS, "the wrong answer sails through"

    withgate = Orchestrator(layers, AdaptiveThresholdRouter(),
                            mandatory_layers=(5,)).verify(step, _ctx())
    assert 5 in withgate.layers_run, withgate.layers_run
    assert withgate.decision is Decision.FAIL_REVISE, withgate
    assert withgate.triggering_error_type is ErrorType.ANSWER
    assert withgate.fused.source_layer_id == 5, withgate.fused
    print(f"  [E] mandatory gate reverses PASS -> {withgate.decision.value} "
          f"on a Finish the cheap layers approved OK")


def test_gate_does_not_fire_on_ordinary_steps() -> None:
    """Inapplicable means free: a Search step must not pay for the gate."""
    calls = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return '{"supported": false, "confidence": 0.9, "answer": ""}'

    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.PASS, 1.0, ErrorType.NONE, Cost(120, 20))),
        5: Layer5AnswerGate(counting),
    }
    res = Orchestrator(layers, AdaptiveThresholdRouter(),
                       mandatory_layers=(5,)).verify(_step("Search[Escondido]"), _ctx())
    assert not calls, "the gate was billed on a Search step"
    assert 5 not in res.layers_run and res.decision is Decision.PASS
    print("  [F] gate costs nothing on Search/Lookup steps OK")


def test_hard_fail_stops_escalation() -> None:
    """A settled FAIL must not be escalated past. On the first n=30 distractor
    run 80 of the 100 Layer-1 FAIL decisions asked for L3/L4 anyway, and after
    Layer 5 rejected `Finish[none]` on question 5ac3e0f7 the cascade still spent
    L3 and L4 confirming it. Neither layer can overturn a rule."""
    l3_calls, l4_calls = [], []
    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.UNSURE, 0.5, ErrorType.NONE, Cost(120, 20))),
        3: MockLayer(3, lambda s_, c_: (l3_calls.append(1) or Signal(
            Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90)))),
        4: MockLayer(4, lambda s_, c_: (l4_calls.append(1) or Signal(
            Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200)))),
        5: Layer5AnswerGate(make_mock_gate_fn(True, 1.0, "")),
    }
    orch = Orchestrator(layers, AdaptiveThresholdRouter(), mandatory_layers=(5,))
    res = orch.verify(_step("Finish[none]"), _ctx())

    assert res.decision is Decision.FAIL_REVISE, res
    assert res.fused.hard_fail and res.fused.source_layer_id == 5, res.fused
    assert not l3_calls and not l4_calls, (
        "escalated past a deterministic rejection", res.layers_run)
    assert res.total_cost.tokens == 120, ("only L2 should have been billed",
                                          res.total_cost)

    # An ordinary (non-deterministic) FAIL is still escalatable.
    soft = fuse([SignalRecord(4, Signal(Verdict.FAIL, 0.1, ErrorType.REASONING))])
    assert soft.verdict is Verdict.FAIL and not soft.hard_fail, soft
    print("  [J] a deterministic FAIL short-circuits escalation OK")


def test_feedback_prefers_the_deciding_layer() -> None:
    """The block message must come from the layer that decided, not from whoever
    ran last."""
    from src.agents.adaptive_react import AdaptiveReActAgent

    class _R:
        pass

    res = _R()
    res.fused = fuse([
        SignalRecord(3, Signal(Verdict.FAIL, 0.2, ErrorType.FACTUAL,
                               raw={"feedback": "Claim refuted by evidence."})),
        SignalRecord(5, Signal(Verdict.FAIL, 0.02, ErrorType.ANSWER,
                               raw={"deterministic": True,
                                    "feedback": "Find the fact first."})),
    ])
    res.signal_history = [
        SignalRecord(5, Signal(Verdict.FAIL, 0.02, ErrorType.ANSWER,
                               raw={"deterministic": True,
                                    "feedback": "Find the fact first."})),
        SignalRecord(3, Signal(Verdict.FAIL, 0.2, ErrorType.FACTUAL,
                               raw={"feedback": "Claim refuted by evidence."})),
    ]
    fb = AdaptiveReActAgent._feedback(res, "Finish[none]", 1)
    assert fb.startswith("Find the fact first."), fb
    assert "Do not commit an answer yet" in fb, "the directive must be appended"
    print("  [K] feedback comes from the deciding layer, plus a directive OK")


def _stop_state(conf: float, steps_remaining: int, revise_count: int,
                step_index: int = 1, position: float = 1 / 8) -> _State:
    st = _State(budget=5000, step_index=step_index, position=position,
                steps_remaining=steps_remaining, revise_count=revise_count)
    st.spent = Cost(120, 20)
    st.records.append(SignalRecord(2, Signal(
        Verdict.PASS, conf, ErrorType.REASONING, Cost(120, 20),
        raw={"raw_confidence": conf})))
    st.ran.update({2, 3, 4})
    return st


def test_repeated_revision_costs_more() -> None:
    cm = default_cost_model()
    assert cm.revision_cost(0) == 1.0
    assert cm.revision_cost(1) == 2.0
    assert cm.revision_cost(3) == 4.0

    r = AdaptiveThresholdRouter()

    # (a) revise_count alone, holding the step budget fixed.
    taus, actions = [], []
    for n in range(0, 8):
        st = _stop_state(conf=0.6, steps_remaining=7, revise_count=n)
        actions.append(r.decide(st))
        taus.append(st.traces[-1].tau_accept)
    assert taus == sorted(taus, reverse=True), taus
    assert actions[0] is Action.STOP_FAIL_REVISE, "the first block must still fire"
    assert Action.STOP_PASS in actions, (
        "after enough revisions the step must be accepted instead of re-blocked")
    flip = actions.index(Action.STOP_PASS)

    # (b) the real shape: each revision also burns a step, so both terms move.
    # This is the `Finish['unknown']` trajectory — blocked at steps 6, 8, 10, 12
    # of a max_steps=12 run — and it has to stop re-blocking well before the end.
    seq, max_steps = [], 12
    for n, step_index in enumerate([6, 8, 10, 12]):
        st = _stop_state(conf=0.6, steps_remaining=max(1, max_steps - step_index),
                         revise_count=n, step_index=step_index,
                         position=step_index / max_steps)
        seq.append(r.decide(st))
    assert seq[0] is Action.STOP_FAIL_REVISE, seq
    assert all(a is Action.STOP_PASS for a in seq[1:]), (
        "the 2nd+ block on a late step must be accepted", seq)
    print(f"  [G] tau_accept falls {taus[0]:.3f} -> {taus[-1]:.3f} with "
          f"revise_count; re-blocking stops at revision #{flip + 1} at a fixed "
          f"step budget, and at #2 on the measured late-step trajectory OK")


def test_repeat_penalty_is_configurable_and_off_by_zero() -> None:
    from src.verification.cost_model import DeclaredCostModel
    flat = DeclaredCostModel(revision_repeat_penalty=0.0)
    assert flat.revision_cost(0) == flat.revision_cost(9) == 1.0, "0 = old behaviour"
    cfg = DeclaredCostModel.from_config({"revision_repeat_penalty": 2.0})
    assert cfg.revision_cost(1) == 3.0
    assert "revision_repeat_penalty" in repr(cfg)
    print("  [H] repeat penalty is a cost-model parameter, 0.0 restores the old rule OK")


def test_fuse_prefers_the_gate_over_other_specialists() -> None:
    """On a Finish step the answer gate is the most-informed layer: it is the only
    one that compared the evidence pool against the goal."""
    recs = [
        SignalRecord(1, Signal(Verdict.PASS, 0.9, ErrorType.NONE)),
        SignalRecord(4, Signal(Verdict.PASS, 0.88, ErrorType.NONE)),
        SignalRecord(5, Signal(Verdict.FAIL, 0.05, ErrorType.ANSWER)),
    ]
    f = fuse(recs)
    assert f.source_layer_id == 5 and f.verdict is Verdict.FAIL, f
    assert f.suspected_error_type is ErrorType.ANSWER

    # A Layer 1 FAIL is still authoritative and still not outvoted.
    recs.append(SignalRecord(1, Signal(Verdict.FAIL, 0.03, ErrorType.FORMAT)))
    assert fuse(recs).source_layer_id == 1
    print("  [I] fuse ranks L5 above L3/L4 but below a deterministic L1 FAIL OK")


if __name__ == "__main__":
    test_finish_argument()
    test_applicability()
    test_parse_and_banding()
    test_feedback_is_actionable()
    test_deterministic_rejections_are_free()
    test_unquoted_pass_is_downgraded()
    test_gate_runs_even_when_cheap_layers_pass()
    test_hard_fail_stops_escalation()
    test_feedback_prefers_the_deciding_layer()
    test_gate_does_not_fire_on_ordinary_steps()
    test_repeated_revision_costs_more()
    test_repeat_penalty_is_configurable_and_off_by_zero()
    test_fuse_prefers_the_gate_over_other_specialists()
    print("\nALL CHECKS PASSED ✓")
