import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.calibration import LayeredCalibrator, default_calibrators
from src.verification.cost_model import (
    DeclaredCostModel, RouteContext, default_cost_model,
)
from src.verification.orchestrator import (
    Action, AdaptiveThresholdRouter, Context, Cost, Decision, ErrorType, Layer,
    MockLayer, Orchestrator, Signal, Step, Verdict, _State, SignalRecord,
)


def _mocks(l1, l2, l3, l4) -> dict[int, Layer]:
    return {1: MockLayer(1, l1), 2: MockLayer(2, l2),
            3: MockLayer(3, l3), 4: MockLayer(4, l4)}


def test_no_stored_threshold() -> None:
    r = AdaptiveThresholdRouter()
    forbidden = {"tau", "tau_low", "tau_high", "threshold"}
    assert not (forbidden & set(vars(r))), vars(r)
    print("  [A] router stores no tau/threshold constant OK")


def test_cost_model_is_arithmetic() -> None:
    cm = default_cost_model()
    low = cm.error_cost(RouteContext(reversibility=1.0, task_stakes=0.0, position=1.0))
    high = cm.error_cost(RouteContext(reversibility=0.0, task_stakes=1.0, position=0.0))
    assert high > low
    assert cm.layer_cost(4) > cm.layer_cost(3) > cm.layer_cost(2) > cm.layer_cost(1)
    assert 0.0 < cm.layer_efficacy(4) <= 1.0
    print(f"  [B] cost model arithmetic: error_cost {low:.2f} -> {high:.2f} OK")


def test_tau_recomputed_from_derivation() -> None:
    cm = default_cost_model()
    cal = LayeredCalibrator.default()
    r = AdaptiveThresholdRouter(cost_model=cm, calibrator=cal)

    state = _State(budget=5000, step_index=1, position=1 / 8)
    state.spent = Cost(120, 20)
    sig = Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20),
                 raw={"raw_confidence": 0.5})
    state.records.append(SignalRecord(2, sig))
    state.ran.add(2)

    action = r.decide(state)
    tr = state.traces[-1]

    ctx = RouteContext(reversibility=1.0, task_stakes=0.5, position=1 / 8,
                       budget_remaining=state.budget_remaining_frac)
    target = 3
    c_next = cm.layer_cost(target) / max(state.remaining_budget, 1e-6)
    tau_expected = 1.0 - c_next / max(cm.error_cost(ctx) * cm.layer_efficacy(target), 1e-6)
    tau_expected = min(1.0, max(0.0, tau_expected))
    assert abs(tr.tau - tau_expected) < 1e-9, (tr.tau, tau_expected)
    assert tr.target_layer == 3 and tr.layer_id == 2
    print(f"  [C] tau derived per call = {tr.tau:.4f} (target L3) OK")


def test_type_chooses_target_not_whether() -> None:
    cm = default_cost_model()
    r = AdaptiveThresholdRouter(cost_model=cm)

    def state_with(err_type):
        st = _State(budget=5000, step_index=1, position=1 / 8)
        st.spent = Cost(120, 20)
        st.records.append(SignalRecord(2, Signal(
            Verdict.UNSURE, 0.5, err_type, Cost(120, 20),
            raw={"raw_confidence": 0.5})))
        st.ran.add(2)
        return st

    st_fac = state_with(ErrorType.FACTUAL)
    st_rea = state_with(ErrorType.REASONING)
    r.decide(st_fac)
    r.decide(st_rea)
    assert st_fac.traces[-1].target_layer == 3
    assert st_rea.traces[-1].target_layer == 4
    print("  [D] suspected type selects target (L3 vs L4) only OK")


def test_budget_pressure_suppresses_escalation() -> None:
    cm = default_cost_model()
    r = AdaptiveThresholdRouter(cost_model=cm)
    st = _State(budget=5000, step_index=1, position=1 / 8)
    st.spent = Cost(4990, 20)
    st.records.append(SignalRecord(2, Signal(
        Verdict.UNSURE, 0.4, ErrorType.FACTUAL, Cost(120, 20),
        raw={"raw_confidence": 0.4})))
    st.ran.add(2)
    r.decide(st)
    tr = st.traces[-1]
    assert tr.tau == 0.0, tr.tau
    assert tr.action in ("stop_pass", "stop_fail_revise")
    print(f"  [E] budget pressure drives tau->0, no escalation (action={tr.action}) OK")


def _stop_state(conf: float, steps_remaining: int, err=ErrorType.REASONING,
                verdict=Verdict.PASS) -> _State:
    """A state parked on the stop branch: both escalation targets already ran,
    so the accept/revise rule is what decides.

    The signal must be *decisive* (not UNSURE) for the threshold to be consulted
    at all — an all-abstaining state short-circuits to STOP_PASS, which is what
    `test_abstention_never_revises` covers."""
    st = _State(budget=5000, step_index=1, position=1 / 8,
                steps_remaining=steps_remaining)
    st.spent = Cost(120, 20)
    st.records.append(SignalRecord(2, Signal(
        verdict, conf, err, Cost(120, 20), raw={"raw_confidence": conf})))
    st.ran.update({2, 3, 4})
    return st


def test_accept_threshold_is_derived_not_half() -> None:
    cm = default_cost_model()
    r = AdaptiveThresholdRouter(cost_model=cm)
    st = _stop_state(conf=0.6, steps_remaining=7)
    action = r.decide(st)
    tr = st.traces[-1]

    ctx = RouteContext(reversibility=1.0, task_stakes=0.5, position=1 / 8,
                       budget_remaining=st.budget_remaining_frac)
    c_rev = cm.revision_cost() / 7.0
    expected = 1.0 - c_rev / max(cm.error_cost(ctx) * cm.revise_efficacy(), 1e-6)
    expected = min(1.0, max(0.0, expected))
    assert abs(tr.tau_accept - expected) < 1e-9, (tr.tau_accept, expected)
    assert tr.target_layer is None, "should be on the stop branch"
    # conf=0.6 clears the old hardcoded 0.5 but not the derived threshold, so
    # the two rules disagree — this is the regression guard for that constant.
    assert action == Action.STOP_FAIL_REVISE, action
    print(f"  [G] accept threshold derived = {tr.tau_accept:.4f} "
          f"(conf 0.60 -> {action.value}, old rule said pass) OK")


def test_step_scarcity_suppresses_revision() -> None:
    r = AdaptiveThresholdRouter()
    early = _stop_state(conf=0.6, steps_remaining=7)
    late = _stop_state(conf=0.6, steps_remaining=1)
    a_early, a_late = r.decide(early), r.decide(late)
    t_early, t_late = early.traces[-1].tau_accept, late.traces[-1].tau_accept

    assert t_early > t_late, (t_early, t_late)
    assert a_early == Action.STOP_FAIL_REVISE
    assert a_late == Action.STOP_PASS, "no steps left to recover in — accept"
    print(f"  [H] step scarcity lowers tau_accept {t_early:.3f} -> {t_late:.3f}, "
          f"revision suppressed at the end OK")


def test_revise_bias_sweeps_monotonically() -> None:
    taus = []
    for bias in (0.25, 1.0, 4.0, 16.0):
        r = AdaptiveThresholdRouter(revise_bias=bias)
        st = _stop_state(conf=0.6, steps_remaining=7)
        r.decide(st)
        taus.append(st.traces[-1].tau_accept)
    assert taus == sorted(taus, reverse=True), taus
    assert taus[0] > 0.6 > taus[-1], taus
    print(f"  [I] revise_bias sweeps tau_accept {taus[0]:.3f} -> {taus[-1]:.3f} "
          f"(brackets conf=0.6) OK")


def test_abstention_never_revises() -> None:
    """Every layer UNSURE means we obtained no evidence. Revising then costs an
    agent step and cannot improve anything — this is what produced 21 revisions
    and 4 step-exhausted trajectories in the n=15 Ollama run, because 0.5 is the
    layers' 'I could not tell' value and tau_accept sits near 0.89."""
    from src.verification.orchestrator import FixedThresholdRouter, fuse

    r = AdaptiveThresholdRouter()
    st = _stop_state(conf=0.5, steps_remaining=7, verdict=Verdict.UNSURE)
    assert not fuse(st.records).decisive
    assert r.decide(st) == Action.STOP_PASS, "abstention must not revise"
    assert st.traces[-1].tau_accept > 0.5, "threshold would have said revise"

    # The fixed baseline router shares the rule via BaseRouter.stop_action.
    st2 = _stop_state(conf=0.4, steps_remaining=7, verdict=Verdict.UNSURE)
    assert FixedThresholdRouter().stop_action(st2) == Action.STOP_PASS

    # A decisive low score still revises — abstention is not a blanket amnesty.
    st3 = _stop_state(conf=0.4, steps_remaining=7, verdict=Verdict.FAIL)
    assert r.decide(st3) == Action.STOP_FAIL_REVISE
    print("  [K] all-UNSURE -> STOP_PASS; decisive FAIL still revises OK")


def test_narrow_pass_does_not_certify_the_step() -> None:
    """Layer 1's pass_confidence is the constant 0.90, meaning "no rule
    violation" — not P(step is good). At step 1 the derived accept threshold is
    0.9006, so treating it as a probability revised clean steps by a margin of
    0.0006, decided by float noise. A narrow PASS must abstain instead."""
    from src.verification.orchestrator import fuse
    from src.verification.layer1_rule import Layer1Adapter

    l1 = Layer1Adapter(pass_confidence=0.9)
    orch = Orchestrator({1: l1}, AdaptiveThresholdRouter())
    step = Step(action="Search[Sergei Aleksandrovich Tokarev]",
                thought="I need to find which university he taught at.",
                step_index=1)
    res = orch.verify(step)

    assert res.layers_run == [1]
    assert not res.fused.decisive, "an L1-only PASS must not be decisive"
    assert res.decision is Decision.PASS, res.decision
    # The threshold really was above 0.9 — this is the margin that used to decide.
    tr = res.decision_traces[-1]
    assert tr.tau_accept > 0.9 > tr.calibrated_confidence - 1e-9, tr
    assert abs(tr.tau_accept - 0.9) < 0.01, f"expected a knife edge, got {tr}"

    # An L1 FAIL is still decisive and still revises.
    res2 = orch.verify(Step(action="Browse[http://x.com]", step_index=1))
    assert res2.fused.decisive and res2.decision is Decision.FAIL_REVISE
    print(f"  [O] L1-only PASS abstains (tau_accept={tr.tau_accept:.4f} vs "
          f"conf=0.9000); L1 FAIL still revises OK")


def test_deterministic_fail_is_not_outvoted() -> None:
    """Layer 1 checks grammar/policy/duplicates — facts, not opinions. A high
    Layer 2 score used to override it because fuse() picked the record with the
    *highest* confidence, i.e. the most optimistic layer. In the n=15 run the same
    duplicate action was blocked at one step and let through at another purely
    because L2 happened to score 1.0 there."""
    from src.verification.orchestrator import fuse

    st = _State(budget=5000, step_index=2, position=2 / 8, steps_remaining=6)
    st.records.append(SignalRecord(1, Signal(
        Verdict.FAIL, 0.03, ErrorType.FORMAT, Cost(0, 1),
        raw={"raw_confidence": 0.03, "error": "duplicates an action already run"})))
    st.records.append(SignalRecord(2, Signal(
        Verdict.PASS, 1.0, ErrorType.NONE, Cost(120, 20),
        raw={"raw_confidence": 1.0})))
    st.ran.update({1, 2})

    fused = fuse(st.records)
    assert fused.verdict is Verdict.FAIL, fused
    assert fused.source_layer_id == 1, fused
    assert fused.pass_score == 0.03, fused
    assert AdaptiveThresholdRouter().stop_action(st) == Action.STOP_FAIL_REVISE
    print("  [L] Layer 1 FAIL survives a Layer 2 PASS at confidence 1.0 OK")


def test_fuse_prefers_informed_layer_not_optimistic_one() -> None:
    from src.verification.orchestrator import fuse

    recs = [
        SignalRecord(2, Signal(Verdict.PASS, 0.95, ErrorType.NONE, Cost(120, 20))),
        SignalRecord(4, Signal(Verdict.FAIL, 0.20, ErrorType.REASONING, Cost(800, 200))),
    ]
    fused = fuse(recs)
    assert fused.source_layer_id == 4 and fused.verdict is Verdict.FAIL, fused

    # An UNSURE specialist must not displace a decisive cheaper layer.
    recs2 = [
        SignalRecord(2, Signal(Verdict.FAIL, 0.20, ErrorType.NONE, Cost(120, 20))),
        SignalRecord(4, Signal(Verdict.UNSURE, 0.50, ErrorType.NONE, Cost(800, 200))),
    ]
    fused2 = fuse(recs2)
    assert fused2.source_layer_id == 2 and fused2.pass_score == 0.20, fused2
    assert fused2.decisive
    print("  [M] fuse() ranks by layer id among decisive signals OK")


def test_fixed_router_keeps_its_baseline_constant() -> None:
    from src.verification.orchestrator import FixedThresholdRouter
    fixed = FixedThresholdRouter()
    for conf, expected in ((0.6, Action.STOP_PASS), (0.4, Action.STOP_FAIL_REVISE)):
        st = _stop_state(conf=conf, steps_remaining=7)
        assert fixed.stop_action(st) == expected, (conf, expected)
    print("  [J] FixedThresholdRouter baseline still splits at 0.5 OK")


def test_inapplicable_layers_are_skipped() -> None:
    """Verification is pre-hoc, so on step 1 there is no previous observation and
    no evidence pool. Layers that judge a Thought *against* observed text must be
    skipped there instead of asked and then believed."""
    from src.verification.layer4_llm_judge import Layer4LLMJudge, make_mock_judge_fn
    from src.verification.layer3_retrieval import Layer3RetrievalVerifier

    l4 = Layer4LLMJudge(make_mock_judge_fn(False, 0.9, "unsupported"))
    assert not l4.applicable(Step(action="Search[X]", thought="t", step_index=1),
                             Context())
    assert l4.applicable(Step(action="Search[X]", thought="t", step_index=2,
                              prev_observation="Obama was born in 1961."),
                         Context())

    l3 = Layer3RetrievalVerifier(lambda *a, **k: (0.0, 0.0, 1.0))
    assert not l3.applicable(Step(action="Search[X]", thought="t", step_index=1),
                             Context())
    assert l3.applicable(Step(action="Search[X]", thought="t", step_index=1),
                         Context(scratch={"observations": ["Some evidence."]}))

    # End to end: with only L1 applicable, the step must pass, not be revised for
    # lack of a verifier — and no L3/L4 tokens may be spent.
    orch = Orchestrator({1: MockLayer(1, Signal(Verdict.PASS, 0.9, ErrorType.NONE,
                                                Cost(0, 1))),
                         4: l4},
                        AdaptiveThresholdRouter())
    res = orch.verify(Step(action="Search[X]", thought="t", step_index=1))
    assert res.layers_run == [1], res.layers_run
    assert res.decision.value == "pass", res.decision
    assert res.total_cost.tokens == 0, res.total_cost
    print("  [N] L3/L4 skipped when there is no observation to check against OK")


def test_end_to_end_and_trace_logged() -> None:
    router = AdaptiveThresholdRouter(cost_model=default_cost_model(),
                                     calibrator=LayeredCalibrator.default())
    step = Step(action="Search[X]", step_index=1, task_stakes=0.9, reversibility=0.2)
    orch = Orchestrator(_mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200))), router)
    res = orch.verify(step)
    assert res.decision_traces, "expected at least one DecisionTrace"
    tr = res.decision_traces[0]
    for fld in ("tau", "calibrated_confidence", "error_cost", "budget_remaining",
                "action", "layer_id"):
        assert hasattr(tr, fld), fld
    print(f"  [F] end-to-end logs {len(res.decision_traces)} trace(s), "
          f"layers_run={res.layers_run} OK")


if __name__ == "__main__":
    test_no_stored_threshold()
    test_cost_model_is_arithmetic()
    test_tau_recomputed_from_derivation()
    test_type_chooses_target_not_whether()
    test_budget_pressure_suppresses_escalation()
    test_accept_threshold_is_derived_not_half()
    test_step_scarcity_suppresses_revision()
    test_revise_bias_sweeps_monotonically()
    test_abstention_never_revises()
    test_narrow_pass_does_not_certify_the_step()
    test_deterministic_fail_is_not_outvoted()
    test_fuse_prefers_informed_layer_not_optimistic_one()
    test_fixed_router_keeps_its_baseline_constant()
    test_inapplicable_layers_are_skipped()
    test_end_to_end_and_trace_logged()
    print("ALL CHECKS PASSED ✓")
