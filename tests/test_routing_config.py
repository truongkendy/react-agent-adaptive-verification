import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.calibration import TemperatureScaling
from src.verification.cost_model import RouteContext
from src.verification.orchestrator import (
    AdaptiveThresholdRouter, Cost, ErrorType, Layer, MockLayer, Orchestrator,
    Signal, SignalRecord, Step, Verdict, _State,
)
from src.verification.routing_config import (
    add_routing_args, load_routing_config,
)

FITTED_CAL = {"3": 1.8, "4": 2.4}
FITTED_CM = {
    "error_weights": {"base": 1.0, "irreversibility": 2.0, "stakes": 2.0,
                      "earliness": 1.0, "scarcity": 1.0},
    "layer_costs": {"1": 0.0, "2": 118.0, "3": 305.0, "4": 790.0},
    "efficacy": {"1": 0.21, "2": 0.11, "3": 0.42, "4": 0.37},
}


def _write(d: Path, cal=FITTED_CAL, cm=FITTED_CM) -> tuple[Path, Path]:
    cal_p, cm_p = d / "calibration.json", d / "cost_model.json"
    cal_p.write_text(json.dumps(cal))
    cm_p.write_text(json.dumps(cm))
    return cal_p, cm_p


def _mocks(l1, l2, l3, l4) -> dict[int, Layer]:
    return {1: MockLayer(1, l1), 2: MockLayer(2, l2),
            3: MockLayer(3, l3), 4: MockLayer(4, l4)}


def test_defaults_are_declared_and_identity() -> None:
    cfg = load_routing_config()
    assert cfg.calibration_path is None and cfg.cost_model_path is None
    assert cfg.is_identity_calibration, "declared defaults must be a no-op"
    for p in (0.03, 0.5, 0.9, 1.0):
        assert abs(cfg.calibrator.calibrate(4, p) - p) < 1e-4, p
    assert cfg.cost_model.layer_cost(4) == 800.0
    print("  [A] no paths -> declared cost model + identity calibration OK")


def test_loaded_calibration_is_not_identity() -> None:
    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d))
        cfg = load_routing_config(cal_p, cm_p)
        assert not cfg.is_identity_calibration
        assert cfg.calibration_path == str(cal_p)
        assert cfg.cost_model_path == str(cm_p)
        # T>1 pulls an over-confident layer toward 0.5 — the whole point.
        assert abs(cfg.calibrator.calibrate(4, 0.90)
                   - TemperatureScaling(2.4).apply(0.90)) < 1e-12
        assert 0.5 < cfg.calibrator.calibrate(4, 0.90) < 0.90
        assert cfg.cost_model.layer_cost(4) == 790.0
        assert abs(cfg.cost_model.layer_efficacy(3) - 0.42) < 1e-12
    print("  [B] loaded config: L4 0.900 -> "
          f"{TemperatureScaling(2.4).apply(0.90):.3f}, costs/efficacy applied OK")


def test_missing_path_raises_not_silently_identity() -> None:
    for kwargs in ({"calibration_path": "configs/does_not_exist.json"},
                   {"cost_model_path": "configs/does_not_exist.json"}):
        try:
            load_routing_config(**kwargs)
            assert False, f"expected FileNotFoundError for {kwargs}"
        except FileNotFoundError:
            pass
    print("  [C] a missing config path raises instead of falling back OK")


def test_zero_efficacy_is_warned() -> None:
    cm = dict(FITTED_CM, efficacy={"1": 0.2, "2": 0.1, "3": 0.0, "4": 0.37})
    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d), cm=cm)
        cfg = load_routing_config(cal_p, cm_p)
        assert any("efficacy[L3]" in w for w in cfg.warnings), cfg.warnings
        assert any("revise_efficacy" in w for w in cfg.warnings), cfg.warnings
        assert "WARNING" in cfg.describe()
    print("  [D] zero efficacy + missing revise_efficacy surface as warnings OK")


def test_router_derives_thresholds_from_loaded_config() -> None:
    """The loaded numbers must reach the derivation, not just be stored."""
    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d))
        cfg = load_routing_config(cal_p, cm_p)
        r = AdaptiveThresholdRouter(cost_model=cfg.cost_model,
                                    calibrator=cfg.calibrator)

        st = _State(budget=5000, step_index=1, position=1 / 8)
        st.spent = Cost(118, 20)
        st.records.append(SignalRecord(4, Signal(
            Verdict.PASS, 0.90, ErrorType.NONE, Cost(790, 200),
            raw={"raw_confidence": 0.90})))
        st.ran.add(4)
        r.decide(st)
        tr = st.traces[-1]

        assert abs(tr.raw_confidence - 0.90) < 1e-12
        assert abs(tr.calibrated_confidence
                   - TemperatureScaling(2.4).apply(0.90)) < 1e-12, tr
        assert tr.calibrated_confidence < tr.raw_confidence

        cm = cfg.cost_model
        ctx = RouteContext(reversibility=1.0, task_stakes=0.5, position=1 / 8,
                           budget_remaining=st.budget_remaining_frac)
        c_next = cm.layer_cost(3) / max(st.remaining_budget, 1e-6)
        expected = min(1.0, max(0.0, 1.0 - c_next / max(
            cm.error_cost(ctx) * cm.layer_efficacy(3), 1e-6)))
        assert abs(tr.tau - expected) < 1e-9, (tr.tau, expected)
    print(f"  [E] router tau derived from loaded cost model = {tr.tau:.4f}, "
          f"conf {tr.raw_confidence:.2f} -> {tr.calibrated_confidence:.3f} OK")


def test_escalation_cost_follows_cost_model() -> None:
    """Affordability and the derivation must read the same per-layer costs."""
    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d))
        cfg = load_routing_config(cal_p, cm_p)
        r = AdaptiveThresholdRouter(cost_model=cfg.cost_model)
        assert r.escalation_cost == {3: 305, 4: 790}, r.escalation_cost
        # An explicit override still wins, for sweeps.
        r2 = AdaptiveThresholdRouter(cost_model=cfg.cost_model,
                                     escalation_cost={3: 1, 4: 2})
        assert r2.escalation_cost == {3: 1, 4: 2}
    print(f"  [F] escalation_cost derived from cost model = "
          f"{r.escalation_cost} OK")


def test_orchestrator_and_router_share_one_calibrator() -> None:
    """If the two calibrate differently, fuse()'s pass_score and the router's p
    are different numbers and every derived threshold is meaningless."""
    from src.agents.adaptive_react import build_default_orchestrator

    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d))
        cfg = load_routing_config(cal_p, cm_p)
        router = AdaptiveThresholdRouter(cost_model=cfg.cost_model,
                                         calibrator=cfg.calibrator)
        # Low L1/L2 confidence so the chain actually escalates and L3/L4 — the
        # two layers the fitted temperatures apply to — end up in the history.
        orch = Orchestrator(_mocks(
            Signal(Verdict.UNSURE, 0.50, ErrorType.NONE, Cost(0, 1)),
            Signal(Verdict.UNSURE, 0.50, ErrorType.FACTUAL, Cost(118, 20)),
            Signal(Verdict.PASS, 0.85, ErrorType.NONE, Cost(305, 90)),
            Signal(Verdict.PASS, 0.90, ErrorType.NONE, Cost(790, 200))),
            router, calibrators=cfg.calibrator.as_dict())
        res = orch.verify(Step(action="Search[X]", step_index=1))

        for rec in res.signal_history:
            raw = rec.signal.raw["raw_confidence"]
            assert abs(rec.signal.confidence
                       - cfg.calibrator.calibrate(rec.layer_id, raw)) < 1e-12, rec
        by_layer = {r.layer_id: r.signal for r in res.signal_history}
        assert 3 in by_layer, f"expected escalation to L3, got {res.layers_run}"
        for lid in (3, 4):
            if lid in by_layer:
                s = by_layer[lid]
                assert s.confidence < s.raw["raw_confidence"], (lid, s)
        # ...and the builder wires the same object into both halves.
        built = build_default_orchestrator(
            llm=None, on_verify_call=lambda: None, routing=cfg,
            use_layer3=False, use_layer4=True)
        assert built.calibrators == built.router.calibrator.as_dict()
        assert built.router.cost_model is cfg.cost_model
    print(f"  [G] orchestrator + router share one calibrator "
          f"(layers_run={res.layers_run}) OK")


def test_agent_forwards_routing_and_knobs() -> None:
    """The CLI's flags have to survive the whole path: routing_config -> agent ->
    orchestrator -> router. Construction touches neither llm nor env."""
    from src.agents.adaptive_react import AdaptiveReActAgent

    with tempfile.TemporaryDirectory() as d:
        cal_p, cm_p = _write(Path(d))
        cfg = load_routing_config(cal_p, cm_p)
        agent = AdaptiveReActAgent(llm=None, env=None, routing=cfg,
                                   max_steps=6, budget=1234, revise_bias=2.5,
                                   use_layer3=False, use_layer4=True)
        orch = agent.orchestrator
        assert orch.budget == 1234 and orch.max_steps == 6
        assert orch.router.revise_bias == 2.5
        assert orch.router.cost_model is cfg.cost_model
        assert orch.router.calibrator is cfg.calibrator
        assert orch.calibrators == cfg.calibrator.as_dict()
        # --no-layer3 must leave L3 unregistered so escalation redirects to L4.
        assert set(orch.layers) == {1, 2, 4, 5}, sorted(orch.layers)
        # L5 is a mandatory gate, not an escalation target.
        assert orch.mandatory_layers == (5,), orch.mandatory_layers

        off = AdaptiveReActAgent(llm=None, env=None, routing=cfg, max_steps=6,
                                 use_layer3=False, use_layer5=False)
        assert set(off.orchestrator.layers) == {1, 2, 4}, sorted(off.orchestrator.layers)
        assert off.orchestrator.mandatory_layers == ()
    print("  [I] agent forwards routing + budget/revise_bias to the router OK")


def test_argparse_flags_present() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    add_routing_args(ap)
    args = ap.parse_args([])
    assert args.calibration == "" and args.cost_model == ""
    args = ap.parse_args(["--calibration", "a.json", "--cost-model", "b.json"])
    assert args.calibration == "a.json" and args.cost_model == "b.json"
    print("  [H] --calibration / --cost-model flags parse, default to '' OK")


if __name__ == "__main__":
    test_defaults_are_declared_and_identity()
    test_loaded_calibration_is_not_identity()
    test_missing_path_raises_not_silently_identity()
    test_zero_efficacy_is_warned()
    test_router_derives_thresholds_from_loaded_config()
    test_escalation_cost_follows_cost_model()
    test_orchestrator_and_router_share_one_calibrator()
    test_agent_forwards_routing_and_knobs()
    test_argparse_flags_present()
    print("ALL CHECKS PASSED ✓")
