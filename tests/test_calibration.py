import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.calibration import (
    Calibrator, IdentityCalibrator, TemperatureScaling,
    calibrators_from_config, default_calibrators, expected_calibration_error,
    fit_temperature,
)
from src.verification.orchestrator import (
    Cost, ErrorType, FixedThresholdRouter, Layer, MockLayer, Orchestrator,
    Signal, Step, Verdict,
)


def test_identity() -> None:
    cal = IdentityCalibrator()
    assert isinstance(cal, Calibrator)
    for p in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert cal.apply(p) == p
    print("  [A] IdentityCalibrator is pass-through OK")


def test_temperature_identity_and_bounds() -> None:
    t1 = TemperatureScaling(1.0)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert abs(t1.apply(p) - p) < 1e-4, p
    try:
        TemperatureScaling(0.0)
        assert False, "expected ValueError for T=0"
    except ValueError:
        pass
    print("  [B] TemperatureScaling(T=1) ~ identity, T>0 enforced OK")


def test_temperature_direction() -> None:
    hot = TemperatureScaling(1.5)
    cold = TemperatureScaling(0.5)
    assert abs(hot.apply(0.5) - 0.5) < 1e-6
    assert abs(cold.apply(0.5) - 0.5) < 1e-6
    assert 0.5 < hot.apply(0.9) < 0.9
    assert cold.apply(0.9) > 0.9
    assert 0.1 < hot.apply(0.1) < 0.5
    print("  [C] T>1 softens, T<1 sharpens, 0.5 is fixed OK")


def test_default_map() -> None:
    cals = default_calibrators()
    assert isinstance(cals[1], IdentityCalibrator)
    assert isinstance(cals[2], IdentityCalibrator)
    assert isinstance(cals[3], TemperatureScaling) and cals[3].T == 1.0
    assert isinstance(cals[4], TemperatureScaling) and cals[4].T == 1.0
    cals = default_calibrators(temperatures={4: 1.5, 2: 1.2})
    assert cals[4].T == 1.5
    assert isinstance(cals[2], TemperatureScaling) and cals[2].T == 1.2
    assert isinstance(cals[1], IdentityCalibrator)
    cals = calibrators_from_config({"3": 1.3, "4": 1.8})
    assert cals[3].T == 1.3 and cals[4].T == 1.8
    print("  [D] default map + fitted-T override + config OK")


def _mocks(l1, l2, l3, l4) -> dict[int, Layer]:
    return {1: MockLayer(1, l1), 2: MockLayer(2, l2),
            3: MockLayer(3, l3), 4: MockLayer(4, l4)}


def test_orchestrator_default_is_identity() -> None:
    router = FixedThresholdRouter(tau_low=0.3, tau_high=0.8)
    step = Step(action="Search[X]", step_index=1)
    orch = Orchestrator(_mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20)),
        Signal(Verdict.PASS, 0.90, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.95, ErrorType.NONE, Cost(800, 200))), router)
    res = orch.verify(step)
    by_layer = {r.layer_id: r.signal for r in res.signal_history}
    assert abs(by_layer[3].confidence - 0.90) < 1e-4
    assert abs(by_layer[3].raw["raw_confidence"] - 0.90) < 1e-9
    print("  [E] orchestrator default (T=1) leaves confidence unchanged OK")


def test_orchestrator_applies_fitted_T() -> None:
    router = FixedThresholdRouter(tau_low=0.3, tau_high=0.8)
    step = Step(action="Search[X]", step_index=1)
    calibrators = default_calibrators(temperatures={4: 1.5})
    orch = Orchestrator(_mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.UNSURE, 0.5, ErrorType.REASONING, Cost(120, 20)),
        Signal(Verdict.PASS, 0.90, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.95, ErrorType.NONE, Cost(800, 200))),
        router, calibrators=calibrators)
    res = orch.verify(step)
    by_layer = {r.layer_id: r.signal for r in res.signal_history}
    assert 4 in by_layer, "expected escalation to Layer 4"
    l4 = by_layer[4]
    assert abs(l4.raw["raw_confidence"] - 0.95) < 1e-9
    assert 0.5 < l4.confidence < 0.95
    assert abs(l4.confidence - TemperatureScaling(1.5).apply(0.95)) < 1e-9
    print(f"  [F] orchestrator applies fitted T: 0.95 -> {l4.confidence:.3f} OK")


def test_fit_and_ece() -> None:
    raw = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
           0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    correct = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    T = fit_temperature(raw, correct)
    assert T > 1.0, f"over-confident layer should fit T>1, got {T}"
    cal = TemperatureScaling(T)
    cal_confs = [cal.apply(p) for p in raw]
    ece_before = expected_calibration_error(raw, correct, n_bins=5)
    ece_after = expected_calibration_error(cal_confs, correct, n_bins=5)
    assert ece_after < ece_before, (ece_before, ece_after)
    assert expected_calibration_error([0.0, 1.0], [0, 1], n_bins=10) == 0.0
    print(f"  [G] fit_temperature T={T:.2f}, ECE {ece_before:.3f} -> {ece_after:.3f} OK")


if __name__ == "__main__":
    test_identity()
    test_temperature_identity_and_bounds()
    test_temperature_direction()
    test_default_map()
    test_orchestrator_default_is_identity()
    test_orchestrator_applies_fitted_T()
    test_fit_and_ece()
    print("ALL CHECKS PASSED ✓")
