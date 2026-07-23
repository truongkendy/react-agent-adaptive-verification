from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

_EPS = 1e-6


def _clamp01(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def _logit(p: float) -> float:
    p = _clamp01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Calibrator(ABC):

    @abstractmethod
    def apply(self, raw_conf: float) -> float:
        ...


class IdentityCalibrator(Calibrator):

    def apply(self, raw_conf: float) -> float:
        return raw_conf

    def __repr__(self) -> str:
        return "IdentityCalibrator()"


class TemperatureScaling(Calibrator):

    def __init__(self, T: float = 1.0):
        if not (T > 0.0):
            raise ValueError(f"Temperature T must be > 0, got {T!r}.")
        self.T = float(T)

    def apply(self, raw_conf: float) -> float:
        z = _logit(raw_conf)
        return _sigmoid(z / self.T)

    def __repr__(self) -> str:
        return f"TemperatureScaling(T={self.T:g})"


TemperatureCalibrator = TemperatureScaling


class LayeredCalibrator:

    def __init__(self, calibrators: dict[int, Calibrator]):
        self._calibrators: dict[int, Calibrator] = dict(calibrators)

    def calibrate(self, layer_id: int, raw_conf: float) -> float:
        cal = self._calibrators.get(int(layer_id))
        if cal is None:
            return raw_conf
        return cal.apply(raw_conf)

    def get(self, layer_id: int) -> Calibrator | None:
        return self._calibrators.get(int(layer_id))

    def as_dict(self) -> dict[int, Calibrator]:
        """The underlying layer_id -> Calibrator map, for `Orchestrator`'s
        `calibrators=` argument. Exposed so one fitted config can drive both
        `Orchestrator._calibrate` and the router's threshold derivation — if the
        two disagree, `fuse()`'s pass_score and the router's `p` are different
        numbers and the derivation is meaningless."""
        return dict(self._calibrators)

    @classmethod
    def default(cls, temperatures: dict[int, float] | None = None) -> "LayeredCalibrator":
        return cls(default_calibrators(temperatures=temperatures))

    @classmethod
    def from_config(cls, config: dict) -> "LayeredCalibrator":
        return cls(calibrators_from_config(config))

    @classmethod
    def load(cls, path: str | Path) -> "LayeredCalibrator":
        return cls(load_calibration_config(path))

    def __repr__(self) -> str:
        return f"LayeredCalibrator({self._calibrators!r})"


def default_calibrators(
    temperatures: dict[int, float] | None = None,
) -> dict[int, Calibrator]:
    calibrators: dict[int, Calibrator] = {
        1: IdentityCalibrator(),
        2: IdentityCalibrator(),
        3: TemperatureScaling(1.0),
        4: TemperatureScaling(1.0),
    }
    if temperatures:
        for layer_id, T in temperatures.items():
            calibrators[int(layer_id)] = TemperatureScaling(T)
    return calibrators


def calibrators_from_config(config: dict) -> dict[int, Calibrator]:
    temps = {int(k): float(v) for k, v in config.items()}
    return default_calibrators(temperatures=temps)


def load_calibration_config(path: str | Path) -> dict[int, Calibrator]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return calibrators_from_config(config)


def _nll(raw_confs: Sequence[float], correct: Sequence[int], T: float) -> float:
    cal = TemperatureScaling(T)
    total = 0.0
    for p, y in zip(raw_confs, correct):
        q = _clamp01(cal.apply(p))
        total += -(y * math.log(q) + (1 - y) * math.log(1.0 - q))
    return total / max(1, len(raw_confs))


def fit_temperature(
    raw_confs: Sequence[float],
    correct: Sequence[int],
    T_min: float = 0.05,
    T_max: float = 10.0,
) -> float:
    if len(raw_confs) != len(correct):
        raise ValueError("raw_confs and correct must have the same length.")
    if not raw_confs:
        return 1.0

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda t: _nll(raw_confs, correct, t),
            bounds=(T_min, T_max),
            method="bounded",
        )
        return float(min(T_max, max(T_min, res.x)))
    except Exception:
        pass

    def _grid(lo: float, hi: float, n: int) -> list[float]:
        log_lo, log_hi = math.log(lo), math.log(hi)
        return [math.exp(log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]

    best_T, best_nll = 1.0, float("inf")
    for T in _grid(T_min, T_max, 200):
        val = _nll(raw_confs, correct, T)
        if val < best_nll:
            best_T, best_nll = T, val

    span = (T_max - T_min) / 200.0
    lo = max(T_min, best_T - span)
    hi = min(T_max, best_T + span)
    for T in _grid(lo, hi, 100):
        val = _nll(raw_confs, correct, T)
        if val < best_nll:
            best_T, best_nll = T, val

    return float(min(T_max, max(T_min, best_T)))


def expected_calibration_error(
    confs: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> float:
    if len(confs) != len(correct):
        raise ValueError("confs and correct must have the same length.")
    n = len(confs)
    if n == 0:
        return 0.0

    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_count = [0] * n_bins

    for p, y in zip(confs, correct):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bin_conf[idx] += p
        bin_acc[idx] += y
        bin_count[idx] += 1

    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        avg_conf = bin_conf[b] / bin_count[b]
        avg_acc = bin_acc[b] / bin_count[b]
        ece += (bin_count[b] / n) * abs(avg_conf - avg_acc)
    return ece


def _demo() -> None:
    from src.verification.orchestrator import (
        Cost, ErrorType, FixedThresholdRouter, Layer, MockLayer,
        Orchestrator, Signal, Step, Verdict, _print,
    )

    def mocks(l1: Signal, l2: Signal, l3: Signal, l4: Signal) -> dict[int, Layer]:
        return {1: MockLayer(1, l1), 2: MockLayer(2, l2),
                3: MockLayer(3, l3), 4: MockLayer(4, l4)}

    step = Step(action="Search[High Plains (United States)]",
                thought="Find the elevation range of the High Plains.", step_index=1)
    router = FixedThresholdRouter(tau_low=0.3, tau_high=0.8)

    print("### (a) default calibrators — T=1 everywhere, identity in effect")
    _print("default", Orchestrator(mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20)),
        Signal(Verdict.PASS, 0.90, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.95, ErrorType.NONE, Cost(800, 200))), router).verify(step))

    print("### (b) fitted T=1.5 for Layer 4 — raw conf pulled toward 0.5")
    l4_raw = 0.95
    calibrators = default_calibrators(temperatures={4: 1.5})
    print(f"    L4 apply({l4_raw}) = {calibrators[4].apply(l4_raw):.3f}  "
          f"(was {l4_raw} raw)")
    orch_b = Orchestrator(
        mocks(
            Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
            Signal(Verdict.UNSURE, 0.5, ErrorType.REASONING, Cost(120, 20)),
            Signal(Verdict.PASS, 0.90, ErrorType.NONE, Cost(300, 90)),
            Signal(Verdict.PASS, l4_raw, ErrorType.NONE, Cost(800, 200))),
        router, calibrators=calibrators)
    res_b = orch_b.verify(step)
    _print("T=1.5@L4", res_b)
    for rec in res_b.signal_history:
        if rec.layer_id == 4:
            print(f"    L4 record: calibrated conf={rec.signal.confidence:.3f}, "
                  f"raw_confidence={rec.signal.raw['raw_confidence']:.3f}")

    print("### (c) fit_temperature + ECE before/after (synthetic over-confident layer)")
    raw_confs = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    correct   = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
                 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    T_fit = fit_temperature(raw_confs, correct)
    cal = TemperatureScaling(T_fit)
    cal_confs = [cal.apply(p) for p in raw_confs]
    ece_before = expected_calibration_error(raw_confs, correct, n_bins=5)
    ece_after = expected_calibration_error(cal_confs, correct, n_bins=5)
    print(f"    fitted T          = {T_fit:.3f}")
    print(f"    ECE before        = {ece_before:.3f}")
    print(f"    ECE after (T-fit) = {ece_after:.3f}")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    _demo()
