"""Resolve the router's fitted parameters from disk.

`fit_calibration.py` and `fit_efficacy.py` emit machine-readable JSON, but until
something loads it the router runs on the declared defaults — where the
calibration is the identity and the thresholds land in the same 0.75-0.90 band as
the layer confidences. This module is that loader, shared by every experiment so
a run's routing parameters are one resolved object rather than per-script glue.

Deliberately *not* auto-discovering `configs/*.json`: a run's thresholds would
then depend on whether a fit script happened to have been run in that checkout,
so the same command could produce two different routers. Paths are explicit, and
a missing path is an error, not a silent fallback to the identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.verification.calibration import (
    IdentityCalibrator, LayeredCalibrator, TemperatureScaling,
)
from src.verification.cost_model import DeclaredCostModel, default_cost_model

DEFAULT_CALIBRATION_PATH = "configs/calibration.json"
DEFAULT_COST_MODEL_PATH = "configs/cost_model.json"


@dataclass
class RoutingConfig:
    """The router's fitted parameters plus where each half came from, so a run's
    meta can record whether calibration was real or the identity."""
    calibrator: LayeredCalibrator
    cost_model: DeclaredCostModel
    calibration_path: str | None = None
    cost_model_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_identity_calibration(self) -> bool:
        """True when no layer's confidence is actually transformed — i.e. the
        run's calibrated confidences equal its raw ones."""
        for cal in self.calibrator.as_dict().values():
            if isinstance(cal, IdentityCalibrator):
                continue
            if isinstance(cal, TemperatureScaling) and abs(cal.T - 1.0) < 1e-9:
                continue
            return False
        return True

    def describe(self) -> str:
        cal_src = self.calibration_path or "declared defaults (T=1, identity)"
        cm_src = self.cost_model_path or "declared defaults"
        lines = [f"calibration: {cal_src}", f"cost model : {cm_src}"]
        if self.calibration_path and self.is_identity_calibration:
            lines.append("  NOTE: loaded temperatures are all T=1 -> calibration "
                         "is still the identity")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def load_routing_config(
    calibration_path: str | Path | None = None,
    cost_model_path: str | Path | None = None,
) -> RoutingConfig:
    """Build a `RoutingConfig` from the given paths. `None`/empty means "use the
    declared defaults" for that half; a path that does not exist raises."""
    warnings: list[str] = []

    if calibration_path:
        p = Path(calibration_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"calibration config not found: {p}. Fit one first: "
                f"python experiments/fit_calibration.py --records <dev.jsonl>")
        calibrator = LayeredCalibrator.load(p)
        cal_path: str | None = str(p)
    else:
        calibrator, cal_path = LayeredCalibrator.default(), None

    if cost_model_path:
        p = Path(cost_model_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"cost model config not found: {p}. Fit one first: "
                f"python experiments/fit_efficacy.py --records <dev.jsonl>")
        cost_model = DeclaredCostModel.load(p)
        cm_path: str | None = str(p)
        # A zero-efficacy specialist clamps to EPS, which drives tau to 0 and
        # silently switches that layer off for the whole run. Say so out loud.
        for lid in (3, 4):
            if cost_model.efficacy.get(lid, 0.0) <= 0.0:
                warnings.append(
                    f"efficacy[L{lid}] = {cost_model.efficacy.get(lid, 0.0)} "
                    f"-> tau collapses to 0, L{lid} will never be escalated to")
        if "revise_efficacy" not in _config_keys(p):
            warnings.append("revise_efficacy absent from config -> falling back "
                            "to the DECLARED 0.5 (not measured)")
    else:
        cost_model, cm_path = default_cost_model(), None

    return RoutingConfig(calibrator=calibrator, cost_model=cost_model,
                         calibration_path=cal_path, cost_model_path=cm_path,
                         warnings=warnings)


def _config_keys(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return set(obj) if isinstance(obj, dict) else set()


def add_routing_args(ap) -> None:
    """Shared argparse flags, so every experiment names these the same way."""
    ap.add_argument("--calibration", default="",
                    help=f"per-layer temperature JSON from fit_calibration.py "
                         f"(e.g. {DEFAULT_CALIBRATION_PATH}); empty = declared "
                         f"defaults, i.e. calibration is the identity")
    ap.add_argument("--cost-model", default="",
                    help=f"layer cost/efficacy JSON from fit_efficacy.py "
                         f"(e.g. {DEFAULT_COST_MODEL_PATH}); empty = declared "
                         f"defaults")


def _demo() -> None:
    cfg = load_routing_config()
    print("### defaults")
    print(cfg.describe())
    print(f"  is_identity_calibration = {cfg.is_identity_calibration}")
    print(f"  L4 calibrate(0.90) = {cfg.calibrator.calibrate(4, 0.90):.4f}")
    print(f"  cost_model = {cfg.cost_model}")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cal_p = Path(d) / "calibration.json"
        cm_p = Path(d) / "cost_model.json"
        cal_p.write_text(json.dumps({"3": 1.8, "4": 2.4}))
        cm_p.write_text(json.dumps({
            "layer_costs": {"1": 0, "2": 118, "3": 305, "4": 790},
            "efficacy": {"1": 0.21, "2": 0.11, "3": 0.42, "4": 0.0},
        }))
        cfg2 = load_routing_config(cal_p, cm_p)
        print("\n### loaded (synthetic)")
        print(cfg2.describe())
        print(f"  is_identity_calibration = {cfg2.is_identity_calibration}")
        print(f"  L4 calibrate(0.90) = {cfg2.calibrator.calibrate(4, 0.90):.4f} "
              f"(was 0.9000 raw)")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    _demo()
