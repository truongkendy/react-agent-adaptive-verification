from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_EPS = 1e-6


@dataclass
class RouteContext:
    reversibility: float = 1.0
    task_stakes: float = 0.5
    position: float = 0.0
    budget_remaining: float = 1.0


class CostModel(ABC):
    @abstractmethod
    def error_cost(self, ctx: RouteContext) -> float:
        ...

    @abstractmethod
    def layer_cost(self, layer_id: int) -> float:
        ...

    @abstractmethod
    def layer_efficacy(self, layer_id: int) -> float:
        ...

    @abstractmethod
    def revision_cost(self, revise_count: int = 0) -> float:
        """Cost of sending a step back to be revised, in *agent steps*. A
        revision burns one step out of the trajectory's remaining step budget,
        so 1.0 is the natural unit.

        `revise_count` is how many revisions this episode has already spent, and
        the cost must grow with it. Measured on `adaptive_n30_ollama_steps12.json`
        and `adaptive_n30_distractor.json`, a flat cost makes repeated revision
        free at the margin, and the agent does not converge: one trajectory had
        `Finish['unknown']` blocked at steps 6, 8, 10 and 12, another alternated
        `Search[Pandikona]` / `Search[Berger Blanc Suisse]` through five
        consecutive blocks. Revisions grew to fill whatever step budget they were
        given (3->6, 2->4, 2->3 when max_steps went 8->12) and 0 of the 15
        revisions in the distractor run produced a correct answer. The nth
        revision has empirically lower value than the first, so it must not be
        priced the same."""
        ...

    @abstractmethod
    def revise_efficacy(self) -> float:
        """P(the agent actually repairs the step | we asked it to revise)."""
        ...


_DEFAULT_ERROR_WEIGHTS: dict[str, float] = {
    "base": 1.0,
    "irreversibility": 2.0,
    "stakes": 2.0,
    "earliness": 1.0,
    "scarcity": 1.0,
}
_DEFAULT_LAYER_COSTS: dict[int, float] = {
    1: 0.0, 2: 120.0, 3: 300.0, 4: 800.0, 5: 400.0,
}
_DEFAULT_EFFICACY: dict[int, float] = {
    1: 0.30, 2: 0.30, 3: 0.55, 4: 0.60, 5: 0.70,
}
# One wasted agent step. DECLARED, not measured — see fit_efficacy.py.
_DEFAULT_REVISION_COST: float = 1.0
_DEFAULT_REVISE_EFFICACY: float = 0.50
# Multiplier on the revision cost per revision already spent this episode:
# cost = base * (1 + penalty * revise_count). At the default 1.0 the second
# revision costs 2 steps, the third 3, so `tau_accept` collapses after two
# blocks and the step is accepted instead of re-blocked forever. See
# `CostModel.revision_cost` for the measurements this is priced against.
_DEFAULT_REVISION_REPEAT_PENALTY: float = 1.0


class DeclaredCostModel(CostModel):

    def __init__(
        self,
        error_weights: dict[str, float] | None = None,
        layer_costs: dict[int, float] | None = None,
        efficacy: dict[int, float] | None = None,
        revision_cost: float | None = None,
        revise_efficacy: float | None = None,
        revision_repeat_penalty: float | None = None,
    ):
        self.error_weights: dict[str, float] = dict(_DEFAULT_ERROR_WEIGHTS)
        if error_weights:
            self.error_weights.update({k: float(v) for k, v in error_weights.items()})

        src_costs = layer_costs if layer_costs is not None else _DEFAULT_LAYER_COSTS
        self.layer_costs: dict[int, float] = {int(k): float(v) for k, v in src_costs.items()}

        src_eff = efficacy if efficacy is not None else _DEFAULT_EFFICACY
        self.efficacy: dict[int, float] = {int(k): float(v) for k, v in src_eff.items()}

        self._revision_cost = float(
            _DEFAULT_REVISION_COST if revision_cost is None else revision_cost)
        self._revise_efficacy = float(
            _DEFAULT_REVISE_EFFICACY if revise_efficacy is None else revise_efficacy)
        self._revision_repeat_penalty = max(0.0, float(
            _DEFAULT_REVISION_REPEAT_PENALTY if revision_repeat_penalty is None
            else revision_repeat_penalty))

    def error_cost(self, ctx: RouteContext) -> float:
        w = self.error_weights
        return (
            w["base"]
            + w["irreversibility"] * (1.0 - ctx.reversibility)
            + w["stakes"] * ctx.task_stakes
            + w["earliness"] * (1.0 - ctx.position)
            + w["scarcity"] * (1.0 - ctx.budget_remaining)
        )

    def layer_cost(self, layer_id: int) -> float:
        return self.layer_costs.get(int(layer_id), 0.0)

    def layer_efficacy(self, layer_id: int) -> float:
        return max(_EPS, self.efficacy.get(int(layer_id), _EPS))

    def revision_cost(self, revise_count: int = 0) -> float:
        n = max(0, int(revise_count))
        return self._revision_cost * (1.0 + self._revision_repeat_penalty * n)

    def revise_efficacy(self) -> float:
        return max(_EPS, self._revise_efficacy)

    @classmethod
    def from_config(cls, config: dict) -> "DeclaredCostModel":
        return cls(
            error_weights=config.get("error_weights"),
            layer_costs=config.get("layer_costs"),
            efficacy=config.get("efficacy"),
            revision_cost=config.get("revision_cost"),
            revise_efficacy=config.get("revise_efficacy"),
            revision_repeat_penalty=config.get("revision_repeat_penalty"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeclaredCostModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_config(json.load(f))

    def __repr__(self) -> str:
        return (f"DeclaredCostModel(error_weights={self.error_weights}, "
                f"layer_costs={self.layer_costs}, efficacy={self.efficacy}, "
                f"revision_cost={self._revision_cost}, "
                f"revise_efficacy={self._revise_efficacy}, "
                f"revision_repeat_penalty={self._revision_repeat_penalty})")


def default_cost_model() -> DeclaredCostModel:
    return DeclaredCostModel()
