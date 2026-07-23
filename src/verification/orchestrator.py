from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

from src.verification.calibration import (
    Calibrator, LayeredCalibrator, default_calibrators,
)
from src.verification.cost_model import CostModel, RouteContext, default_cost_model

EPS = 1e-6


class Verdict(str, Enum):
    PASS   = "pass"
    FAIL   = "fail"
    UNSURE = "unsure"


class ErrorType(str, Enum):
    NONE      = "none"
    FORMAT    = "format"
    FACTUAL   = "factual"
    REASONING = "reasoning"
    # The accumulated evidence does not support the answer being committed. Its
    # own type because it is not a property of the Thought (REASONING) or of a
    # claim inside it (FACTUAL) — it is a mismatch between the evidence pool and
    # the goal, which no pre-hoc layer looks at. See layer5_answer.py.
    ANSWER    = "answer"


@dataclass
class Cost:
    tokens: int = 0
    latency_ms: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.tokens + other.tokens,
                    self.latency_ms + other.latency_ms)


@dataclass
class Signal:
    verdict: Verdict
    confidence: float
    suspected_error_type: ErrorType = ErrorType.NONE
    cost: Cost = field(default_factory=Cost)
    raw: dict | None = None


@dataclass
class Step:
    action: str
    thought: str = ""
    goal: str = ""
    prev_observation: str = ""
    step_index: int = 0
    reversibility: float = 1.0
    task_stakes: float = 0.5


@dataclass
class Context:
    revise_count: int = 0
    scratch: dict = field(default_factory=dict)


class Layer(ABC):

    def __init__(self, layer_id: int):
        self.layer_id = layer_id

    @abstractmethod
    def run(self, step: Step, context: Context) -> Signal:
        ...

    def applicable(self, step: Step, context: Context) -> bool:
        """Whether this layer has the inputs it needs for *this* step.

        Verification is pre-hoc — a step is judged before its own observation
        exists — so on step 1 there is no previous observation and no evidence
        pool. Layers that check a Thought *against* observed text cannot do their
        job there, and running them anyway produced confident-sounding nonsense
        ("Reason only from the actual Observation; do not fabricate facts." on the
        very first step). An inapplicable layer is skipped rather than run and
        discounted, so it costs nothing and cannot be escalated to."""
        return True

    def commit(self, step: Step) -> None:
        """Called after the step actually ran in the environment. Layers that
        accumulate per-episode state must record it here, not in `run()`: a step
        that `run()` sees may still be revised away and never executed."""
        return None

    def reset(self) -> None:
        return None


Behavior = Signal | Callable[[Step, Context], Signal]


class MockLayer(Layer):

    def __init__(self, layer_id: int, behavior: Behavior):
        super().__init__(layer_id)
        self._behavior = behavior

    def run(self, step: Step, context: Context) -> Signal:
        if callable(self._behavior):
            return self._behavior(step, context)
        return self._behavior


LAYER_DOMAIN: dict[int, ErrorType] = {
    1: ErrorType.FORMAT,
    2: ErrorType.NONE,
    3: ErrorType.FACTUAL,
    4: ErrorType.REASONING,
    5: ErrorType.ANSWER,
}


@dataclass
class SignalRecord:
    layer_id: int
    signal: Signal


@dataclass
class FusedView:
    verdict: Verdict
    confidence: float
    suspected_error_type: ErrorType
    pass_score: float
    source_layer_id: int
    # False when *every* layer abstained (all UNSURE). Verification then produced
    # no evidence about this step, which is not the same as evidence that it is
    # bad — see `BaseRouter.stop_action`.
    decisive: bool = True
    # The chosen signal is a FAIL that is a *fact*, not a judgement: a Layer 1
    # rule violation, or a Layer 5 deterministic rejection (a conceded
    # non-answer, an answer absent from every observation). No further evidence
    # can change it, so escalating past it buys nothing and costs a layer. On the
    # first n=30 distractor run 80 of the 100 Layer-1 FAIL decisions escalated
    # anyway.
    hard_fail: bool = False


# Layers whose FAIL is a fact rather than an opinion. Layer 1 checks action
# grammar, tool names, the no-URL policy, the Lookup-after-Search precondition
# and duplicates: none of that is probabilistic, so it must not be outvoted by an
# LLM layer that happened to score the step highly.
DETERMINISTIC_LAYERS: frozenset[int] = frozenset({1})

# Layers whose PASS certifies only their own narrow domain, not the step overall.
# Layer 1's `pass_confidence` (0.90) is a constant meaning "the rules found no
# violation" — it is not P(step is good), and it says nothing about whether the
# reasoning is sound or the facts hold. Comparing it to a derived accept
# threshold is a category error, and a costly one: at step 1 the threshold works
# out to 0.9006 against L1's 0.9000, so a clean step was revised by a margin of
# six ten-thousandths, decided by floating-point noise in the error cost.
NARROW_PASS_LAYERS: frozenset[int] = frozenset({1})


def pass_score(sig: Signal) -> float:
    return sig.confidence


def fuse(records: list[SignalRecord]) -> FusedView:
    if not records:
        raise ValueError("fuse() requires at least one SignalRecord.")

    # UNSURE carries no information about the step. A layer that failed to parse
    # its own reply, or found no evidence to check against, reports the midpoint
    # of pass_score — and reading that 0.5 as "50% likely bad" is what drove
    # nearly every revision in the n=15 Ollama run. A narrow layer's PASS is
    # likewise not evidence that the step is good overall.
    decisive = [r for r in records
                if r.signal.verdict is Verdict.FAIL
                or (r.signal.verdict is Verdict.PASS
                    and r.layer_id not in NARROW_PASS_LAYERS)]

    hard_fail = next((r for r in records
                      if r.signal.verdict is Verdict.FAIL
                      and (r.layer_id in DETERMINISTIC_LAYERS
                           or (r.signal.raw or {}).get("deterministic"))), None)

    if hard_fail is not None:
        chosen = hard_fail
    else:
        # Prefer the most-informed layer that actually decided. Ranking by
        # confidence instead would always pick the most *optimistic* layer,
        # since pass_score is P(step is good).
        pool = decisive or records
        specialists = [r for r in pool if r.layer_id >= 3]
        chosen = max(specialists or pool, key=lambda r: r.layer_id)

    flaggers = [r for r in records
                if r.signal.suspected_error_type != ErrorType.NONE]
    if flaggers:
        top = max(flaggers, key=lambda r: (r.layer_id, r.signal.confidence))
        suspected = top.signal.suspected_error_type
    else:
        suspected = ErrorType.NONE

    s = chosen.signal
    return FusedView(verdict=s.verdict, confidence=s.confidence,
                     suspected_error_type=suspected, pass_score=pass_score(s),
                     source_layer_id=chosen.layer_id,
                     decisive=bool(decisive),
                     hard_fail=hard_fail is not None)


@dataclass
class _State:
    budget: int
    step_index: int = 0
    revise_count: int = 0
    records: list[SignalRecord] = field(default_factory=list)
    ran: set[int] = field(default_factory=set)
    spent: Cost = field(default_factory=Cost)
    reversibility: float = 1.0
    task_stakes: float = 0.5
    position: float = 0.0
    n_factual_claims: int = 0
    # Agent steps still available *after* this one. A revision consumes one of
    # them, so this is the scarcity term for the accept/revise decision.
    steps_remaining: int = 1
    # Layer ids actually registered on the orchestrator. Escalation targets are
    # picked from here, so an ablated layer redirects instead of dead-ending.
    available: set[int] = field(default_factory=lambda: {1, 2, 3, 4})
    traces: list["DecisionTrace"] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return self.budget - self.spent.tokens

    @property
    def budget_remaining_frac(self) -> float:
        return self.remaining_budget / self.budget if self.budget else 0.0

    def can_afford(self, tokens: int) -> bool:
        return self.remaining_budget >= tokens

    def record(self, layer_id: int, signal: Signal) -> None:
        self.records.append(SignalRecord(layer_id, signal))
        self.ran.add(layer_id)
        self.spent = self.spent + signal.cost


class Action(str, Enum):
    STOP_PASS         = "stop_pass"
    STOP_FAIL_REVISE  = "stop_fail_revise"
    CALL_LAYER_3      = "call_layer_3"
    CALL_LAYER_4      = "call_layer_4"


_STOP_ACTIONS = {Action.STOP_PASS, Action.STOP_FAIL_REVISE}
_ACTION_TO_LAYER = {Action.CALL_LAYER_3: 3, Action.CALL_LAYER_4: 4}


class BaseRouter(ABC):

    @abstractmethod
    def decide(self, state: _State) -> Action:
        ...

    def stop_action(self, state: _State) -> Action:
        """Accept-or-revise, for when escalation is impossible (target already
        run, unaffordable, or unregistered). Split out so the orchestrator's
        fallback path and the router's own terminal branch share one rule
        instead of each carrying a private constant.

        The base rule is the fixed 0.5 on the fused pass-score — deliberate for
        the fixed-threshold baseline; `AdaptiveThresholdRouter` derives it."""
        fused = fuse(state.records)
        if not fused.decisive:
            return Action.STOP_PASS
        return (Action.STOP_PASS if fused.pass_score >= 0.5
                else Action.STOP_FAIL_REVISE)

    # Abstention is not a decision rule with a threshold, so it does not belong
    # in the cost model: there is no quantity to compare. Every layer returning
    # UNSURE means we never obtained evidence, and spending an agent step to
    # revise on no evidence is a pure loss — it cannot improve the step and it
    # takes away a step the agent needs to reach an answer. Escalation is still
    # allowed while abstaining (more evidence is exactly what we want); only the
    # terminal accept/revise choice defaults to accept.


class FixedThresholdRouter(BaseRouter):

    def __init__(self, tau_low: float = 0.3, tau_high: float = 0.8,
                 escalation_cost: dict[int, int] | None = None):
        assert 0.0 <= tau_low <= tau_high <= 1.0
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.escalation_cost = escalation_cost or {3: 300, 4: 800}

    def decide(self, state: _State) -> Action:
        fused = fuse(state.records)
        ps = fused.pass_score

        if ps >= self.tau_high:
            return Action.STOP_PASS
        if ps <= self.tau_low:
            return Action.STOP_FAIL_REVISE

        target = 3 if fused.suspected_error_type == ErrorType.FACTUAL else 4

        if target in state.ran or not state.can_afford(self.escalation_cost.get(target, 0)):
            return self.stop_action(state)

        return Action.CALL_LAYER_3 if target == 3 else Action.CALL_LAYER_4


@dataclass
class DecisionTrace:
    """One router decision. `tau` is the *escalation* threshold (spend more?);
    `tau_accept` is the *accept* threshold (is this step good enough to run?).
    Both are re-derived per call from the cost model — neither is stored."""
    layer_id: int
    tau: float
    calibrated_confidence: float
    error_cost: float
    budget_remaining: int
    action: str
    target_layer: int | None = None
    tau_accept: float = 0.0
    raw_confidence: float = 0.0
    step_index: int = 0
    steps_remaining: int = 0
    suspected_error_type: str = ErrorType.NONE.value
    # Whether the layer this decision was derived from could parse its own reply.
    # None for layers that do not parse anything (L1, L3). Logged because a parse
    # failure is indistinguishable from a genuine abstention in every other
    # recorded field -- both surface as confidence 0.5 and a non-decisive fuse --
    # and on the first n=5 Ollama run that hid a 100% Layer 4 parse-failure rate
    # behind what looked like a working cascade.
    parse_ok: bool | None = None


class AdaptiveThresholdRouter(BaseRouter):

    def __init__(self, cost_model: CostModel | None = None,
                 calibrator: LayeredCalibrator | None = None,
                 escalation_cost: dict[int, int] | None = None,
                 factual_claim_min: int = 1,
                 revise_bias: float = 1.0):
        self.cost_model = cost_model or default_cost_model()
        self.calibrator = calibrator or LayeredCalibrator.default()
        # Affordability uses the *same* per-layer costs the thresholds are
        # derived from, so a fitted cost_model.json moves both together. A
        # private constant here would let the router derive tau from measured
        # costs while refusing escalations on declared ones.
        self.escalation_cost = escalation_cost or {
            lid: int(self.cost_model.layer_cost(lid)) for lid in _ACTION_TO_LAYER.values()
        }
        # Route to L3 (factual specialist) when the step's Thought carries at
        # least this many verifiable factual claims — not only when L2 happens
        # to diagnose FACTUAL (which fires rarely).
        self.factual_claim_min = factual_claim_min
        # Multiplier on the revision cost, exposed purely so the accept
        # threshold can be swept without editing the derivation. >1 makes
        # revising look more expensive (more lenient), <1 more aggressive.
        self.revise_bias = float(revise_bias)

    # -- derivation ---------------------------------------------------------
    #
    # Both thresholds come out of the same expected-cost comparison. Writing
    # p = P(step is good) = calibrated confidence, E = error_cost:
    #
    #   escalate  iff  cost(layer) < (1-p) * E * efficacy(layer)
    #   revise    iff  cost(revision) < (1-p) * E * revise_efficacy
    #
    # (For revise: accepting a bad step costs (1-p)E; revising costs c_rev and
    # still leaves (1-p)E(1-eff) unrepaired, so the eff-weighted saving must
    # beat c_rev.) Rearranged, each becomes "p < 1 - c/(E*eff)".
    #
    # Costs are normalized against the scarce resource each one consumes:
    # layers spend the token budget, revisions spend the step budget.

    def _ctx(self, state: _State) -> RouteContext:
        return RouteContext(
            reversibility=state.reversibility,
            task_stakes=state.task_stakes,
            position=state.position,
            budget_remaining=state.budget_remaining_frac,
        )

    def _calibrated_confidence(
            self, state: _State,
            fused: FusedView) -> tuple[int, float, float, bool | None]:
        """The `p` in the derivation must be the *fused* pass-score's source, not
        whichever layer happened to run last. Reading `records[-1]` meant a
        Layer 1 FAIL could be recorded in `fused` and then ignored by the
        threshold, because Layer 2 ran afterwards and reported 1.0 — the fused
        verdict and the number the thresholds compared against were two different
        signals."""
        src = next((r for r in reversed(state.records)
                    if r.layer_id == fused.source_layer_id), state.records[-1])
        raw = src.signal.raw or {}
        raw_conf = raw.get("raw_confidence", src.signal.confidence)
        return (src.layer_id, raw_conf,
                self.calibrator.calibrate(src.layer_id, raw_conf),
                raw.get("parse_ok"))

    def _escalation_threshold(self, state: _State, target: int,
                              error_cost: float) -> float:
        c_next = self.cost_model.layer_cost(target) / max(state.remaining_budget, EPS)
        delta = self.cost_model.layer_efficacy(target)
        return min(1.0, max(0.0, 1.0 - c_next / max(error_cost * delta, EPS)))

    def _accept_threshold(self, state: _State, error_cost: float) -> float:
        # `revise_count` is how many revisions this episode has already spent.
        # Pricing the nth revision like the first is what let one trajectory
        # re-block `Finish['unknown']` at steps 6, 8, 10 and 12 — see
        # `CostModel.revision_cost`.
        c_rev = ((self.cost_model.revision_cost(state.revise_count) * self.revise_bias)
                 / max(float(state.steps_remaining), EPS))
        delta = self.cost_model.revise_efficacy()
        return min(1.0, max(0.0, 1.0 - c_rev / max(error_cost * delta, EPS)))

    def _evaluate(self, state: _State, *,
                  allow_escalation: bool) -> tuple[Action, DecisionTrace]:
        fused = fuse(state.records)
        L, raw_conf, cal_conf, parse_ok = self._calibrated_confidence(state, fused)

        prefer_l3 = (fused.suspected_error_type == ErrorType.FACTUAL
                     or state.n_factual_claims >= self.factual_claim_min)
        # The suspected error type sets the *preference order*, never whether to
        # escalate. If the preferred specialist is ablated or already run, fall
        # through to the other one rather than dead-ending on it.
        order = (3, 4) if prefer_l3 else (4, 3)
        candidates = [t for t in order
                      if t in state.available and t not in state.ran]
        target = candidates[0] if candidates else order[0]

        error_cost = self.cost_model.error_cost(self._ctx(state))
        tau = self._escalation_threshold(state, target, error_cost)
        tau_accept = self._accept_threshold(state, error_cost)

        # A deterministic FAIL is already settled — spending L3/L4 to confirm a
        # duplicate action or an answer that appears in no observation is pure
        # cost. This is not a threshold rule; there is no quantity to compare.
        can_escalate = (allow_escalation and not fused.hard_fail
                        and bool(candidates) and cal_conf < tau
                        and state.can_afford(self.escalation_cost.get(target, 0)))

        if can_escalate:
            action = Action.CALL_LAYER_3 if target == 3 else Action.CALL_LAYER_4
        elif not fused.decisive:
            # Every layer abstained — see BaseRouter.stop_action.
            action = Action.STOP_PASS
        else:
            action = (Action.STOP_PASS if cal_conf >= tau_accept
                      else Action.STOP_FAIL_REVISE)

        trace = DecisionTrace(
            layer_id=L, tau=tau, calibrated_confidence=cal_conf,
            error_cost=error_cost, budget_remaining=state.remaining_budget,
            action=action.value,
            target_layer=(target if can_escalate else None),
            tau_accept=tau_accept, raw_confidence=raw_conf,
            step_index=state.step_index, steps_remaining=state.steps_remaining,
            suspected_error_type=fused.suspected_error_type.value,
            parse_ok=parse_ok,
        )
        return action, trace

    def decide(self, state: _State) -> Action:
        action, trace = self._evaluate(state, allow_escalation=True)
        state.traces.append(trace)
        return action

    def stop_action(self, state: _State) -> Action:
        action, trace = self._evaluate(state, allow_escalation=False)
        state.traces.append(trace)
        return action


class Decision(str, Enum):
    PASS        = "pass"
    FAIL_REVISE = "fail_revise"


@dataclass
class OrchestratorResult:
    decision: Decision
    signal_history: list[SignalRecord]
    total_cost: Cost
    triggering_error_type: ErrorType | None
    fused: FusedView
    layers_run: list[int]
    decision_traces: list[DecisionTrace] = field(default_factory=list)


def _count_factual_claims(step: Step) -> int:
    """Number of verifiable factual claims in the step's Thought. Lazily imports
    Layer 3's claim extractor to avoid a circular import at module load."""
    if not step.thought:
        return 0
    try:
        from src.verification.layer3_retrieval import extract_claims
    except Exception:
        return 0
    return len(extract_claims(step.thought))


class Orchestrator:

    def __init__(self, layers: dict[int, Layer], router: BaseRouter,
                 baseline_chain: tuple[int, ...] = (1, 2), budget: int = 5000,
                 calibrators: dict[int, Calibrator] | None = None,
                 max_steps: int = 8,
                 mandatory_layers: tuple[int, ...] = (5,),
                 mandatory_finish_layers: tuple[int, ...] = ()):
        self.layers = layers
        self.router = router
        self.baseline_chain = tuple(baseline_chain)
        self.budget = budget
        self.calibrators = calibrators if calibrators is not None else default_calibrators()
        self.max_steps = max_steps
        # Gates that must run — when applicable — before ANY accept/revise
        # becomes final, whatever the router would have decided on the cheap
        # layers alone. This is not a threshold rule (there is no quantity to
        # compare), so it is deliberately not in the cost model; it is a
        # declarative "this step may not be committed unverified" list.
        #
        # It exists for Layer 5. The escalation path cannot carry it: on a
        # `Finish` step Layer 2 returns a decisive PASS at confidence 1.0 in 43
        # of 303 decisions on the distractor run, and the router then stops
        # before any specialist runs. The one step where being wrong is
        # irreversible would be the one the cascade skips.
        self.mandatory_layers = tuple(mandatory_layers)
        # Additional mandatory gates that apply ONLY to Finish[...] steps.
        # Layer 4 (reasoning judge) is the default: Finish is the one step
        # where a reasoning error (wrong conclusion drawn from evidence) is
        # irreversible, and L2's rubric cannot catch arithmetic mistakes or
        # invalid inferences. Budget=5000 makes tau≈0.956, so L2's conf=1.0
        # always passes without escalation — adding L4 here ensures the
        # reasoning check runs regardless of L2's score.
        self.mandatory_finish_layers = tuple(mandatory_finish_layers)

    def reset(self) -> None:
        for layer in self.layers.values():
            layer.reset()

    def commit(self, step: Step) -> None:
        """Tell the layers the step really executed. Must be called by the agent
        *after* the environment call, so per-episode state (e.g. Layer 1's action
        history) records executed actions rather than attempted ones."""
        for layer in self.layers.values():
            layer.commit(step)

    def verify(self, step: Step, context: Context | None = None) -> OrchestratorResult:
        context = context or Context()
        position = min(1.0, step.step_index / max(1, self.max_steps))
        applicable = {lid for lid, layer in self.layers.items()
                      if layer.applicable(step, context)}
        # On a Finish step, promote mandatory_finish_layers into the active
        # mandatory set. This runs the reasoning judge (L4) on every answer
        # commitment, since that is the one step where an undetected reasoning
        # error is irreversible — regardless of what the cheap layers scored.
        is_finish = step.action.strip().lower().startswith("finish[")
        active_mandatory = self.mandatory_layers
        if is_finish:
            active_mandatory = active_mandatory + tuple(
                lid for lid in self.mandatory_finish_layers
                if lid not in active_mandatory)
        state = _State(budget=self.budget, step_index=step.step_index,
                       revise_count=context.revise_count,
                       reversibility=step.reversibility,
                       task_stakes=step.task_stakes,
                       position=position,
                       n_factual_claims=_count_factual_claims(step),
                       steps_remaining=max(1, self.max_steps - step.step_index),
                       available=set(applicable))

        remaining_chain = [l for l in self.baseline_chain
                           if l in applicable and l not in state.ran]
        for lid in self.baseline_chain:
            if lid not in applicable or lid in state.ran:
                continue
            if not state.can_afford(0):
                break
            self._run(lid, step, context, state)
            # Check whether this is the last runnable baseline layer.
            is_last_in_chain = all(
                l in state.ran or l not in applicable
                for l in self.baseline_chain if l != lid
            )
            action = self.router.decide(state)
            if action in _STOP_ACTIONS:
                action, final = self._settle(action, step, context, state,
                                             applicable, active_mandatory)
                if final:
                    # On a decisive FAIL (hard_fail or deterministic), stop
                    # immediately — no subsequent baseline layer can overturn it.
                    # On a mid-chain PASS, continue so remaining baseline layers
                    # run before the router commits. Skipping them is safe only
                    # when the router can escalate; without escalation targets
                    # (e.g. --no-layer3 --no-layer4) L2 would be permanently
                    # skipped every time L1 passes, defeating the baseline chain.
                    fused_now = fuse(state.records)
                    if fused_now.hard_fail or is_last_in_chain:
                        return self._finish(action, state)
                    # else: continue to the next baseline layer
                    continue
                # A mandatory gate just ran and the router now wants more
                # evidence. Leave the baseline chain to the escalation loop.
                break

        # Mandatory gates run right after the baseline chain, BEFORE any
        # escalation. Deferring them to the terminal path meant L3 and L4 were
        # both spent first: on a Finish step Layer 2 abstains often enough that
        # the router escalates twice, and only once both specialists were
        # exhausted did the gate get to speak — after which its deterministic
        # rejection made both of those calls retroactively pointless.
        for lid in self._pending_mandatory(state, applicable, active_mandatory):
            self._run(lid, step, context, state)

        # No layer could run at all (everything ablated or inapplicable). There is
        # nothing to fuse, and a step must not be revised for lack of a verifier.
        if not state.records:
            return OrchestratorResult(
                decision=Decision.PASS, signal_history=[], total_cost=state.spent,
                triggering_error_type=None,
                fused=FusedView(Verdict.UNSURE, 0.5, ErrorType.NONE, 0.5, 0,
                                decisive=False),
                layers_run=[], decision_traces=list(state.traces))

        while True:
            action = self.router.decide(state)
            if action in _STOP_ACTIONS:
                action, final = self._settle(action, step, context, state,
                                             applicable, active_mandatory)
                if final:
                    return self._finish(action, state)
                continue
            target = _ACTION_TO_LAYER[action]
            if (target in state.ran or target not in applicable
                    or not state.can_afford(0)):
                # The router wants a layer we cannot run (ablated away, already
                # run, or out of budget). Fall back to its accept/revise rule
                # rather than a constant of our own.
                stop, _ = self._settle(self.router.stop_action(state),
                                       step, context, state, applicable,
                                       active_mandatory)
                return self._finish(stop, state)
            self._run(target, step, context, state)

    def _pending_mandatory(self, state: _State, applicable: set[int],
                           active_mandatory: tuple[int, ...] | None = None
                           ) -> list[int]:
        layers = active_mandatory if active_mandatory is not None else self.mandatory_layers
        return [lid for lid in layers
                if lid in applicable and lid in self.layers and lid not in state.ran]

    def _settle(self, action: Action, step: Step, context: Context,
                state: _State, applicable: set[int],
                active_mandatory: tuple[int, ...] | None = None) -> tuple[Action, bool]:
        """Run any pending mandatory gate before a stop action is honored.

        Returns `(action, is_final)`. `is_final` is False when a gate ran and the
        router now wants to escalate instead of stopping — the caller re-enters
        the escalation loop. Terminates because every layer runs at most once
        (`state.ran`), so the pending set strictly shrinks."""
        while action in _STOP_ACTIONS:
            pending = self._pending_mandatory(state, applicable, active_mandatory)
            if not pending:
                return action, True
            for lid in pending:
                self._run(lid, step, context, state)
            action = self.router.decide(state)
        return action, False

    def _run(self, lid: int, step: Step, context: Context, state: _State) -> None:
        signal = self.layers[lid].run(step, context)
        state.record(lid, self._calibrate(lid, signal))

    def _calibrate(self, lid: int, signal: Signal) -> Signal:
        calibrator = self.calibrators.get(lid)
        if calibrator is None:
            return signal
        raw_conf = signal.confidence
        cal_conf = calibrator.apply(raw_conf)
        raw = dict(signal.raw) if signal.raw else {}
        raw["raw_confidence"] = raw_conf
        return replace(signal, confidence=cal_conf, raw=raw)

    def _finish(self, action: Action, state: _State) -> OrchestratorResult:
        fused = fuse(state.records)
        if action == Action.STOP_PASS:
            decision, trigger = Decision.PASS, None
        else:
            decision = Decision.FAIL_REVISE
            # Was: `else ErrorType.FORMAT`, which labelled every reasonless
            # revision a format error. Nothing was wrong with the format — no
            # layer had flagged anything — so the log said `[Verify:format]` on
            # steps whose grammar was fine, and the agent was pointed at the
            # wrong thing. NONE here means "flagged, cause unattributed".
            trigger = fused.suspected_error_type
        return OrchestratorResult(
            decision=decision,
            signal_history=list(state.records),
            total_cost=state.spent,
            triggering_error_type=trigger,
            fused=fused,
            layers_run=sorted(state.ran),
            decision_traces=list(state.traces),
        )


def _print(title: str, res: OrchestratorResult) -> None:
    print(f"===== {title} =====")
    for rec in res.signal_history:
        s = rec.signal
        print(f"  L{rec.layer_id}: verdict={s.verdict.value:<6} "
              f"conf={s.confidence:.2f} suspect={s.suspected_error_type.value:<9} "
              f"tokens={s.cost.tokens}")
    print(f"  -> decision={res.decision.value}  layers_run={res.layers_run}  "
          f"trigger={res.triggering_error_type.value if res.triggering_error_type else '-'}  "
          f"fused(src=L{res.fused.source_layer_id}, ps={res.fused.pass_score:.2f})  "
          f"cost={res.total_cost.tokens} tok\n")


def _demo() -> None:
    router = FixedThresholdRouter(tau_low=0.3, tau_high=0.8)

    def mocks(l1: Signal, l2: Signal, l3: Signal, l4: Signal) -> dict[int, Layer]:
        return {1: MockLayer(1, l1), 2: MockLayer(2, l2),
                3: MockLayer(3, l3), 4: MockLayer(4, l4)}

    step = Step(action="Search[High Plains (United States)]",
                thought="Find the elevation range of the High Plains.", step_index=1)

    layers_a = mocks(
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.PASS, 0.7, ErrorType.NONE, Cost(120, 20)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200)))
    _print("(a) cheap PASS at L1", Orchestrator(layers_a, router).verify(step))

    layers_b = mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(300, 90)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200)))
    _print("(b) middle-band → escalate L3", Orchestrator(layers_b, router).verify(step))

    layers_c = mocks(
        Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1)),
        Signal(Verdict.FAIL, 0.4, ErrorType.FACTUAL, Cost(120, 20)),
        Signal(Verdict.FAIL, 0.1, ErrorType.FACTUAL, Cost(300, 90)),
        Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200)))
    _print("(c) disagreement → fusion resolves", Orchestrator(layers_c, router).verify(step))


if __name__ == "__main__":
    _demo()
