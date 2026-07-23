from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.llm import LLM
from src.agents.prompts import SYSTEM_PROMPT
from src.tools.wikipedia import WikiEnv
from src.verification.layer1_rule import Layer1Adapter, RuleBasedVerifier
from src.verification.layer2_rubric import Layer2RubricVerifier
from src.verification.layer4_llm_judge import Layer4LLMJudge
from src.verification.layer5_answer import Layer5AnswerGate, finish_argument
from src.verification.orchestrator import (
    AdaptiveThresholdRouter, BaseRouter, Context, Decision, ErrorType,
    Orchestrator, Step, Verdict,
)
from src.verification.routing_config import RoutingConfig, load_routing_config

# Verification LLM calls are one-shot rather than continuing the ReAct
# transcript, so they get their own small system prompt and the layer owns the
# response format.
VERIFY_SYSTEM = ("You are a meticulous verifier for a reasoning agent. "
                 "Follow the instructions exactly and output only what is asked.")

# Both LLM layers ask for one flat JSON object, so the verify call gets a prefill
# /stop contract of its own rather than being left unconstrained. Unconstrained,
# an 8B model echoed the prompt's own format instructions back and then emitted
# several JSON objects in a row -- on the n=5 Ollama run that made Layer 4
# unparseable on 5/5 replies and Layer 2 on 8/15. Prefilling the opening brace
# removes the preamble at the source; stopping at the closing brace ends the
# reply after the first object. The prefill is not echoed back by any backend, so
# it has to be reattached here.
_JSON_PREFILL = "{"
_JSON_STOP = ["}"]


@dataclass
class AdaptiveReActResult:
    question: str
    prediction: str | None
    trajectory: str
    n_steps: int
    n_react_calls: int              # LLM calls spent generating Thought/Action
    n_verify_calls: int             # LLM calls spent inside verification (L2/L4)
    n_llm_calls: int                # n_react_calls + n_verify_calls
    finished: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    gold: str | None = None
    n_revisions: int = 0            # steps the orchestrator sent back to revise
    layer_runs: dict[int, int] = field(default_factory=dict)  # layer_id -> times run
    verify_est_tokens: int = 0      # orchestrator's own (estimated) token spend
    decision_traces: list = field(default_factory=list)
    # layer_id -> times that layer failed to parse its own LLM reply, over
    # layer_id -> times it tried. A parse failure degrades to UNSURE at
    # confidence 0.5, which `fuse()` treats as non-decisive: the layer is billed
    # in full and contributes nothing. Without this tally that is invisible --
    # the first n=5 Ollama run had Layer 4 failing 5/5 and read as a healthy
    # cascade whose specialists happened to abstain.
    layer_parse_fail: dict[int, int] = field(default_factory=dict)
    layer_parse_total: dict[int, int] = field(default_factory=dict)
    # Per-step, per-layer signal data for calibration fitting.
    # Each entry: {step_index, layer_id, verdict, raw_confidence, parse_ok}
    # raw_confidence is the layer's own score BEFORE orchestrator calibration.
    # `parse_ok` is None for layers that don't parse (L1/L3) and True/False for
    # LLM layers; parse failures are kept so `fit_calibration.py` can exclude them.
    layer_signals: list[dict] = field(default_factory=list)


def build_default_orchestrator(
    llm: LLM,
    on_verify_call,
    *,
    router: BaseRouter | None = None,
    routing: RoutingConfig | None = None,
    revise_bias: float = 1.0,
    budget: int = 5000,
    max_steps: int = 8,
    use_layer4: bool = True,
    use_layer3: bool = True,
    use_layer5: bool = True,
    nli_model: str | None = None,
) -> Orchestrator:
    """Wire real layers over `llm`. L1 (rules) and L2 (rubric) form the baseline
    chain; L4 (LLM judge) and — when `use_layer3` — L3 (retrieval + DeBERTa NLI)
    are escalation targets. The orchestrator falls back gracefully when the
    router asks for a layer that is not registered.

    `routing` carries the fitted calibration + cost model (see
    `src.verification.routing_config`); its calibrator is handed to *both* the
    router and `Orchestrator._calibrate` so the pass_score `fuse()` reports and
    the `p` the thresholds are derived from are the same number. An explicit
    `router` overrides the derived one, but still shares the calibrator.
    """
    routing = routing or load_routing_config()

    def _one_shot(prompt: str) -> str:
        on_verify_call()
        body = llm.generate(VERIFY_SYSTEM, prompt, _JSON_PREFILL, _JSON_STOP)
        return f"{_JSON_PREFILL}{body}}}".strip()

    def _one_shot_free(prompt: str) -> str:
        """Unconstrained one-shot — no stop-sequence contract.

        Layer 5's `quote` field is copied verbatim from Wikipedia evidence and
        can contain `}` characters (e.g. inside parenthetical asides). The
        `_JSON_STOP=["}"]` contract fires on that first inner `}`, truncating
        the reply mid-string and causing a parse failure. The free variant
        lets the model generate a complete reply and relies on
        `first_json_object` to extract the first well-formed object — exactly
        what was done before the stop contract was added for the lighter layers.
        L5 only runs on Finish steps (~1-2 per trajectory), so the extra
        tokens are negligible."""
        on_verify_call()
        return llm.generate(VERIFY_SYSTEM, prompt, "", [])

    layers: dict = {
        1: Layer1Adapter(RuleBasedVerifier()),
        2: Layer2RubricVerifier(_one_shot),
    }
    if use_layer3:
        from src.verification.layer3_retrieval import Layer3RetrievalVerifier
        from src.verification.nli_deberta import DEFAULT_MODEL, get_cached_nli
        # Build the NLI model eagerly so the (slow) load/download happens once,
        # up front, rather than mid-trajectory on the first FACTUAL escalation.
        nli_fn = get_cached_nli(nli_model or DEFAULT_MODEL)
        layers[3] = Layer3RetrievalVerifier(nli_fn)
    if use_layer4:
        layers[4] = Layer4LLMJudge(_one_shot)
    if use_layer5:
        # L5 gets the unconstrained generator: its `quote` field is copied
        # verbatim from Wikipedia and can contain `}`, which would truncate
        # the `_JSON_STOP` variant prematurely (see `_one_shot_free` above).
        layers[5] = Layer5AnswerGate(_one_shot_free)

    return Orchestrator(
        layers=layers,
        router=router or AdaptiveThresholdRouter(
            cost_model=routing.cost_model,
            calibrator=routing.calibrator,
            revise_bias=revise_bias,
        ),
        baseline_chain=(1, 2),
        budget=budget,
        max_steps=max_steps,
        calibrators=routing.calibrator.as_dict(),
        # L5 is a gate, not an escalation target: it must run on every Finish
        # step even when L1/L2 already produced a decisive PASS.
        mandatory_layers=(5,) if use_layer5 else (),
        # L4 (reasoning judge) is NOT run mandatorily on every Finish step.
        # Measured on n=30 distractor: L4 FAILed 24/32 times (75%) but had a
        # 21% false-positive rate (5/24 FAILs were correct answers). The current
        # fuse() picks L5 (layer_id=5) over L4 (layer_id=4) when both run, so
        # mandatory L4 costs tokens without changing outcomes. L4 still runs via
        # escalation when L2 confidence is low enough to cross the router's tau.
        mandatory_finish_layers=(),
    )


def _canon_action(action: str) -> str:
    """Same normalization Layer 1 uses for duplicate detection, so the block
    counter and the duplicate rule agree on what "the same action" means."""
    m = re.match(r"^(\w+)\[(.*)\]$", (action or "").strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        return re.sub(r"\s+", " ", (action or "").strip().lower())
    arg = re.sub(r"\s+", " ", m.group(2).strip().lower())
    return f"{m.group(1).lower()}[{arg}]"


def _extract_goal_entities(goal: str) -> list[str]:
    """Return capitalized multi-word phrases from the goal that are likely
    named entities.  Used to give the agent concrete Search[X] suggestions
    when a Finish step is blocked and the layer feedback is generic.

    The approach is deliberately simple: HotpotQA questions are short and
    their entities are almost always proper nouns that start with a capital
    letter. False positives (sentence starters, generic nouns) are filtered
    by a small stopword set. A missed entity is never harmful — the directive
    still includes the generic form as a fallback.
    """
    _STOP = {"What", "Who", "When", "Where", "Why", "How", "Which", "Did",
              "Do", "Does", "Is", "Was", "Are", "Were", "Will", "The", "A", "An",
              "In", "At", "On", "Of", "For", "To", "And", "Or", "But",
              "Both", "Either", "Same", "First", "Last", "Other", "Their",
              "His", "Her", "Its", "This", "That", "These", "Those"}
    # Match sequences of capitalized tokens (handles "New York", "J. K. Rowling")
    spans = re.findall(
        r'\b([A-Z][a-zA-Z\'\.]+(?:\s+[A-Z][a-zA-Z\'\.]+)*)\b', goal or "")
    seen: set[str] = set()
    out: list[str] = []
    for s in spans:
        # Trim any leading stopwords so "Were Scott Derrickson" → "Scott Derrickson"
        # while "The United States" → "United States".
        words = s.split()
        while words and words[0] in _STOP:
            words = words[1:]
        s = " ".join(words)
        if not s or s in _STOP or len(s) <= 1:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _directive(action: str, n_blocks: int, goal: str = "") -> str:
    """What to do instead, in the agent's own action vocabulary.

    A verdict alone changed nothing: across the two n=30 adaptive runs the
    retries did switch tool or target 87-89% of the time and still converted 0
    blocks into a correct answer, because nothing told the agent *which* other
    move to make. The second and later block on the same action says so
    explicitly — repeating the identical complaint is what produced
    `Search[Pandikona]` / `Search[Berger Blanc Suisse]` five times in a row.

    For Finish steps, `goal` is used to suggest concrete Search[entity] moves
    rather than a generic placeholder, giving the agent something to act on."""
    a = (action or "").strip()
    low = a.lower()
    repeat = ""
    if n_blocks >= 2:
        repeat = (f" You have now been stopped {n_blocks} times on this same "
                  f"action — do NOT emit it again; change the entity or the tool.")

    if low.startswith("finish["):
        # Extract the proposed answer from Finish[<answer>]
        proposed = a[len("finish["):-1].strip() if a.endswith("]") else ""
        # Suggest concrete Search targets derived from the goal question
        entities = _extract_goal_entities(goal)
        if entities:
            suggestions = " or ".join(f"Search[{e}]" for e in entities[:2])
            base = (f"Do not commit \"{proposed}\" as the final answer yet — it was "
                    f"not verified by the evidence. Retrieve the specific fact first: "
                    f"{suggestions}, then Lookup[<the attribute the question asks for>].")
        elif proposed and proposed.lower() not in {"", "none", "unknown", "i do not know"}:
            base = (f"Do not commit \"{proposed}\" — it was not verified. "
                    "Name the specific fact the question asks for and retrieve it first: "
                    "Search[<entity from the question>] or Lookup[<keyword>] in the "
                    "page already open.")
        else:
            base = ("Do not commit an answer yet: find the fact first with "
                    "Search[<entity from the question>] or Lookup[<keyword>] in the "
                    "page already open.")
        return base + repeat

    if low.startswith("lookup["):
        return ("Either Lookup a different keyword that would appear verbatim in "
                "the page, or Search[<the other entity named in the question>]."
                + repeat)
    if low.startswith("search["):
        entities = _extract_goal_entities(goal)
        # Suggest the other entities in the goal, not the one just searched
        searched_arg = a[len("search["):-1].strip().lower() if a.endswith("]") else ""
        alts = [e for e in entities if e.lower() != searched_arg]
        if alts:
            alt_str = " or ".join(f"Search[{e}]" for e in alts[:2])
            return (f"Search a different entity: {alt_str}. "
                    "Or Lookup[<keyword>] inside the page you already retrieved."
                    + repeat)
        return ("Search a different entity named in the question, or Lookup[<keyword>] "
                "inside the page you already retrieved instead of searching again."
                + repeat)
    return ("Emit a valid action: Search[<entity>], Lookup[<keyword>] or "
            "Finish[<answer>]." + repeat)


def _stakes_for(action: str) -> tuple[float, float]:
    """Finish[...] commits the final answer: low reversibility, high stakes.
    Ordinary Search/Lookup steps are fully reversible and lower-stakes."""
    if action.strip().lower().startswith("finish["):
        return 0.2, 0.9      # reversibility, task_stakes
    return 1.0, 0.4


class AdaptiveReActAgent:
    """ReAct agent whose every step is checked by the adaptive verification
    orchestrator *before* it touches the environment. A FAIL_REVISE decision
    injects a `[Verify]` observation and skips the env call so the model can
    self-correct — mirroring how the Layer-1-only agent injects `[Layer1]`."""

    def __init__(self, llm: LLM, env: WikiEnv, orchestrator: Orchestrator | None = None,
                 max_steps: int = 8, verbose: bool = False, use_layer4: bool = True,
                 use_layer3: bool = True, use_layer5: bool = True,
                 routing: RoutingConfig | None = None,
                 budget: int = 5000, revise_bias: float = 1.0):
        self.llm = llm
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose
        self._n_verify_calls = 0
        self.orchestrator = orchestrator or build_default_orchestrator(
            llm, self._count_verify_call, max_steps=max_steps,
            use_layer4=use_layer4, use_layer3=use_layer3, use_layer5=use_layer5,
            routing=routing, budget=budget, revise_bias=revise_bias)

    def _count_verify_call(self) -> None:
        self._n_verify_calls += 1

    def run(self, question: str, gold: str | None = None) -> AdaptiveReActResult:
        self.env.reset()
        self.orchestrator.reset()
        self._n_verify_calls = 0

        user = f"Question: {question}"
        traj = ""
        prev_obs = ""
        observations: list[str] = []      # evidence pool for Layer 3 retrieval/NLI
        prediction, finished, n_react = None, False, 0
        n_revisions, verify_est_tokens = 0, 0
        layer_runs: dict[int, int] = {}
        parse_fail: dict[int, int] = {}
        parse_total: dict[int, int] = {}
        traces: list = []
        layer_signals: list[dict] = []
        # Normalized action -> times it has been blocked this episode. Drives the
        # escalating directive in `_feedback`: repeating the same generic
        # complaint produced the same action again, e.g. `Finish['unknown']`
        # blocked four times in one trajectory.
        n_blocks: dict[str, int] = {}
        tok_in0, tok_out0 = self.llm.prompt_tokens, self.llm.completion_tokens

        i = 0
        for i in range(1, self.max_steps + 1):
            out = self.llm.generate(
                SYSTEM_PROMPT, user, traj + f"Thought {i}:",
                stop=[f"\nObservation {i}:"],
            ).strip()
            n_react += 1

            thought, action = self._parse(out, i)
            if action is not None:
                action = action.split("\n")[0].strip()
            if action is None:
                action = self.llm.generate(
                    SYSTEM_PROMPT, user,
                    traj + f"Thought {i}: {thought}\nAction {i}:",
                    stop=["\n"],
                ).strip()
                n_react += 1

            rev, stakes = _stakes_for(action)
            step = Step(action=action, thought=thought, goal=question,
                        prev_observation=prev_obs, step_index=i,
                        reversibility=rev, task_stakes=stakes)
            # Feed the accumulated observations as the evidence pool for Layer 3
            # (claim checking) and Layer 5 (answer sufficiency); the current
            # step's observation does not exist yet (we verify first).
            #
            # `revise_count` was left at its default 0 on every call, so the
            # router could not see that it had already spent revisions on this
            # trajectory and priced the fifth block exactly like the first. That
            # is how one question absorbed five consecutive blocks and ran out of
            # steps with no answer.
            ctx = Context(revise_count=n_revisions,
                          scratch={"observations": list(observations)})
            result = self.orchestrator.verify(step, ctx)

            for rec in result.signal_history:
                layer_runs[rec.layer_id] = layer_runs.get(rec.layer_id, 0) + 1
                ok = (rec.signal.raw or {}).get("parse_ok")
                if ok is not None:
                    lid = rec.layer_id
                    parse_total[lid] = parse_total.get(lid, 0) + 1
                    if not ok:
                        parse_fail[lid] = parse_fail.get(lid, 0) + 1
                # Collect raw (pre-calibration) confidence for calibration fitting.
                # After _calibrate() the orchestrator stashes the original under
                # raw["raw_confidence"]; fall back to signal.confidence when that
                # key is absent (L1/L3 which the calibrator touches but doesn't
                # meaningfully change, and any path that skipped calibration).
                raw_conf = (rec.signal.raw or {}).get("raw_confidence",
                                                       rec.signal.confidence)
                layer_signals.append({
                    "step_index": i,
                    "layer_id": rec.layer_id,
                    "verdict": rec.signal.verdict.value,
                    "raw_confidence": raw_conf,
                    "parse_ok": ok,
                })
            verify_est_tokens += result.total_cost.tokens
            traces.extend(result.decision_traces)

            if result.decision == Decision.FAIL_REVISE:
                n_revisions += 1
                n_blocks[_canon_action(action)] = (
                    n_blocks.get(_canon_action(action), 0) + 1)
                feedback = self._feedback(
                    result, action, n_blocks[_canon_action(action)], goal=question)
                etype = result.triggering_error_type
                # NONE means "flagged, cause unattributed" — don't print a type we
                # cannot substantiate, it used to read as `[Verify:format]`.
                tag = (f"[Verify:{etype.value}]"
                       if etype and etype != ErrorType.NONE else "[Verify]")
                obs = f"{tag} {feedback}"
                if self.verbose:
                    print(f"{tag} step {i} FAIL_REVISE: "
                          f"{feedback}  | layers={result.layers_run}")
                traj += (f"Thought {i}: {thought}\nAction {i}: {action}\n"
                         f"Observation {i}: {obs}\n")
                continue

            obs, done = self.env.step(action)
            obs = obs.replace("\\n", "")
            # The step really ran: let the layers record it. Layer 1's duplicate
            # detection and Lookup-after-Search precondition are about *executed*
            # actions, so this must not happen for revised-away steps.
            self.orchestrator.commit(step)

            if self.verbose:
                print(f"Thought {i}: {thought}")
                print(f"Action {i}: {action}")
                print(f"Observation {i}: {obs}  | verify layers={result.layers_run}\n")

            traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prev_obs = obs
            observations.append(obs)

            if done:
                prediction = self.env.answer
                finished = True
                break

        return AdaptiveReActResult(
            question=question,
            prediction=prediction,
            trajectory=traj.strip(),
            n_steps=i,
            n_react_calls=n_react,
            n_verify_calls=self._n_verify_calls,
            n_llm_calls=n_react + self._n_verify_calls,
            finished=finished,
            prompt_tokens=self.llm.prompt_tokens - tok_in0,
            completion_tokens=self.llm.completion_tokens - tok_out0,
            gold=gold,
            n_revisions=n_revisions,
            layer_runs=layer_runs,
            verify_est_tokens=verify_est_tokens,
            decision_traces=traces,
            layer_parse_fail=parse_fail,
            layer_parse_total=parse_total,
            layer_signals=layer_signals,
        )

    @staticmethod
    def _feedback(result, action: str = "", n_blocks: int = 1,
                  goal: str = "") -> str:
        """Pull the most actionable hint out of the layers that actually
        complained, then say what to do instead.

        Previously this scanned *every* signal, so a passing layer's rationale
        could be handed to the agent as the reason it failed — one trajectory was
        told `[Verify:format] The thought directly addresses the goal by
        specifying what to search for.` Only FAIL verdicts and layers that named
        an error type have standing to explain a revision.

        The remaining half of the problem is that a verdict is not a repair.
        Measured on `adaptive_n30_ollama_steps12.json`, 12 premature `Finish[...]`
        actions were blocked, all 12 of them wrong answers — and none became a
        correct answer, because what the agent got back was
        "The Thought does not follow from the previous Observation" and what it
        needed was a different retrieval. So: prefer a layer that supplied a
        `feedback` string (Layer 5 names the answer the evidence supports), and
        always append a concrete directive keyed to the blocked action, escalated
        when the same action has been blocked before.

        `goal` is passed through to `_directive` so Finish-step blocks can name
        concrete `Search[entity]` suggestions from the question rather than a
        generic placeholder."""
        complainers = [rec for rec in result.signal_history
                       if rec.signal.verdict is Verdict.FAIL
                       or rec.signal.suspected_error_type != ErrorType.NONE]
        # Rank by *standing*, not by run order. The layer that decided the fused
        # verdict speaks first, then the most-informed layer. Ordering by run
        # order let a layer escalated to *after* the answer gate shadow it: on
        # question 5ac3e0f7 the block on `Finish[none]` was explained as
        # "Claim refuted by evidence ..." (Layer 3) instead of the gate's own
        # "you are about to answer that you do not know — find the fact first".
        src = getattr(result.fused, "source_layer_id", 0)
        complainers.sort(key=lambda rec: (rec.layer_id == src, rec.layer_id),
                         reverse=True)
        reason = ""
        # `feedback` first across all complainers, before falling back to the
        # weaker keys: `feedback` is the key a layer uses when it has something
        # actionable to say, `rationale` when it only has a verdict to justify.
        for key in ("feedback", "rationale", "error"):
            for rec in complainers:
                if (rec.signal.raw or {}).get(key):
                    reason = str((rec.signal.raw or {})[key])
                    break
            if reason:
                break
        if not reason:
            reason = "This step was flagged as low-quality."
        directive = _directive(action, n_blocks, goal=goal)
        # Avoid duplicating the directive when the layer's own message already
        # contains a concrete action (Search[X], Lookup[Y], or Finish[Z]). L5's
        # repair/null/ungrounded feedbacks all end with a specific retrieval
        # suggestion; L4's new `feedback` field may too. Appending the generic
        # directive on top of those adds noise and potentially conflicting advice.
        _already_concrete = any(marker in reason
                                 for marker in ("Search[", "Lookup[", "Finish["))
        if directive and reason and not _already_concrete:
            return f"{reason} {directive}".strip()
        return reason

    @staticmethod
    def _parse(out: str, i: int) -> tuple[str, str | None]:
        out = out.strip()
        if out.startswith(f"Thought {i}:"):
            out = out[len(f"Thought {i}:"):].strip()
        marker = f"\nAction {i}:"
        if marker in out:
            thought, action = out.split(marker, 1)
            return thought.strip(), action.strip()
        marker = f"Action {i}:"
        if marker in out:
            thought, action = out.split(marker, 1)
            return thought.strip(), action.strip()
        return out.split("\n")[0].strip(), None
