from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.verification.orchestrator import (
    Context, Cost, ErrorType, Layer, Signal, Step, Verdict,
)


class NLILabel(str, Enum):
    ENTAILMENT    = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL       = "neutral"


class ClaimStatus(str, Enum):
    SUPPORTED    = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL      = "neutral"
    NO_EVIDENCE  = "no_evidence"


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str = ""
    relevance: float = 0.0


RetrieveFn = Callable[[str, list[Evidence], int], list[Evidence]]
NLIFn      = Callable[[str, str], dict]


_INTENT_PAT = re.compile(
    r"^\s*("
    r"i\s+(will|need|should|must|can|am\s+going|'?ll|want|have\s+to)"
    r"|let\s+me|let'?s|next\b|now\s+i|i'?m\s+going|first\b|then\s+i"
    r"|to\s+(find|answer|search|check|verify)"
    r"|tôi\s+(sẽ|cần|nên|muốn|phải)|mình\s+(sẽ|cần)|bước\s+tiếp\s+theo"
    r"|tiếp\s+theo|hãy\b|cần\s+(tìm|tra|kiểm)"
    r")",
    re.IGNORECASE,
)
_NUM_PAT = re.compile(r"\d")
_ASSERT_PAT = re.compile(
    r"\b(is|was|were|are|has|have|had|born|died|located|founded|released|"
    r"answer|result|means|refers|known\s+as|equals?|là|có|được|thuộc)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _has_proper_noun(sentence: str) -> bool:
    tokens = sentence.split()
    for tok in tokens[1:]:
        core = tok.strip(".,;:'\"()[]-")
        if core and core[0].isupper() and not core.isupper():
            return True
        if core.isupper() and len(core) > 1:
            return True
    return False


def is_factual_claim(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 4 or _INTENT_PAT.search(s):
        return False
    if _NUM_PAT.search(s):
        return True
    if _has_proper_noun(s):
        return True
    return bool(_ASSERT_PAT.search(s))


def extract_claims(text: str) -> list[str]:
    return [s for s in split_sentences(text) if is_factual_claim(s)]


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "with",
    "that", "this", "these", "those", "it", "its", "from", "which", "who",
    "what", "when", "where", "how", "he", "she", "they", "them", "his", "her",
    "có", "là", "và", "của", "một", "các", "những", "được", "cho", "với", "ở",
}
_WORD_PAT = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_PAT.findall(text.lower()) if t not in _STOPWORDS}


def keyword_overlap_retrieve(claim: str, pool: list[Evidence], k: int) -> list[Evidence]:
    q = _content_tokens(claim)
    if not q:
        return []
    scored: list[Evidence] = []
    for ev in pool:
        overlap = len(q & _content_tokens(ev.text))
        if overlap == 0:
            continue
        scored.append(replace(ev, relevance=overlap / len(q)))
    scored.sort(key=lambda e: (e.relevance, -len(e.text)), reverse=True)
    return scored[:k]


def _to_label(raw: object) -> NLILabel:
    try:
        return NLILabel(str(raw).strip().lower())
    except ValueError:
        return NLILabel.NEUTRAL


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def make_mock_nli(rules: list[tuple[str, str, str, float]] | None = None,
                  default_label: str = "neutral",
                  default_score: float = 0.5) -> NLIFn:
    rules = rules or []

    def _fn(premise: str, hypothesis: str) -> dict:
        p, h = premise.lower(), hypothesis.lower()
        for hyp_sub, prem_sub, label, score in rules:
            if hyp_sub.lower() in h and prem_sub.lower() in p:
                return {"label": label, "score": score}
        return {"label": default_label, "score": default_score}

    return _fn


@dataclass
class _ClaimResult:
    claim: str
    status: ClaimStatus
    strength: float
    best_source: str | None = None


class Layer3RetrievalVerifier(Layer):

    def __init__(self, nli_fn: NLIFn | None = None,
                 retrieve_fn: RetrieveFn = keyword_overlap_retrieve,
                 model_name: str | None = None,
                 k: int = 3, max_claims: int = 8,
                 contradiction_threshold: float = 0.6,
                 entailment_threshold: float = 0.6,
                 unsure_confidence: float = 0.5,
                 max_nli_tokens: int = 512, latency_per_call_ms: float = 15.0,
                 trajectory_key: str = "observations",
                 current_obs_key: str = "current_observation"):
        super().__init__(layer_id=3)
        self._nli_fn = nli_fn
        self.model_name = model_name
        self.retrieve_fn = retrieve_fn
        self.k = k
        self.max_claims = max_claims
        self.contradiction_threshold = contradiction_threshold
        self.entailment_threshold = entailment_threshold
        self.unsure_confidence = unsure_confidence
        self.max_nli_tokens = max_nli_tokens
        self.latency_per_call_ms = latency_per_call_ms
        self.trajectory_key = trajectory_key
        self.current_obs_key = current_obs_key

    def _resolve_nli(self) -> NLIFn:
        if self._nli_fn is None:
            from src.verification.nli_deberta import DEFAULT_MODEL, get_cached_nli
            self._nli_fn = get_cached_nli(self.model_name or DEFAULT_MODEL)
        return self._nli_fn

    def _collect_claims(self, step: Step, context: Context) -> list[str]:
        texts: list[str] = []
        if step.thought:
            texts.append(step.thought)
        cur_obs = context.scratch.get(self.current_obs_key) if context else None
        if cur_obs:
            texts.append(str(cur_obs))
        seen, claims = set(), []
        for text in texts:
            for c in extract_claims(text):
                key = c.lower()
                if key not in seen:
                    seen.add(key)
                    claims.append(c)
        return claims

    def _build_pool(self, step: Step, context: Context) -> list[Evidence]:
        raw_obs: list = []
        if context and self.trajectory_key in context.scratch:
            raw_obs = list(context.scratch[self.trajectory_key])

        pool: list[Evidence] = []
        for i, item in enumerate(raw_obs):
            if isinstance(item, Evidence):
                pool.append(item)
            else:
                for sent in split_sentences(str(item)):
                    pool.append(Evidence(text=sent, source=f"obs#{i}"))
        if step.prev_observation:
            for sent in split_sentences(step.prev_observation):
                pool.append(Evidence(text=sent, source="prev_obs"))

        seen, deduped = set(), []
        for ev in pool:
            if ev.text and ev.text not in seen:
                seen.add(ev.text)
                deduped.append(ev)
        return deduped

    def _truncate_pair(self, evidence_text: str, claim: str) -> tuple[str, str]:
        budget_chars = self.max_nli_tokens * 4
        claim_cap = budget_chars // 2
        claim = claim[:claim_cap]
        ev_budget = max(0, budget_chars - len(claim) - 16)
        return evidence_text[:ev_budget], claim

    def applicable(self, step: Step, context: Context) -> bool:
        """Needs something to check claims *against*. With an empty evidence pool
        every claim comes back unverifiable, so the layer can only return UNSURE —
        it cannot inform the decision, and running it just burns budget and makes
        the step look doubtful. On step 1 the pool is always empty: the agent has
        not observed anything yet."""
        context = context or Context()
        if step.prev_observation.strip():
            return True
        return bool(context.scratch.get(self.trajectory_key))

    def run(self, step: Step, context: Context) -> Signal:
        context = context or Context()
        t0 = time.perf_counter()

        claims = self._collect_claims(step, context)
        pool = self._build_pool(step, context)

        triples: list[dict] = []
        results: list[_ClaimResult] = []
        n_calls, tokens = 0, 0
        nli_fn: NLIFn | None = None

        for claim in claims[:self.max_claims]:
            evidences = self.retrieve_fn(claim, pool, self.k)
            if not evidences:
                results.append(_ClaimResult(claim, ClaimStatus.NO_EVIDENCE, 0.0))
                triples.append({"claim": claim, "evidence": None,
                                "source": None, "label": None, "score": None})
                continue

            if nli_fn is None:
                nli_fn = self._resolve_nli()

            best_contra, best_entail = 0.0, 0.0
            contra_src = entail_src = None
            for ev in evidences:
                premise, hypothesis = self._truncate_pair(ev.text, claim)
                out = nli_fn(premise, hypothesis)
                n_calls += 1
                tokens += _est_tokens(premise) + _est_tokens(hypothesis)
                label = _to_label(out.get("label"))
                score = float(out.get("score", 0.0))
                triples.append({"claim": claim, "evidence": ev.text,
                                "source": ev.source, "label": label.value,
                                "score": score})
                if label is NLILabel.CONTRADICTION and score > best_contra:
                    best_contra, contra_src = score, ev.source
                elif label is NLILabel.ENTAILMENT and score > best_entail:
                    best_entail, entail_src = score, ev.source

            if best_contra >= self.contradiction_threshold:
                results.append(_ClaimResult(claim, ClaimStatus.CONTRADICTED,
                                            best_contra, contra_src))
            elif best_entail >= self.entailment_threshold:
                results.append(_ClaimResult(claim, ClaimStatus.SUPPORTED,
                                            best_entail, entail_src))
            else:
                results.append(_ClaimResult(claim, ClaimStatus.NEUTRAL,
                                            max(best_entail, best_contra)))

        latency_ms = ((time.perf_counter() - t0) * 1000.0
                      + n_calls * self.latency_per_call_ms)
        cost = Cost(tokens=tokens, latency_ms=latency_ms)

        return self._aggregate(results, triples, n_calls, cost)

    def _aggregate(self, results: list[_ClaimResult], triples: list[dict],
                   n_calls: int, cost: Cost) -> Signal:
        n_no_ev = sum(1 for r in results if r.status is ClaimStatus.NO_EVIDENCE)
        contradicted = [r for r in results if r.status is ClaimStatus.CONTRADICTED]
        supported = [r for r in results if r.status is ClaimStatus.SUPPORTED]

        raw: dict = {
            "triples": triples,
            "n_nli_calls": n_calls,
            "n_claims": len(results),
            "n_no_evidence": n_no_ev,
            "claim_status": [(r.claim, r.status.value) for r in results],
        }

        if contradicted:
            worst = max(contradicted, key=lambda r: r.strength)
            confidence = max(0.0, 1.0 - worst.strength)
            raw["reason"] = "contradiction"
            raw["feedback"] = (f"Claim refuted by evidence: “{worst.claim}”. "
                               f"Rely on the Observation, do not fabricate.")
            return Signal(Verdict.FAIL, confidence, ErrorType.FACTUAL, cost, raw)

        if not supported:
            raw["reason"] = ("no_claims" if not results
                             else "no_evidence" if n_no_ev == len(results)
                             else "all_neutral")
            raw["feedback"] = None
            return Signal(Verdict.UNSURE, self.unsure_confidence,
                          ErrorType.NONE, cost, raw)

        if len(supported) == len(results):
            confidence = min(r.strength for r in supported)
            raw["reason"] = "all_supported"
            raw["feedback"] = None
            return Signal(Verdict.PASS, confidence, ErrorType.NONE, cost, raw)

        raw["reason"] = "partial_support"
        raw["feedback"] = None
        return Signal(Verdict.UNSURE, self.unsure_confidence,
                      ErrorType.NONE, cost, raw)


def _show(title: str, sig: Signal) -> None:
    print(f"----- {title} -----")
    print(f"  verdict={sig.verdict.value:<6} confidence={sig.confidence:.3f}  "
          f"suspect={sig.suspected_error_type.value}  "
          f"nli_calls={sig.raw['n_nli_calls']}  tokens={sig.cost.tokens}  "
          f"reason={sig.raw.get('reason')}")
    for t in sig.raw["triples"]:
        ev = (t["evidence"][:48] + "…") if t["evidence"] and len(t["evidence"]) > 48 \
            else t["evidence"]
        print(f"    claim=“{t['claim'][:52]}”")
        print(f"      vs evidence={ev!r}  ->  {t['label']}"
              + (f" ({t['score']:.2f})" if t["score"] is not None else ""))
    if sig.raw.get("feedback"):
        print(f"  feedback: {sig.raw['feedback']}")
    print()


def _demo() -> None:
    step_a = Step(
        action="Finish[8,849 metres]",
        thought="Mount Everest has a peak at 8,849 metres above sea level.",
        goal="Height of Everest's peak?",
        step_index=3,
    )
    ctx_a = Context(scratch={"observations": [
        "Mount Everest is Earth's highest mountain above sea level, located in the "
        "Himalayas. Its peak is 8,849 metres above sea level.",
        "The mountain was first summited in 1953 by Tenzing Norgay and Edmund Hillary.",
    ]})
    nli_a = make_mock_nli(rules=[
        ("everest", "8,849", "entailment", 0.96),
        ("everest", "everest", "entailment", 0.88),
    ])
    _show("(a) claim supported → PASS",
          Layer3RetrievalVerifier(nli_a).run(step_a, ctx_a))

    step_b = Step(
        action="Finish[984 metres]",
        thought="The Eiffel Tower is 984 metres tall.",
        goal="Height of the Eiffel Tower?",
        step_index=2,
    )
    ctx_b = Context(scratch={"observations": [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "The Eiffel Tower is 330 metres tall, about the height of an 81-storey building.",
    ]})
    nli_b = make_mock_nli(rules=[
        ("eiffel", "eiffel", "contradiction", 0.93),
    ])
    _show("(b) claim contradicted → FAIL / FACTUAL",
          Layer3RetrievalVerifier(nli_b).run(step_b, ctx_b))

    step_c = Step(
        action="Finish[14 million]",
        thought="The population of Tokyo is 14 million.",
        goal="Population of Tokyo?",
        step_index=1,
    )
    ctx_c = Context(scratch={"observations": [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.",
    ]})
    nli_c = make_mock_nli(rules=[("tokyo", "tokyo", "entailment", 0.9)])
    _show("(c) no evidence → UNSURE (low conf)",
          Layer3RetrievalVerifier(nli_c).run(step_c, ctx_c))

    from src.verification.orchestrator import (
        FixedThresholdRouter, MockLayer, Orchestrator,
    )
    layers = {
        1: MockLayer(1, Signal(Verdict.PASS, 0.6, ErrorType.NONE, Cost(0, 1))),
        2: MockLayer(2, Signal(Verdict.UNSURE, 0.5, ErrorType.FACTUAL, Cost(120, 20))),
        3: Layer3RetrievalVerifier(nli_b),
        4: MockLayer(4, Signal(Verdict.PASS, 0.9, ErrorType.NONE, Cost(800, 200))),
    }
    res = Orchestrator(layers, FixedThresholdRouter(0.3, 0.8)).verify(step_b, ctx_b)
    print("----- (d) end-to-end via Orchestrator (L3 FACTUAL specialist) -----")
    for rec in res.signal_history:
        s = rec.signal
        print(f"  L{rec.layer_id}: {s.verdict.value:<6} conf={s.confidence:.2f} "
              f"suspect={s.suspected_error_type.value}")
    print(f"  -> decision={res.decision.value}  layers_run={res.layers_run}  "
          f"trigger={res.triggering_error_type.value if res.triggering_error_type else '-'}  "
          f"fused(src=L{res.fused.source_layer_id}, ps={res.fused.pass_score:.2f})  "
          f"cost={res.total_cost.tokens} tok")


if __name__ == "__main__":
    _demo()
