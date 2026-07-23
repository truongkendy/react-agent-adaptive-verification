import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.layer1_rule import (
    Layer1Adapter, RuleBasedVerifier, ViolationType, MAX_ARG_LEN,
)
from src.verification.orchestrator import Context, ErrorType, Step, Verdict


def _last_type(v: RuleBasedVerifier) -> ViolationType:
    return v.violations[-1].type


def test_rules() -> None:
    v = RuleBasedVerifier()
    ok, err = v.check("Search[Barack Obama]")
    assert ok and err is None, err

    v = RuleBasedVerifier()
    ok, _ = v.check("Search Barack Obama")
    assert not ok and _last_type(v) is ViolationType.MALFORMED

    v = RuleBasedVerifier()
    ok, _ = v.check("Browse[x]")
    assert not ok and _last_type(v) is ViolationType.UNKNOWN_TOOL

    v = RuleBasedVerifier()
    ok, _ = v.check("Search[]")
    assert not ok and _last_type(v) is ViolationType.EMPTY_ARG

    v = RuleBasedVerifier()
    ok, _ = v.check("Search[" + "a" * (MAX_ARG_LEN + 1) + "]")
    assert not ok and _last_type(v) is ViolationType.ARG_TOO_LONG

    v = RuleBasedVerifier()
    ok, _ = v.check("Search[http://example.com]")
    assert not ok and _last_type(v) is ViolationType.POLICY

    # Duplicate detection is about actions that actually *ran*, so the first one
    # has to be committed before the second counts as a repeat.
    v = RuleBasedVerifier()
    assert v.check("Search[Obama]")[0]
    v.commit("Search[Obama]")
    ok, _ = v.check("Search[Obama]")
    assert not ok and _last_type(v) is ViolationType.DUPLICATE

    v = RuleBasedVerifier()
    assert v.check("finish[yes]")[0]

    print("  [A] RuleBasedVerifier: form rules OK")


def test_precondition() -> None:
    v = RuleBasedVerifier()
    ok, _ = v.check("Lookup[eastern sector]")
    assert not ok and _last_type(v) is ViolationType.PRECONDITION

    v = RuleBasedVerifier()
    assert v.check("Search[Colorado orogeny]")[0]
    v.commit("Search[Colorado orogeny]")
    ok, err = v.check("Lookup[eastern sector]")
    assert ok, err
    print("  [A3] Rule precondition (Lookup-after-Search) OK")


def test_normalized_duplicate() -> None:
    v = RuleBasedVerifier()
    assert v.check("Search[Barack Obama]")[0]
    v.commit("Search[Barack Obama]")
    ok, _ = v.check("search[  barack   obama ]")
    assert not ok and _last_type(v) is ViolationType.DUPLICATE
    assert v.check("Search[Michelle Obama]")[0]
    print("  [A4] Normalized duplicate (case, whitespace) OK")


def test_check_is_pure() -> None:
    """A step can pass Layer 1 and still be revised away by a later layer, in
    which case it never ran. `check()` must not remember it, or the retry is
    rejected as a duplicate of something that never happened — that is how a
    correct `Finish[1755]` got refused in the n=15 Ollama run."""
    v = RuleBasedVerifier()
    for _ in range(3):
        ok, err = v.check("Finish[1755]")
        assert ok, f"uncommitted action must stay retryable, got {err!r}"
    assert v._history == [], v._history

    # Only after it actually ran does a repeat become a duplicate.
    v.commit("Finish[1755]")
    ok, _ = v.check("Finish[1755]")
    assert not ok and _last_type(v) is ViolationType.DUPLICATE

    # Same for the Lookup precondition: a Search that was revised away leaves no
    # page open, so Lookup must still be refused.
    v2 = RuleBasedVerifier()
    assert v2.check("Search[Colorado orogeny]")[0]        # checked, not committed
    ok, _ = v2.check("Lookup[eastern sector]")
    assert not ok and _last_type(v2) is ViolationType.PRECONDITION
    print("  [A5] check() is pure; only commit() records executed actions OK")


def test_reset() -> None:
    v = RuleBasedVerifier()
    v.check("Search[Obama]")
    v.commit("Search[Obama]")
    v.check("Search[Obama]")
    assert v.violation_count == 1
    v.reset()
    assert v.violation_count == 0
    ok, err = v.check("Search[Obama]")
    assert ok, err
    print("  [A2] reset() isolates per-episode state OK")


def test_adapter_signal() -> None:
    layer = Layer1Adapter(pass_confidence=0.9, fail_confidence=0.03)

    sig = layer.run(Step(action="Search[Obama]"), Context())
    assert sig.verdict is Verdict.PASS
    assert sig.suspected_error_type is ErrorType.NONE
    assert sig.confidence == 0.9
    assert sig.cost.tokens == 0
    assert sig.raw["ok"] is True

    sig = layer.run(Step(action="Browse[x]"), Context())
    assert sig.verdict is Verdict.FAIL
    assert sig.suspected_error_type is ErrorType.FORMAT
    assert sig.confidence == 0.03
    assert sig.raw["ok"] is False and sig.raw["error"]

    layer.reset()
    assert layer.run(Step(action="Search[Obama]"), Context()).verdict is Verdict.PASS
    print("  [B] Layer1Adapter -> Signal mapping OK")


if __name__ == "__main__":
    test_rules()
    test_precondition()
    test_normalized_duplicate()
    test_check_is_pure()
    test_reset()
    test_adapter_signal()
    print("ALL CHECKS PASSED ✓")
