from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.verification.base import BaseVerifier
from src.verification.orchestrator import (
    Context, Cost, ErrorType, Layer, Signal, Step, Verdict,
)


class ViolationType(str, Enum):
    MALFORMED    = "malformed"
    UNKNOWN_TOOL = "unknown_tool"
    EMPTY_ARG    = "empty_arg"
    ARG_TOO_LONG = "arg_too_long"
    POLICY       = "policy"
    PRECONDITION = "precondition"
    DUPLICATE    = "duplicate_action"


@dataclass
class Violation:
    type: ViolationType
    message: str


_VALID_TOOLS = {"search", "lookup", "finish"}
_URL_PAT     = re.compile(r"https?://", re.IGNORECASE)
MAX_ARG_LEN  = 200


class RuleBasedVerifier(BaseVerifier):

    def __init__(self, max_arg_len: int = MAX_ARG_LEN):
        self.max_arg_len = max_arg_len
        self._history: list[str] = []
        self.violations: list[Violation] = []

    def reset(self) -> None:
        self._history.clear()
        self.violations.clear()

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def check(self, action: str) -> tuple[bool, str | None]:
        """Pure predicate — does NOT record the action. `commit()` does that.

        `check()` used to append on pass, so an action that passed Layer 1 but was
        then revised away by a later layer still entered the history and its retry
        was rejected as a duplicate. That locked the agent out of actions it had
        never actually run, including a correct `Finish[...]`: in the n=15 Ollama
        run, `Finish[1755]` (the gold answer) was revised at step 4 and then
        refused at step 5 as a duplicate, ending the trajectory with no answer."""
        v = self._run_rules(action)
        if v:
            self.violations.append(v)
            return False, v.message
        return True, None

    def commit(self, action: str) -> None:
        """Record an action that actually ran, for duplicate detection and the
        Lookup-after-Search precondition."""
        self._history.append(action)


    def _run_rules(self, action: str) -> Violation | None:
        m = re.match(r"^(\w+)\[([^\]]*)\]$", action, re.IGNORECASE)
        if not m:
            return Violation(ViolationType.MALFORMED,
                             f"Action is not in the form Tool[argument]: '{action}'. "
                             f"Must be Search[...], Lookup[...] or Finish[...].")

        raw_tool, arg = m.group(1), m.group(2).strip()
        tool = raw_tool.lower()

        if tool not in _VALID_TOOLS:
            return Violation(ViolationType.UNKNOWN_TOOL,
                             f"Tool '{raw_tool}' does not exist. "
                             f"Only use: Search, Lookup, Finish.")

        if not arg:
            return Violation(ViolationType.EMPTY_ARG,
                             f"{raw_tool}[] requires a non-empty argument.")

        if len(arg) > self.max_arg_len:
            return Violation(ViolationType.ARG_TOO_LONG,
                             f"Argument too long ({len(arg)} chars, max {self.max_arg_len}). "
                             f"Use a shorter search.")

        if _URL_PAT.search(arg):
            return Violation(ViolationType.POLICY,
                             f"Argument contains a URL — use the entity name instead of a URL.")

        if tool == "lookup" and not any(
                h.strip().lower().startswith("search[") for h in self._history):
            return Violation(ViolationType.PRECONDITION,
                             "Lookup must be used AFTER a Search (no page is open yet). "
                             "Search first.")

        canon = self._canon(action)
        if canon in {self._canon(h) for h in self._history}:
            return Violation(ViolationType.DUPLICATE,
                             f"'{action}' duplicates an action already run (ignoring "
                             f"case and whitespace differences). Try a different search direction.")

        return None

    @staticmethod
    def _canon(action: str) -> str:
        m = re.match(r"^(\w+)\[([^\]]*)\]$", action.strip(), re.IGNORECASE)
        if not m:
            return re.sub(r"\s+", " ", action.strip().lower())
        tool = m.group(1).lower()
        arg = re.sub(r"\s+", " ", m.group(2).strip().lower())
        return f"{tool}[{arg}]"


class Layer1Adapter(Layer):

    def __init__(self, verifier: RuleBasedVerifier | None = None,
                 pass_confidence: float = 0.9, fail_confidence: float = 0.03):
        super().__init__(layer_id=1)
        self._verifier = verifier or RuleBasedVerifier()
        self.pass_confidence = pass_confidence
        self.fail_confidence = fail_confidence

    def reset(self) -> None:
        self._verifier.reset()

    def commit(self, step: Step) -> None:
        self._verifier.commit(step.action)

    def run(self, step: Step, context: Context) -> Signal:
        t0 = time.perf_counter()
        ok, err = self._verifier.check(step.action)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if ok:
            return Signal(Verdict.PASS, self.pass_confidence, ErrorType.NONE,
                          Cost(tokens=0, latency_ms=latency_ms), raw={"ok": True})
        return Signal(Verdict.FAIL, self.fail_confidence, ErrorType.FORMAT,
                      Cost(tokens=0, latency_ms=latency_ms),
                      raw={"ok": False, "error": err})


def _demo() -> None:
    layer = Layer1Adapter(pass_confidence=0.9)
    for action in ["Search[High Plains (United States)]",
                   "Searchhh High Plains",
                   "Browse[http://x.com]"]:
        sig = layer.run(Step(action=action), Context())
        print(f"  {action!r:42} -> {sig.verdict.value:<4} "
              f"conf={sig.confidence:.2f} suspect={sig.suspected_error_type.value} "
              f"| {sig.raw.get('error') or 'ok'}")


if __name__ == "__main__":
    _demo()
