"""Checks for `DistractorBackend` and the `--distractor` data guard.

The guard is the point of the file. A fullwiki dev record *also* carries a
`context` field — paragraphs an IR system retrieved — and only ~28% of those hold
both gold titles, so "has context" cannot be the test for "is the distractor
set". Getting that wrong serves evidence that usually cannot answer the question
and reads as a catastrophic agent failure rather than a mis-set flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_baseline import check_distractor_data
from src.tools.wikipedia import DistractorBackend, WikiEnv


class _Args:
    def __init__(self, distractor=True, data="x.json"):
        self.distractor = distractor
        self.data = data


CTX = [
    ("Scott Derrickson", ["Scott Derrickson is an American director.",
                          " He lives in Los Angeles."]),
    ("Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker."]),
    ("Tyler Bates", ["Tyler Bates is an American musician."]),
]


def check_load_and_serve() -> None:
    b = DistractorBackend()
    assert b.fetch_intro("Ed Wood") is None, "empty before load()"
    b.load(CTX)
    assert len(b.pages) == 3
    intro = b.fetch_intro("Scott Derrickson")
    assert intro and "American director" in intro
    # Sentences are joined into one page body, so Lookup can scan them.
    assert "Los Angeles" in intro
    print("  [A] load() installs the paragraph set OK")


def check_case_insensitive() -> None:
    """The agent copies titles out of a `Similar: [...]` list and often lowercases
    them; an exact-only match would report a page that is present as missing."""
    b = DistractorBackend()
    b.load(CTX)
    assert b.fetch_intro("ed wood") == b.fetch_intro("Ed Wood")
    assert b.fetch_intro("  ED WOOD  ") == b.fetch_intro("Ed Wood")
    assert b.fetch_intro("Ed Woods") is None, "must not fuzzy-match a real miss"
    print("  [B] title match is case/whitespace insensitive, not fuzzy OK")


def check_swap_between_questions() -> None:
    b = DistractorBackend()
    b.load(CTX)
    b.load([("Other Page", ["Unrelated text."])])
    assert b.fetch_intro("Ed Wood") is None, "previous question must not leak"
    assert b.fetch_intro("Other Page") is not None
    b.load([])
    assert b.pages == {} and b.fetch_intro("Other Page") is None
    print("  [C] load() replaces rather than accumulates OK")


def check_env_integration() -> None:
    env = WikiEnv(DistractorBackend())
    env.backend.load(CTX)
    obs, done = env.step("Search[Ed Wood]")
    assert not done and "filmmaker" in obs, obs
    obs, _ = env.step("Search[Nowhere]")
    assert obs.startswith("Could not find"), obs
    # A miss lists the titles that *are* available, which is how the agent
    # recovers; with only 3 pages loaded it must not invent more.
    assert "Ed Wood" in obs
    obs, _ = env.step("Search[Scott Derrickson]")
    obs, _ = env.step("Lookup[Los Angeles]")
    assert "Los Angeles" in obs, obs
    print("  [D] WikiEnv Search/Lookup work against the served pages OK")


def check_guard() -> None:
    dist = [{"supporting_facts": [("A", 0), ("B", 0)],
             "context": [("A", ["x"]), ("B", ["y"]), ("C", ["z"])]}] * 20
    check_distractor_data(_Args(), dist)          # must not exit

    # fullwiki-shaped: context present, gold titles mostly absent.
    full = [{"supporting_facts": [("A", 0), ("B", 0)],
             "context": [("C", ["z"]), ("D", ["w"])]}] * 20
    for args, label in ((_Args(), "fullwiki data"),
                        (_Args(), "no gold in context")):
        try:
            check_distractor_data(args, full)
        except SystemExit as e:
            assert "distractor dev set" in str(e), str(e)
        else:
            raise AssertionError(f"guard must reject {label}")

    # Flag off -> never inspects the data.
    check_distractor_data(_Args(distractor=False), full)
    check_distractor_data(_Args(), [])            # empty: nothing to judge
    print("  [E] guard rejects fullwiki, accepts distractor, ignores flag-off OK")


if __name__ == "__main__":
    print("DistractorBackend checks")
    check_load_and_serve()
    check_case_insensitive()
    check_swap_between_questions()
    check_env_integration()
    check_guard()
    print("ALL CHECKS PASSED ✓")
