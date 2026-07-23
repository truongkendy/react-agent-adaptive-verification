"""Checks for `src.verification.json_extract`.

The multi-object case is the regression this module exists for: the greedy
`\\{.*\\}` it replaced spanned the first `{` to the last `}` and never parsed, so
Layer 4 abstained on 5/5 replies in the first n=5 Ollama run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verification.json_extract import first_json_object, iter_json_objects
from src.verification.layer2_rubric import RUBRIC, parse_scores
from src.verification.layer4_llm_judge import parse_judgement


def check_multi_object() -> None:
    """The reply that broke the greedy regex, verbatim in shape."""
    reply = (
        'rationale = why you think so.  If the Thought is a non-sequitur, this\n'
        'should explain what was expected instead.\n\n'
        '{\n  "reasoning_sound": false,\n  "confidence": 0.8,\n'
        '  "rationale": "The thought does not follow from the observation."\n}'
        ' {"reasoning_sound": true, "confidence": 1.0, "rationale": "ok"}'
        ' {"reasoning_sound": false, "confidence": 0.4, "rationale": "no"}'
    )
    objs = list(iter_json_objects(reply))
    assert len(objs) == 3, objs
    # First object wins — it is the one the model committed to before rambling.
    assert objs[0]["confidence"] == 0.8

    parsed = parse_judgement(reply)
    assert parsed is not None, "multi-object reply must still parse"
    assert parsed["reasoning_sound"] is False
    assert parsed["confidence"] == 0.8
    # reasoning_sound=false at confidence 0.8 -> P(sound) = 0.2, a real FAIL
    # signal. Under the old parser this became UNSURE at 0.5 and was discarded
    # by fuse() as non-decisive.
    print("  [A] multi-object reply parses, keeps the first object OK")


def check_braces_in_strings() -> None:
    obj = first_json_object('{"rationale": "unbalanced } brace", "reasoning_sound": true}',
                            ("reasoning_sound",))
    assert obj is not None and obj["reasoning_sound"] is True, obj
    print("  [B] a } inside a string does not terminate the object OK")


def check_nesting() -> None:
    objs = list(iter_json_objects('{"a": {"b": {"c": 1}}} {"d": 2}'))
    assert objs == [{"a": {"b": {"c": 1}}}, {"d": 2}], objs
    print("  [C] nested objects are spanned, not split OK")


def check_required_keys_select() -> None:
    """The object the caller wants is not always the first one printed."""
    reply = '{"note": "thinking aloud"} {"reasoning_sound": true, "confidence": 0.9}'
    assert first_json_object(reply) == {"note": "thinking aloud"}
    picked = first_json_object(reply, ("reasoning_sound",))
    assert picked is not None and picked["reasoning_sound"] is True, picked
    print("  [D] required_keys selects the right object OK")


def check_negatives() -> None:
    assert first_json_object("no json here") is None
    assert first_json_object("") is None
    assert first_json_object('{"unterminated": 1') is None
    assert parse_judgement('{"confidence": 0.5}') is None, "missing key must fail"
    assert list(iter_json_objects('{"trailing": 1,}')) == [], "invalid JSON yields nothing"
    print("  [E] malformed / missing-key input still returns None OK")


def check_layer2_multi_object() -> None:
    # Each criterion has its own range, so score each at its own max.
    names = [c.name for c in RUBRIC]
    tops = {c.name: c.max_score for c in RUBRIC}
    good = "{" + ", ".join(f'"{n}": {tops[n]}' for n in names) + "}"
    reply = f'Here are the scores.\n{good} {{"note": "done"}}'
    scores = parse_scores(reply, RUBRIC)
    assert scores is not None, "Layer 2 must parse a multi-object reply"
    assert all(scores[n] == tops[n] for n in names), scores
    # A partial object must not be mistaken for the answer.
    partial = '{"' + names[0] + '": 1} ' + good
    picked = parse_scores(partial, RUBRIC)
    assert picked is not None and picked == scores, picked
    print("  [F] Layer 2 rubric parse survives extra objects OK")


def check_fenced() -> None:
    parsed = parse_judgement('```json\n{"reasoning_sound": "yes", "confidence": 1.7}\n```')
    assert parsed is not None and parsed["reasoning_sound"] is True
    assert parsed["confidence"] == 1.0, "confidence must stay clamped to [0,1]"
    print("  [G] fenced block + string bool + clamping OK")


if __name__ == "__main__":
    print("json_extract checks")
    check_multi_object()
    check_braces_in_strings()
    check_nesting()
    check_required_keys_select()
    check_negatives()
    check_layer2_multi_object()
    check_fenced()
    print("ALL CHECKS PASSED ✓")
