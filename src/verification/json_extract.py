"""Tolerant JSON-object extraction from an LLM reply.

Why this exists: the LLM layers used to locate their JSON with
`re.search(r"\\{.*\\}", text, re.DOTALL)`. That pattern is *greedy*, so when a
model emits several objects in a row it matches from the first `{` to the last
`}` and hands `json.loads` a concatenation, which never parses. Measured on the
n=5 Ollama run: Layer 4 failed to parse 5/5 replies and Layer 2 8/15, even
though the *first* object in each reply was well-formed and informative. Every
one of those became a `Verdict.UNSURE` at confidence 0.5, which `fuse()`
correctly treats as non-decisive — so the whole cascade abstained on every step
and the run reduced to baseline + Layer 1 at 2.4x the LLM calls.

`iter_json_objects` scans for *balanced* braces instead, tracking string state so
a `}` inside a rationale does not end the object, and yields each candidate
left to right. `first_json_object` takes the first one carrying the keys the
caller actually needs, which is what makes multi-object replies survivable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence

_FENCE_RE = re.compile(r"```(?:json)?|```")


def iter_json_objects(text: str) -> Iterator[dict]:
    """Yield every JSON object in `text`, left to right, outermost first.

    Balanced-brace scan, not a regex: `{"a": {"b": 1}} {"c": 2}` yields two
    objects, and a `}` inside a string literal is ignored. When an outer span
    fails to parse the scan advances by one character rather than past the whole
    span, so an inner object still gets a chance.
    """
    t = _FENCE_RE.sub("", text or "")
    i, n = 0, len(t)
    while i < n:
        if t[i] != "{":
            i += 1
            continue

        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            ch = t[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if j >= n or depth != 0:
            return  # unterminated: nothing later can close it either

        try:
            obj = json.loads(t[i:j + 1])
        except (json.JSONDecodeError, ValueError):
            obj = None

        if isinstance(obj, dict):
            yield obj
            i = j + 1
        else:
            i += 1


def first_json_object(text: str,
                      required_keys: Sequence[str] = ()) -> dict | None:
    """First object in `text` that carries all of `required_keys`, else None.

    Filtering on the keys matters when a model emits several objects: the one it
    was asked for is not always the first one it happens to print.
    """
    req = tuple(required_keys)
    for obj in iter_json_objects(text):
        if all(k in obj for k in req):
            return obj
    return None


def _demo() -> None:
    multi = ('rationale = why you think so.\n'
             '{"reasoning_sound": false, "confidence": 0.8, "rationale": "no"} '
             '{"reasoning_sound": true, "confidence": 1.0, "rationale": "ok"}')
    print("multi-object ->", first_json_object(multi, ("reasoning_sound",)))
    print("greedy would have spanned both and failed to parse.")
    nested = '{"scores": {"a": 1}, "note": "brace } inside a string"}'
    print("nested/string ->", first_json_object(nested, ("scores",)))
    print("no json       ->", first_json_object("nothing here"))


if __name__ == "__main__":
    _demo()
