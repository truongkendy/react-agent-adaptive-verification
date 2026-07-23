from __future__ import annotations

import re
from typing import Protocol


class WikiBackend(Protocol):
    def fetch_intro(self, title: str) -> str | None:
        ...

    def search(self, query: str, limit: int = 5) -> list[str]:
        ...


class MediaWikiBackend:

    API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, lang: str = "en", timeout: int = 20,
                 max_retries: int = 5, backoff_base: float = 2.0,
                 backoff_cap: float = 60.0):
        import requests
        self._requests = requests
        self.API = f"https://{lang}.wikipedia.org/w/api.php"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ReAct-thesis/0.1 (research; contact: truong.nguyen@tyme.com)"})

    def _get(self, params: dict):
        import time
        for attempt in range(self.max_retries + 1):
            r = self.session.get(self.API, params=params, timeout=self.timeout)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r
            if attempt == self.max_retries:
                r.raise_for_status()
                return r
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else min(self.backoff_base ** attempt,
                                                 self.backoff_cap)
            except ValueError:
                wait = min(self.backoff_base ** attempt, self.backoff_cap)
            print(f"  [wiki retry] HTTP {r.status_code}, waiting {wait:.1f}s "
                  f"(attempt {attempt + 1}/{self.max_retries})...")
            time.sleep(wait)
        return r

    def fetch_intro(self, title: str) -> str | None:
        params = {
            "action": "query", "format": "json", "prop": "extracts",
            "exintro": 1, "explaintext": 1, "redirects": 1, "titles": title,
        }
        r = self._get(params)
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return page.get("extract", "") or None

    def search(self, query: str, limit: int = 5) -> list[str]:
        params = {
            "action": "query", "format": "json", "list": "search",
            "srsearch": query, "srlimit": limit,
        }
        r = self._get(params)
        return [hit["title"] for hit in r.json()["query"]["search"]]


class MockWikiBackend:

    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def fetch_intro(self, title: str) -> str | None:
        return self.pages.get(title)

    def search(self, query: str, limit: int = 5) -> list[str]:
        q = query.lower()
        hits = [t for t in self.pages if q in t.lower()]
        return hits[:limit] if hits else list(self.pages)[:limit]


class DistractorBackend(MockWikiBackend):
    """Serves the paragraphs HotpotQA's *distractor* setting supplies with each
    question — the two gold ones plus eight distractors — instead of the live
    encyclopedia.

    This exists to remove a confound rather than to make the task easier. In the
    `fullwiki` setting the agent must find the pages itself, and on the
    `n30_ollama` runs **43% of questions (13/30, the same 13 for baseline and
    adaptive) never retrieved the gold answer into any observation at all**. No
    amount of *step* verification can recover those: a layer judges a step
    before its observation exists, so it cannot know a query will come back
    useless. Verification's measurable ceiling was 17/30 with 13 questions
    excluded by the benchmark's own design. Under `distractor` the evidence is
    always present, so every remaining failure is a reasoning or
    evidence-usage failure — the cascade's actual target domain.

    Titles are matched case-insensitively and the page set is swapped per
    question via `load()`, so one instance can back a whole run.
    """

    def __init__(self) -> None:
        super().__init__({})
        self._lower: dict[str, str] = {}

    def load(self, context: list) -> None:
        """Install one question's paragraphs. `context` is HotpotQA's
        `[[title, [sentence, ...]], ...]`; a plain `{title: text}` also works."""
        pages: dict[str, str] = {}
        if isinstance(context, dict):
            pages = {str(t): str(v) for t, v in context.items()}
        else:
            for entry in context or ():
                title, sentences = entry[0], entry[1]
                text = ("".join(sentences) if isinstance(sentences, (list, tuple))
                        else str(sentences))
                pages[str(title)] = text.strip()
        self.pages = pages
        self._lower = {t.lower(): t for t in pages}

    def fetch_intro(self, title: str) -> str | None:
        """Exact title first, then case-insensitively. The agent copies titles
        out of a `Similar: [...]` list, so an exact-only match would reject a
        page that is demonstrably present and read as a retrieval failure."""
        if title in self.pages:
            return self.pages[title]
        return self.pages.get(self._lower.get(title.strip().lower(), ""))


def _to_sentences(page: str) -> list[str]:
    paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
    sentences: list[str] = []
    for p in paragraphs:
        sentences += p.split(". ")
    out: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not s.endswith((".", "!", "?", ":")):
            s += "."
        out.append(s)
    return out


def _page_obs(page: str, n: int = 5) -> str:
    return " ".join(_to_sentences(page)[:n])


class WikiEnv:
    def __init__(self, backend: WikiBackend):
        self.backend = backend
        self.reset()

    def reset(self) -> None:
        self.page: str | None = None
        self.lookup_keyword: str | None = None
        self.lookup_list: list[str] = []
        self.lookup_cnt: int = 0
        self.answer: str | None = None
        self.steps: int = 0

    def step(self, action: str) -> tuple[str, bool]:
        self.steps += 1
        action = action.split("\n")[0].strip()
        m = re.match(r"^(search|lookup|finish)\[([^\]]*)\]", action, flags=re.IGNORECASE)
        if not m:
            return ("Invalid action. Use Search[...], Lookup[...] or Finish[...].",
                    False)
        verb, arg = m.group(1).lower(), m.group(2)

        if verb == "search":
            return (self._search(arg), False)
        if verb == "lookup":
            return (self._lookup(arg), False)
        self.answer = arg
        return (f"Episode finished. Answer: {arg}", True)

    def _search(self, entity: str, _depth: int = 0) -> str:
        intro = self.backend.fetch_intro(entity)

        if intro is None:
            sims = self.backend.search(entity, limit=5)
            return f"Could not find {entity}. Similar: {sims}."

        if "may refer to:" in intro and _depth == 0:
            return self._search(f"[{entity}]", _depth=1)

        self.page = intro
        self.lookup_keyword = None
        self.lookup_list = []
        self.lookup_cnt = 0
        return _page_obs(intro)

    def _lookup(self, keyword: str) -> str:
        if self.page is None:
            return "No page to look up. Use Search first."
        if keyword != self.lookup_keyword:
            self.lookup_keyword = keyword
            self.lookup_list = [s for s in _to_sentences(self.page)
                                if keyword.lower() in s.lower()]
            self.lookup_cnt = 0
        if self.lookup_cnt >= len(self.lookup_list):
            return f"No more results for '{keyword}'."
        i, n = self.lookup_cnt, len(self.lookup_list)
        self.lookup_cnt += 1
        return f"(Result {i + 1} / {n}) {self.lookup_list[i]}"
