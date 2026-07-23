from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    prompt_tokens: int
    completion_tokens: int

    def generate(self, system: str, user: str, prefill: str,
                 stop: list[str]) -> str:
        ...


class AnthropicLLM:

    def __init__(self, model: str = "claude-sonnet-4-6",
                 max_tokens: int = 512, temperature: float = 0.0):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system: str, user: str, prefill: str,
                 stop: list[str]) -> str:
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
            stop_sequences=stop,
        )
        self.prompt_tokens += resp.usage.input_tokens
        self.completion_tokens += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text


class MockLLM:

    def __init__(self, scripts: dict[str, list[tuple[str, str]]]):
        self.scripts = scripts
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system: str, user: str, prefill: str,
                 stop: list[str]) -> str:
        question = user.split("Question:", 1)[-1].strip()
        steps = self.scripts[question]
        i = prefill.count("Observation ")
        thought, action = steps[i]
        out = f" {thought}\nAction {i + 1}: {action}"
        self.prompt_tokens += len(system) // 4 + len(prefill) // 4
        self.completion_tokens += len(out) // 4
        return out


class OpenAICompatLLM:

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 60, extra_headers: dict | None = None,
                 max_retries: int = 5, backoff_base: float = 2.0,
                 backoff_cap: float = 60.0, retry_after_cap: float = 120.0,
                 tpm_limit: int | None = None, tpm_headroom: float = 0.85):
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        # `backoff_cap` only ever bounded our *own* backoff. A server-supplied
        # Retry-After was honored verbatim, so a quota wall (Groq answers 429
        # with Retry-After ~2000s) put the trajectory to sleep for half an hour
        # per attempt — up to max_retries times. In a research loop that is
        # indistinguishable from a hang, so past this cap we fail loudly instead.
        self.retry_after_cap = retry_after_cap
        # Tokens-per-minute pacing. These experiments are throughput-bound by
        # TPM, not by latency: at 12k TPM and ~2.2k tokens per ReAct step a run
        # can only advance ~5 steps/minute. Waiting *before* the request keeps
        # the run under the limit instead of discovering it via 429s.
        self.tpm_limit = tpm_limit
        self.tpm_headroom = tpm_headroom
        self._token_log: list[tuple[float, int]] = []   # (timestamp, tokens)
        self.throttled_s = 0.0                          # total time spent pacing
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system: str, user: str, prefill: str,
                 stop: list[str]) -> str:
        sys_msg = (system + "\n\nYou are continuing a ReAct transcript. Output "
                   "ONLY the next Thought line followed by the next Action line, "
                   "then stop. Do not write any Observation.")
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"{user}\n{prefill}"},
        ]
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stop": stop,
        }
        est = self._estimate_tokens(sys_msg, f"{user}\n{prefill}")
        r = self._post_with_retry(payload, headers, est_tokens=est)
        data = r.json()
        usage = data.get("usage", {}) or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self._record_tokens(usage.get("total_tokens", 0) or est)
        return data["choices"][0]["message"]["content"] or ""

    def _estimate_tokens(self, *texts: str) -> int:
        """Pre-request estimate, needed because pacing has to happen before we
        know the real usage. ~4 chars/token, plus the completion allowance."""
        return sum(len(t) for t in texts) // 4 + self.max_tokens

    def _record_tokens(self, tokens: int) -> None:
        if self.tpm_limit and tokens:
            import time
            self._token_log.append((time.time(), int(tokens)))

    def _throttle(self, est_tokens: int) -> None:
        """Sleep until `est_tokens` fits under the rolling 60s TPM budget."""
        if not self.tpm_limit:
            return
        import time
        budget = self.tpm_limit * self.tpm_headroom
        while True:
            now = time.time()
            self._token_log = [(t, n) for t, n in self._token_log if now - t < 60.0]
            used = sum(n for _, n in self._token_log)
            if used + est_tokens <= budget or not self._token_log:
                return
            # Wait only until the oldest entry leaves the window, then re-check.
            wait = min(60.0, max(0.5, 60.0 - (now - self._token_log[0][0])))
            print(f"  [pace] {used}+{est_tokens} tok > {budget:.0f}/min budget, "
                  f"waiting {wait:.1f}s...")
            self.throttled_s += wait
            time.sleep(wait)

    def _post_with_retry(self, payload: dict, headers: dict,
                         est_tokens: int = 0):
        import time
        url = self.base_url + "/chat/completions"
        for attempt in range(self.max_retries + 1):
            self._throttle(est_tokens)
            r = self._requests.post(url, json=payload, headers=headers,
                                    timeout=self.timeout)
            if r.status_code != 429 and r.status_code < 500:
                self._raise_for_status(r)
                return r
            if attempt == self.max_retries:
                self._raise_for_status(r)
                return r
            wait = self._retry_after(r)
            if wait is None:
                wait = min(self.backoff_base ** attempt, self.backoff_cap)
            elif wait > self.retry_after_cap:
                raise self._requests.HTTPError(
                    f"{r.status_code} quota wall: server asked to wait "
                    f"{wait:.0f}s, above retry_after_cap={self.retry_after_cap:.0f}s. "
                    f"Failing instead of sleeping — the rate limit is not "
                    f"transient. Lower --n, raise --sleep, or set --tpm to pace "
                    f"under the account's tokens-per-minute limit.\n"
                    f"  -> {r.text[:300]}", response=r)
            print(f"  [retry] HTTP {r.status_code}, waiting {wait:.1f}s "
                  f"(attempt {attempt + 1}/{self.max_retries})...")
            time.sleep(wait)
        return r

    def _raise_for_status(self, r) -> None:
        if r.status_code >= 400:
            raise self._requests.HTTPError(
                f"{r.status_code} {r.reason} for {r.url}\n  -> {r.text[:600]}",
                response=r)

    @staticmethod
    def _retry_after(r) -> float | None:
        val = r.headers.get("Retry-After")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None


class OllamaLLM:

    def __init__(self, model: str = "llama3.1",
                 base_url: str = "http://localhost:11434",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 120):
        import requests
        self._requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system: str, user: str, prefill: str,
                 stop: list[str]) -> str:
        full = f"{system}\n\n{user}\n{prefill}"
        payload = {
            "model": self.model, "prompt": full, "raw": True, "stream": False,
            "options": {"stop": stop, "temperature": self.temperature,
                        "num_predict": self.max_tokens},
        }
        r = self._requests.post(self.base_url + "/api/generate",
                                json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self.prompt_tokens += data.get("prompt_eval_count", 0)
        self.completion_tokens += data.get("eval_count", 0)
        return data.get("response", "") or ""
