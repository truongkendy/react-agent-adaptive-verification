"""Rate-limit handling for OpenAICompatLLM, against a local stub server.

Both behaviours here are regressions from a real failed run: a Groq 429 carrying
`Retry-After: 2089` was honored verbatim, so the experiment slept ~35 minutes per
attempt and looked like a hang.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import OpenAICompatLLM

# Mutable stub script: how many 429s to send, and what Retry-After to attach.
STUB = {"n_429": 0, "retry_after": None, "requests": 0}


class Handler(BaseHTTPRequestHandler):

    def do_POST(self):
        STUB["requests"] += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if STUB["n_429"] > 0:
            STUB["n_429"] -= 1
            self.send_response(429)
            if STUB["retry_after"] is not None:
                self.send_header("Retry-After", str(STUB["retry_after"]))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"rate limit"}}')
            return
        body = json.dumps({
            "choices": [{"message": {"content": " ok\nAction 1: Finish[x]"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                      "total_tokens": 1020},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        return


def start_server() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _llm(url: str, **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(base_url=url, model="stub", api_key="k", **kw)


def test_huge_retry_after_fails_fast(url: str) -> None:
    """The real failure: Retry-After 2089s used to be slept verbatim."""
    STUB.update(n_429=1, retry_after=2089, requests=0)
    llm = _llm(url, retry_after_cap=120.0)
    t0 = time.time()
    try:
        llm.generate("sys", "Question: q", "Thought 1:", [])
        assert False, "expected an HTTPError instead of a 35-minute sleep"
    except Exception as e:
        elapsed = time.time() - t0
        assert "quota wall" in str(e), str(e)[:200]
        assert elapsed < 5.0, f"failed but took {elapsed:.1f}s — did it sleep?"
    print(f"  [A] Retry-After 2089s -> raises in {elapsed:.2f}s, no sleep OK")


def test_small_retry_after_is_still_honored(url: str) -> None:
    """A transient 429 must still be retried, not turned into a hard failure."""
    STUB.update(n_429=1, retry_after=1, requests=0)
    llm = _llm(url, retry_after_cap=120.0)
    t0 = time.time()
    out = llm.generate("sys", "Question: q", "Thought 1:", [])
    elapsed = time.time() - t0
    assert "Finish[x]" in out, out
    assert STUB["requests"] == 2, STUB["requests"]
    assert 0.9 < elapsed < 4.0, elapsed
    print(f"  [B] Retry-After 1s -> retried and succeeded in {elapsed:.2f}s OK")


def test_no_retry_after_uses_capped_backoff(url: str) -> None:
    STUB.update(n_429=1, retry_after=None, requests=0)
    llm = _llm(url, backoff_base=2.0, backoff_cap=1.0)
    t0 = time.time()
    out = llm.generate("sys", "Question: q", "Thought 1:", [])
    assert "Finish[x]" in out
    assert time.time() - t0 < 3.0
    print("  [C] no Retry-After -> own capped backoff, unchanged OK")


def test_tpm_pacing_throttles(url: str) -> None:
    """With a tiny TPM budget the second call must wait rather than 429."""
    STUB.update(n_429=0, retry_after=None, requests=0)
    # Budget 1500*0.85 = 1275. The stub reports 1020 total tokens per reply and
    # the pre-request estimate is ~519 (short prompt + max_tokens allowance), so
    # the first call fits and the second (1020 + 519 > 1275) must be paced.
    llm = _llm(url, tpm_limit=1500, tpm_headroom=0.85)
    llm.generate("sys", "Question: q", "Thought 1:", [])
    assert llm.throttled_s == 0.0, "first call should not wait"
    assert len(llm._token_log) == 1 and llm._token_log[0][1] == 1020

    # Age the recorded entry so the pacer's wait is short but non-zero.
    ts, tok = llm._token_log[0]
    llm._token_log[0] = (ts - 58.5, tok)
    t0 = time.time()
    llm.generate("sys", "Question: q", "Thought 1:", [])
    elapsed = time.time() - t0
    assert llm.throttled_s > 0.0, "second call should have paced"
    assert elapsed >= 0.5, elapsed
    assert STUB["requests"] == 2, "pacing must not cost extra requests"
    print(f"  [D] TPM pacing waited {llm.throttled_s:.1f}s before the 2nd call, "
          f"no 429 needed OK")


def test_no_tpm_limit_means_no_pacing(url: str) -> None:
    STUB.update(n_429=0, retry_after=None, requests=0)
    llm = _llm(url)                      # tpm_limit defaults to None
    for _ in range(3):
        llm.generate("sys", "Question: q", "Thought 1:", [])
    assert llm.throttled_s == 0.0
    assert llm._token_log == [], "should not track tokens when pacing is off"
    assert llm.prompt_tokens == 3000 and llm.completion_tokens == 60
    print("  [E] tpm_limit=None -> no pacing, no bookkeeping, counters intact OK")


if __name__ == "__main__":
    srv, url = start_server()
    try:
        test_huge_retry_after_fails_fast(url)
        test_small_retry_after_is_still_honored(url)
        test_no_retry_after_uses_capped_backoff(url)
        test_tpm_pacing_throttles(url)
        test_no_tpm_limit_means_no_pacing(url)
    finally:
        srv.shutdown()
    print("ALL CHECKS PASSED ✓")
