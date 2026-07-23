import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.llm import OllamaLLM, OpenAICompatLLM
from src.agents.base_react import ReActAgent
from src.tools.wikipedia import MockWikiBackend, WikiEnv

PAGES = {
    "Scott Derrickson": ("Scott Derrickson (born July 16, 1966) is an American "
                         "director, screenwriter and producer."),
    "Ed Wood": ("Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) "
                "was an American filmmaker, actor, and pulp novel author."),
}
SCRIPTS = {
    "Were Scott Derrickson and Ed Wood of the same nationality?": [
        ("I need to search both and compare nationality.", "Search[Scott Derrickson]"),
        ("Scott Derrickson is American. Now Ed Wood.", "Search[Ed Wood]"),
        ("Both American, so yes.", "Finish[yes]"),
    ],
}


def next_step(text: str):
    idx = text.rfind("Question:")
    tail = text[idx:]
    question = tail.split("\n", 1)[0].replace("Question:", "").strip()
    step = tail.count("Observation ")
    thought, action = SCRIPTS[question][step]
    return step, thought, action


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")

        if self.path == "/api/generate":
            step, thought, action = next_step(req["prompt"])
            resp = f" {thought}\nAction {step + 1}: {action}"
            self._send({"response": resp, "done": True,
                        "prompt_eval_count": 1500, "eval_count": 20})

        elif self.path == "/v1/chat/completions":
            user = next(m["content"] for m in req["messages"] if m["role"] == "user")
            step, thought, action = next_step(user)
            content = f"Thought {step + 1}: {thought}\nAction {step + 1}: {action}"
            self._send({"choices": [{"message": {"role": "assistant",
                                                 "content": content}}],
                        "usage": {"prompt_tokens": 1500, "completion_tokens": 20}})
        else:
            self._send({"error": "unknown path"})


def serve():
    httpd = HTTPServer(("127.0.0.1", 0), StubHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def run_with(llm, label):
    env = WikiEnv(MockWikiBackend(PAGES))
    agent = ReActAgent(llm, env, max_steps=8, verbose=False)
    q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    res = agent.run(q, gold="yes")
    print(f"[{label}] pred={res.prediction!r} finished={res.finished} "
          f"steps={res.n_steps} tokens(in/out)={res.prompt_tokens}/{res.completion_tokens}")
    assert res.finished and res.prediction == "yes", f"{label} FAILED"
    assert res.prompt_tokens > 0 and res.completion_tokens > 0, "token tracking error"


if __name__ == "__main__":
    httpd, port = serve()
    base = f"http://127.0.0.1:{port}"
    print(f"Stub server running at {base}\n")

    run_with(OllamaLLM(model="stub", base_url=base), "Ollama")
    run_with(OpenAICompatLLM(base_url=f"{base}/v1", model="stub", api_key=""),
             "OpenAI-compat")

    httpd.shutdown()
    print("\nALL BACKEND CHECKS PASSED ✓")
