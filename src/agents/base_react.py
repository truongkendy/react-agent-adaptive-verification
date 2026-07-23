from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.llm import LLM
from src.agents.prompts import SYSTEM_PROMPT
from src.tools.wikipedia import WikiEnv

if TYPE_CHECKING:
    from src.verification.base import BaseVerifier


@dataclass
class ReActResult:
    question: str
    prediction: str | None
    trajectory: str
    n_steps: int
    n_llm_calls: int
    finished: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    gold: str | None = None
    layer1_violations: int = 0


class ReActAgent:
    def __init__(self, llm: LLM, env: WikiEnv, max_steps: int = 8,
                 verbose: bool = False,
                 verifier: "BaseVerifier | None" = None):
        self.llm = llm
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose
        self.verifier = verifier

    def run(self, question: str, gold: str | None = None) -> ReActResult:
        self.env.reset()
        if self.verifier:
            self.verifier.reset()
        user = f"Question: {question}"
        traj = ""
        prediction, finished, n_calls = None, False, 0
        tok_in0, tok_out0 = self.llm.prompt_tokens, self.llm.completion_tokens

        i = 0
        for i in range(1, self.max_steps + 1):
            out = self.llm.generate(
                SYSTEM_PROMPT, user, traj + f"Thought {i}:",
                stop=[f"\nObservation {i}:"],
            ).strip()
            n_calls += 1

            thought, action = self._parse(out, i)
            if action is not None:
                action = action.split("\n")[0].strip()
            if action is None:
                action = self.llm.generate(
                    SYSTEM_PROMPT, user,
                    traj + f"Thought {i}: {thought}\nAction {i}:",
                    stop=["\n"],
                ).strip()
                n_calls += 1

            if self.verifier:
                ok, err = self.verifier.check(action)
                if not ok:
                    obs = f"[Layer1] {err}"
                    if self.verbose:
                        print(f"[Layer1] Violation step {i}: {err}")
                    traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
                    continue

            obs, done = self.env.step(action)
            obs = obs.replace("\\n", "")
            if self.verifier:
                # Record only executed actions: `check()` no longer does it, so a
                # blocked action can be retried after the model rewrites it.
                self.verifier.commit(action)

            if self.verbose:
                print(f"Thought {i}: {thought}")
                print(f"Action {i}: {action}")
                print(f"Observation {i}: {obs}\n")

            traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"

            if done:
                prediction = self.env.answer
                finished = True
                break

        n_violations = self.verifier.violation_count if self.verifier else 0
        return ReActResult(
            question=question,
            prediction=prediction,
            trajectory=traj.strip(),
            n_steps=i,
            n_llm_calls=n_calls,
            finished=finished,
            prompt_tokens=self.llm.prompt_tokens - tok_in0,
            completion_tokens=self.llm.completion_tokens - tok_out0,
            gold=gold,
            layer1_violations=n_violations,
        )

    @staticmethod
    def _parse(out: str, i: int) -> tuple[str, str | None]:
        out = out.strip()
        if out.startswith(f"Thought {i}:"):
            out = out[len(f"Thought {i}:"):].strip()
        marker = f"\nAction {i}:"
        if marker in out:
            thought, action = out.split(marker, 1)
            return thought.strip(), action.strip()
        marker = f"Action {i}:"
        if marker in out:
            thought, action = out.split(marker, 1)
            return thought.strip(), action.strip()
        return out.split("\n")[0].strip(), None
