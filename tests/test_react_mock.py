import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import MockLLM
from src.agents.base_react import ReActAgent
from src.tools.wikipedia import MockWikiBackend, WikiEnv

PAGES = {
    "Scott Derrickson": ("Scott Derrickson (born July 16, 1966) is an American "
                         "director, screenwriter and producer. He lives in Los "
                         "Angeles, California."),
    "Ed Wood": ("Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) "
                "was an American filmmaker, actor, and pulp novel author."),
    "Milhouse": ("Milhouse Mussolini Van Houten is a recurring character in the "
                 "Fox animated television series The Simpsons. Milhouse was named "
                 "after U.S. president Richard Nixon, whose middle name was Milhous."),
}

SCRIPTS = {
    "Were Scott Derrickson and Ed Wood of the same nationality?": [
        ("I need to search Scott Derrickson and Ed Wood, then compare nationality.",
         "Search[Scott Derrickson]"),
        ("Scott Derrickson is American. Now search Ed Wood.",
         "Search[Ed Wood]"),
        ("Ed Wood is also American, so the answer is yes.",
         "Finish[yes]"),
    ],
    "Who was Milhouse named after?": [
        ("I need to search Milhouse and find who it is named after.",
         "Search[Milhouse]"),
        ("The first sentence doesn't say; let me look up 'named after'.",
         "Lookup[named after]"),
        ("Milhouse was named after Richard Nixon.",
         "Finish[Richard Nixon]"),
    ],
}


def run_case(question: str, gold: str):
    env = WikiEnv(MockWikiBackend(PAGES))
    agent = ReActAgent(MockLLM(SCRIPTS), env, max_steps=8, verbose=True)
    print(f"===== Q: {question} =====")
    res = agent.run(question, gold=gold)
    print(f"Prediction : {res.prediction!r}  | gold: {gold!r}  "
          f"| match: {res.prediction.strip().lower() == gold.lower()}")
    print(f"Steps={res.n_steps}  LLM_calls={res.n_llm_calls}  "
          f"tokens(in/out)={res.prompt_tokens}/{res.completion_tokens}\n")
    return res


if __name__ == "__main__":
    r1 = run_case("Were Scott Derrickson and Ed Wood of the same nationality?", "yes")
    r2 = run_case("Who was Milhouse named after?", "Richard Nixon")

    assert r1.finished and r1.prediction == "yes"
    assert r2.finished and r2.prediction == "Richard Nixon"
    assert "(Result 1 / 1)" in r2.trajectory
    print("ALL CHECKS PASSED ✓")
