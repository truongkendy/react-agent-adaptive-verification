from __future__ import annotations

import functools
from typing import Callable

NLIFn = Callable[[str, str], dict]

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def _pick_device(prefer: str | None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_deberta_nli(model_name: str = DEFAULT_MODEL,
                      device: str | None = None,
                      max_length: int = 512,
                      normalize_labels: bool = True) -> NLIFn:
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
    except ImportError as e:
        raise RuntimeError(
            "transformers + torch are required for real NLI. Install:\n"
            '    pip install "transformers>=4.40" "torch>=2.2" sentencepiece\n'
            "or use make_mock_nli() in layer3_retrieval.py to run offline."
        ) from e

    dev = _pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(dev)
    model.eval()

    id2label: dict[int, str] = dict(model.config.id2label)

    def _canon(label: str) -> str:
        if not normalize_labels:
            return label
        low = label.strip().lower()
        if low.startswith("entail"):
            return "entailment"
        if low.startswith("contradict"):
            return "contradiction"
        return "neutral"

    @torch.no_grad()
    def nli_fn(premise: str, hypothesis: str) -> dict:
        enc = tokenizer(premise, hypothesis, truncation=True,
                        max_length=max_length, return_tensors="pt").to(dev)
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        return {"label": _canon(id2label[idx]), "score": float(probs[idx].item())}

    return nli_fn


@functools.lru_cache(maxsize=2)
def get_cached_nli(model_name: str = DEFAULT_MODEL) -> NLIFn:
    return build_deberta_nli(model_name)


def _demo() -> None:
    try:
        nli_fn = build_deberta_nli()
    except RuntimeError as e:
        print(e)
        return

    pairs = [
        ("The Eiffel Tower is 330 metres tall.", "The Eiffel Tower is 984 metres tall."),
        ("Mount Everest's peak is 8,849 metres above sea level.",
         "Mount Everest has a peak at 8,849 metres above sea level."),
        ("Paris is the capital of France.", "The population of Tokyo is 14 million."),
    ]
    for premise, hypothesis in pairs:
        out = nli_fn(premise, hypothesis)
        print(f"premise={premise!r}\n  hypothesis={hypothesis!r}\n  -> {out}\n")

    from src.verification.layer3_retrieval import Layer3RetrievalVerifier
    from src.verification.orchestrator import Context, Step

    step = Step(action="Finish[984 metres]",
                thought="The Eiffel Tower is 984 metres tall.",
                goal="Height of the Eiffel Tower?", step_index=2)
    ctx = Context(scratch={"observations": [
        "The Eiffel Tower is a tower in Paris, France. It is 330 metres tall."]})
    sig = Layer3RetrievalVerifier(nli_fn).run(step, ctx)
    print(f"Layer3 (real model): verdict={sig.verdict.value} "
          f"conf={sig.confidence:.3f} suspect={sig.suspected_error_type.value}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _demo()
