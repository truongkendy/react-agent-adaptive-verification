from setuptools import setup, find_packages

setup(
    name="react-agent-adaptive-verification",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests",
        "anthropic",
        # NLI stack for Layer 3 (retrieval + DeBERTa) is now a base dependency
        # rather than an optional extra.
        "transformers>=4.40",
        "torch>=2.2",
        "sentencepiece",
    ],
    # Kept as an empty alias so the documented `pip install -e ".[nli]"` still
    # resolves; the packages themselves now live in install_requires above.
    extras_require={
        "nli": [],
    },
)
