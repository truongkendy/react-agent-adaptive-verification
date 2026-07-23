import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hotpotqa"

DIRECT_URLS = {
    "dev_fullwiki":    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    "dev_distractor":  "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "train":           "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
}


def via_huggingface(split: str, config: str = "fullwiki") -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[HF] `datasets` not installed. Skipping. (pip install datasets)")
        return False

    hf_split = "validation" if split.startswith("dev") else "train"
    try:
        print(f"[HF] Downloading hotpot_qa / config={config} / split={hf_split} ...")
        ds = load_dataset("hotpotqa/hotpot_qa", config, split=hf_split,
                          trust_remote_code=True)
    except Exception as e:
        print(f"[HF] Error: {e}")
        return False

    records = []
    for ex in ds:
        sup = ex["supporting_facts"]
        context = ex["context"]
        records.append({
            "_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "supporting_facts": list(zip(sup["title"], sup["sent_id"])),
            "context": list(zip(context["title"], context["sentences"])),
        })

    out = DATA_DIR / f"hotpot_{split}_{config}.json"
    out.write_text(json.dumps(records, ensure_ascii=False))
    print(f"[HF] OK -> {out}  ({len(records)} questions)")
    return True


def via_hf_parquet(split: str, config: str = "fullwiki") -> bool:
    """Third source, and as of 2026-08-14 the only one that works: read the
    Hub's converted parquet directly.

    `datasets` is not a dependency of this repo and `curtis.ml.cmu.edu` (the
    original host, plain HTTP) now times out, so both older paths fail. This one
    needs only `pyarrow`. The Hub schema is column-oriented
    (`supporting_facts: {title: [...], sent_id: [...]}`) and is transposed here
    into the original file's list-of-pairs shape, which is what
    `HotpotExample.from_raw` and `DistractorBackend.load` expect.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[HF-parquet] `pyarrow` not installed. Skipping. (pip install pyarrow)")
        return False

    import io
    import json as _json
    import urllib.request

    hf_split = "validation" if split.startswith("dev") else "train"
    index = (f"https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet")
    try:
        print(f"[HF-parquet] Listing shards for {config}/{hf_split} ...")
        listing = _json.loads(urllib.request.urlopen(index, timeout=60).read())
        urls = listing[config][hf_split]
    except Exception as e:
        print(f"[HF-parquet] Error listing shards: {e}")
        return False

    records = []
    try:
        for u in urls:
            print(f"[HF-parquet] Downloading {u.rsplit('/', 3)[-1]} ...")
            buf = io.BytesIO(urllib.request.urlopen(u, timeout=300).read())
            for ex in pq.read_table(buf).to_pylist():
                sup, ctx = ex["supporting_facts"], ex["context"]
                records.append({
                    "_id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "type": ex.get("type", ""),
                    "level": ex.get("level", ""),
                    "supporting_facts": list(zip(sup["title"], sup["sent_id"])),
                    "context": list(zip(ctx["title"], ctx["sentences"])),
                })
    except Exception as e:
        print(f"[HF-parquet] Error: {e}")
        return False

    out = DATA_DIR / f"hotpot_{split}_{config}.json"
    out.write_text(_json.dumps(records, ensure_ascii=False))
    print(f"[HF-parquet] OK -> {out}  ({len(records)} questions)")
    return True


def via_direct_url(split_key: str) -> bool:
    try:
        import requests
    except ImportError:
        print("[URL] `requests` required (pip install requests).")
        return False

    url = DIRECT_URLS[split_key]
    out = DATA_DIR / f"hotpot_{split_key}.json"
    print(f"[URL] Downloading {url} ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    except Exception as e:
        print(f"[URL] Error: {e}")
        return False
    print(f"[URL] OK -> {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--config", default="fullwiki", choices=["fullwiki", "distractor"])
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    split = f"{args.split}_{args.config}" if args.split == "dev" else "train"

    hf_split = args.split if args.split == "train" else "dev"
    if via_huggingface(hf_split, args.config):
        return
    print("-> Falling back to Hub parquet...")
    if via_hf_parquet(hf_split, args.config):
        return
    print("-> Falling back to original URL...")
    key = split if split in DIRECT_URLS else "dev_fullwiki"
    if not via_direct_url(key):
        print("\n!!! All three sources failed. Check network/firewall.")
        sys.exit(1)


if __name__ == "__main__":
    main()
