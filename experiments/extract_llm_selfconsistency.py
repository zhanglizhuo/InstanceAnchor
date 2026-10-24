#!/usr/bin/env python3
"""
experiments/extract_llm_selfconsistency.py — LLM self-consistency sampling
==========================================================================
Collects N sampled answers per question from a local Ollama model on
multiple-choice benchmarks with FIXED class count C, so that the kappa
plurality decomposition (the paper's core diagnostic) transfers directly:

  "backbones"  =  sampled rollouts (exchangeable draws of one model)
  "classes"    =  the C answer options (fixed per dataset)
  "consensus"  =  plurality answer over N samples
  "agreement"  =  fraction of samples voting the consensus answer

Datasets (single-parquet direct downloads; answers publicly available):
  - MMLU-Pro validation (C=10, 70 questions)     [primary, hard]
  - MMLU all-test (C=4, 14,042 -> 500 shuffled)  [secondary, C-scaling probe]

Model: qwen3.5-9b via Ollama (V100 box). Temperature 0.7, sequential-ish
with a small request pool (Ollama queues internally).

Output (resumable JSONL): results/per_llm/llm_sc_{model}_{dataset}_{n}.jsonl
  per line: {"idx", "gt", "C", "options", "votes": [int] (-1 = unparseable),
             "n_valid"}

Usage (V100):
  ~/.../python experiments/extract_llm_selfconsistency.py \
      --datasets mmlu_pro,mmlu --n_samples 32 --model qwen3.5-9b \
      --ollama_url http://127.0.0.1:11434
"""
import argparse
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATASETS = {
    "mmlu_pro": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "file": "data/validation-00000-of-00001.parquet",
        "n_questions": 70, "C": 10, "answer_mode": "letter",
        "opt_col": "options",
    },
    "mmlu": {
        "repo": "cais/mmlu",
        "file": "all/test-00000-of-00001.parquet",
        "n_questions": 500, "C": 4, "answer_mode": "index",
        "opt_col": "choices",
    },
}
OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "per_llm"
LETTERS = "ABCDEFGHIJ"


def setup_logging(ts):
    logdir = Path(__file__).resolve().parents[1] / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logdir / f"extract_llm_sc_{ts}.log"),
                  logging.StreamHandler()])
    return logging.getLogger()


def load_questions(key, cache_dir, log):
    """Deterministic list of dicts {question, options, answer_idx}."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    cfg = DATASETS[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(cfg["repo"], cfg["file"], repo_type="dataset",
                        cache_dir=str(cache_dir))
    t = pq.read_table(p)
    n = cfg["n_questions"]
    idxs = np.arange(t.num_rows)
    if t.num_rows > n:
        idxs = np.random.RandomState(42).choice(t.num_rows, size=n,
                                                replace=False)
        idxs.sort()
    rows = []
    for i in idxs:
        opts = t.column(cfg["opt_col"]).take([i])[0].as_py()
        ans = t.column("answer").take([i])[0].as_py()
        rows.append({
            "question": t.column("question").take([i])[0].as_py(),
            "options": list(opts),
            "answer_idx": LETTERS.index(ans.strip().upper())
            if cfg["answer_mode"] == "letter" else int(ans),
        })
    log.info(f"{key}: {len(rows)} questions loaded from {p}")
    return rows


def build_prompt(q):
    opt_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return (f"Question: {q['question']}\n"
            f"Options:\n{opt_txt}\n\n"
            f"Answer with ONLY the letter of the correct option. "
            f"Do not explain.")


def parse_answer(text, C):
    """Option letter -> index, or -1. Robust chain:
    1. 'Answer: X' / 'answer is X' / 'option X' / 'choice X'
    2. last standalone letter A-J in the response"""
    t = text.strip()
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Ja-j])", t,
                  re.IGNORECASE)
    if m:
        return LETTERS.index(m.group(1).upper())
    letters = re.findall(r"\b([A-Ja-j])\b", t)
    if letters:
        return LETTERS.index(letters[-1].upper())
    return -1


def sample_once(question, C, model, ollama_url, temperature, log):
    import requests
    payload = {
        "model": model,
        "prompt": build_prompt(question),
        "stream": False,
        "keep_alive": "1h",
        "think": False,
        "options": {"temperature": temperature, "num_predict": 64},
    }
    try:
        r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return parse_answer(r.json()["response"], C)
    except Exception as e:  # noqa: BLE001
        log.warning(f"sample error: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mmlu_pro,mmlu")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(ts)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = Path(__file__).resolve().parents[1] / "data" / "llm_sc_dl"

    for key in args.datasets.split(","):
        cfg = DATASETS[key]
        questions = load_questions(key, cache, log)
        out_path = OUT_ROOT / f"llm_sc_{args.model}_{key}_{args.n_samples}.jsonl"
        done = set()
        if out_path.exists():
            for line in open(out_path):
                done.add(json.loads(line)["idx"])
        log.info(f"{key}: {len(done)} done, resuming")

        def sample_question(idx_q):
            idx, q = idx_q
            votes = [sample_once(q, cfg["C"], args.model, args.ollama_url,
                                 args.temperature, log)
                     for _ in range(args.n_samples)]
            valid = sum(1 for v in votes if 0 <= v < cfg["C"])
            return {"idx": idx, "gt": q["answer_idx"], "C": cfg["C"],
                    "question": q["question"], "options": q["options"],
                    "votes": votes, "n_valid": valid}

        pending = [(i, q) for i, q in enumerate(questions) if i not in done]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for k, rec in enumerate(pool.map(sample_question, pending)):
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if (k + 1) % 10 == 0:
                    log.info(f"{key}: {k+1}/{len(pending)} done "
                             f"({time.time()-t0:.0f}s)")
        log.info(f"{key}: complete -> {out_path}")


if __name__ == "__main__":
    main()
