"""
merge_results.py — Merge per-model baseline_results.json into one file.
Usage:
  python experiments/merge_results.py
  # reads results/baseline/*/baseline_results.json
  # writes results/baseline/merged_baseline_results.json
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "baseline"

merged = {}
shard_dirs = sorted(RESULTS_DIR.glob("*"))
shard_dirs = [d for d in shard_dirs if d.is_dir() and (d / "baseline_results.json").exists()]

for shard_dir in shard_dirs:
    with open(shard_dir / "baseline_results.json") as f:
        payload = json.load(f)
    for subset, subset_data in payload.items():
        if subset not in merged:
            merged[subset] = {"class_names": subset_data.get("class_names", [])}
        for model_key, model_data in subset_data.items():
            if model_key == "class_names":
                continue
            if model_key in merged[subset]:
                merged[subset][model_key].update(model_data)
            else:
                merged[subset][model_key] = model_data

out_path = RESULTS_DIR / "merged_baseline_results.json"
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"Merged {len(shard_dirs)} shards → {out_path}")
