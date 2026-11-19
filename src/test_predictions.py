"""Print each test article's predicted P(biased) from the four evaluated models. Terminal output only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate as ev
from main import read_rows, set_seeds
from method.framing.features import FEATURE_NAMES, featurize_many, load_lexicon, load_nlp
MODELS = (("logistic_regression", "lr"), ("logistic_regression_headline", "lr_headline"), ("knn", "knn"), ("gaussian_nb", "nb"))

# loads the feature numbers without writing any files. it uses the saved cache when it
# matches this exact data, and otherwise works them out in memory with spacy
def load_features(rows: list[dict], lexicon: str) -> np.ndarray:
    hashes = [int.from_bytes(hashlib.sha256(f"{r['headline']}\n{r['body']}".encode()).digest()[:8], "big") for r in rows]
    cached = list(csv.DictReader(ev.CACHE.open(encoding="utf-8", newline=""))) if ev.CACHE.exists() else []
    if len(cached) == len(rows) and [int(r["content_hash"]) for r in cached] == hashes and [int(r["label"]) for r in cached] == [r["label"] for r in rows]:
        return np.array([[float(r[name]) for name in FEATURE_NAMES] for r in cached], dtype=float)
    print("Feature cache does not match this data; computing features in memory (nothing is saved).")
    return featurize_many([(r["headline"], r["body"]) for r in rows], load_nlp(), load_lexicon(lexicon))

# reads which setting each model ended up with, from the evaluation's test_results.csv
def chosen_configs(results: Path) -> dict[str, dict]:
    with (results / "test_results.csv").open(encoding="utf-8", newline="") as handle:
        chosen = {row["model"]: row for row in csv.DictReader(handle)}
    missing = [model for model, _ in MODELS if model not in chosen]
    if missing:
        sys.exit(f"{results / 'test_results.csv'} is missing model(s) {missing}. Run evaluation/evaluate.py first.")
    return chosen

# runs it: rebuilds the same test split, retrains each chosen model on the development
# rows, and prints every test article with each model's probability that it is biased
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/SENG474-CSC503_dataset.csv")
    parser.add_argument("--lexicon", default="data/lexicon/subjclues.tff")
    parser.add_argument("--headline-col", default="title")
    parser.add_argument("--body-col", default="body")
    parser.add_argument("--label-col", default="bias")
    parser.add_argument("--results", default="evaluation/results", help="folder written by evaluation/evaluate.py (read only)")
    args = parser.parse_args(argv)
    results = Path(args.results)
    run = json.loads((results / "run_config.json").read_text(encoding="utf-8"))
    chosen = chosen_configs(results)
    seed, names = int(run["seed"]), list(run["feature_set"])
    set_seeds(seed)
    rows = read_rows(args.data, args.headline_col, args.body_col, args.label_col)
    y = np.array([row["label"] for row in rows])
    x = load_features(rows, args.lexicon)
    dev_idx, test_idx = train_test_split(np.arange(len(y)), test_size=float(run["test_size"]), random_state=seed, stratify=y)
    dev_idx, test_idx = np.asarray(dev_idx), np.asarray(test_idx)
    print(f"Seed {seed}, test size {run['test_size']}: {len(dev_idx)} development rows, {len(test_idx)} test rows")
    print(f"Features: {names}")
    specs = {(spec["model"], spec["config"]): spec for spec in ev.build_specs(names, seed)}
    probs: dict[str, np.ndarray] = {}
    for model, short in MODELS:
        spec = specs.get((model, chosen[model]["config"]))
        if spec is None:
            sys.exit(f"Config {chosen[model]['config']!r} for {model} is not in the evaluation grid.")
        probs[model] = ev.positive_proba(spec, x[dev_idx], y[dev_idx], x[test_idx])
        scored = ev.fold_scores(y[test_idx], probs[model], spec["threshold"])
        saved = {key: int(chosen[model][key]) for key in ("tn", "fp", "fn", "tp")}
        status = "matches test_results.csv" if all(scored[key] == saved[key] for key in saved) else f"DOES NOT MATCH test_results.csv {saved}"
        print(f"  {short:<12} {model:<30} {spec['config']:<22} confusion {[scored[k] for k in ('tn', 'fp', 'fn', 'tp')]} {status}")
    print()
    print(f"{'id':>4}  {'label':>5}  " + "  ".join(f"{short:>11}" for _, short in MODELS) + "  title")
    for position, row_index in enumerate(test_idx):
        row = rows[row_index]
        cells = "  ".join(f"{probs[model][position]:>11.3f}" for model, _ in MODELS)
        print(f"{row['id']:>4}  {row['label']:>5}  {cells}  {row['headline']}")


if __name__ == "__main__":
    main()
