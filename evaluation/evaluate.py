"""Held-out evaluation: cache features, select models on development CV, then score the test set once."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
os.environ["SOURCE_DATE_EPOCH"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from main import read_rows, set_seeds
from method.framing.features import FEATURE_NAMES, featurize_many, load_lexicon, load_nlp
CACHE = Path("data/preprocessed/features.csv")
C_GRID, THRESHOLDS, K_GRID = (0.01, 0.1, 1.0, 10.0, 100.0), tuple(i / 10 for i in range(1, 10)), (1, 3, 5, 7)
CV_FIELDS = ["model", "config", "fold", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
TEST_FIELDS = ["model", "config", "cv_f1", "cv_precision", "cv_recall", "cv_auc", "undefined_auc_folds", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
METRIC_KEYS = {"precision", "recall", "f1", "roc_auc", "cv_f1", "cv_precision", "cv_recall", "cv_auc"}

# counts how many articles have label 0 and how many have label 1
def class_counts(y: np.ndarray) -> dict[str, int]:
    counts = Counter(int(v) for v in y)
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}

# gets the four feature numbers for every article. if we already saved them to
# data/preprocessed/features.csv for this exact data, it reuses them; otherwise it
# runs spacy once and saves the numbers (no article text gets saved)
def feature_matrix(rows: list[dict], lexicon: str) -> np.ndarray:
    hashes = [int.from_bytes(hashlib.sha256(f"{r['headline']}\n{r['body']}".encode()).digest()[:8], "big") for r in rows]
    labels = [r["label"] for r in rows]
    cached = list(csv.DictReader(CACHE.open(encoding="utf-8", newline=""))) if CACHE.exists() else []
    match = len(cached) == len(rows) and [int(r["content_hash"]) for r in cached] == hashes and [int(r["label"]) for r in cached] == labels
    if not match:
        print(f"Computing features for {len(rows)} rows -> {CACHE}")
        matrix = featurize_many([(r["headline"], r["body"]) for r in rows], load_nlp(), load_lexicon(lexicon))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["content_hash", "label", *FEATURE_NAMES])
            for digest, label, values in zip(hashes, labels, matrix):
                writer.writerow([digest, label, *[f"{float(v):.8f}" for v in values]])
        cached = list(csv.DictReader(CACHE.open(encoding="utf-8", newline="")))
    else:
        print(f"Reusing cached features from {CACHE}")
    return np.array([[float(r[name]) for name in FEATURE_NAMES] for r in cached], dtype=float)

# decides which features to use. if headline_attribution is 0 for every article in
# the development set it tells the model nothing, so we drop it
def choose_columns(x_dev: np.ndarray) -> list[str]:
    names = list(FEATURE_NAMES)
    if np.all(x_dev[:, names.index("headline_attribution")] == 0):
        print("Dropping headline_attribution: it is 0 on every development row.")
        names.remove("headline_attribution")
    else:
        print("Using all four features; headline_attribution varies on the development set.")
    return names

# builds one model: a scaler followed by logistic regression, knn, or naive bayes.
# keeping the scaler inside the pipeline means it only learns from the training fold
def make_pipe(kind: str, seed: int, value: float) -> Pipeline:
    if kind == "lr":
        clf = LogisticRegression(C=value, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=seed)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(value), algorithm="brute")
    else:
        clf = GaussianNB()
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

# lists every model and setting we want to try (each C and threshold for logistic
# regression, each k for knn, and naive bayes). the order is fixed so that if two
# settings tie, the earlier one wins every time
def build_specs(names: list[str], seed: int) -> list[dict]:
    full = [FEATURE_NAMES.index(name) for name in names]
    headline = [FEATURE_NAMES.index(name) for name in names if name.startswith("headline_")]
    specs: list[dict] = []
    for c_value in C_GRID:
        for threshold in THRESHOLDS:
            config = f"C={c_value:g};threshold={threshold:.1f}"
            for model, cols in (("logistic_regression", full), ("logistic_regression_headline", headline)):
                specs.append({"model": model, "config": config, "cols": cols, "threshold": threshold, "pipe": make_pipe("lr", seed, c_value)})
    for k_value in K_GRID:
        specs.append({"model": "knn", "config": f"k={k_value}", "cols": full, "threshold": 0.5, "pipe": make_pipe("knn", seed, k_value)})
    specs.append({"model": "gaussian_nb", "config": "default", "cols": full, "threshold": 0.5, "pipe": make_pipe("nb", seed, 0)})
    return specs

# splits the development set into cross-validation folds one time, so every model is
# tested on exactly the same splits. if a class is too small for the number of folds,
# it lowers the fold count and prints a warning
def build_folds(y_dev: np.ndarray, folds: int, repeats: int, seed: int) -> tuple[int, list]:
    smallest = min(class_counts(y_dev).values())
    if smallest < 2:
        sys.exit(f"Development class counts {class_counts(y_dev)} cannot support stratified CV.")
    if folds > smallest:
        print(f"Warning: reducing folds from {folds} to {smallest} (smaller development class has {smallest}).")
        folds = smallest
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return folds, list(splitter.split(np.zeros(len(y_dev)), y_dev))

# scores one set of predictions: precision, recall, f1, roc auc, and the confusion
# matrix counts, treating "biased" (1) as the positive class. roc auc is left as nan
# when the fold only has one class in it, since it can't be computed then
def fold_scores(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, pred, labels=[0, 1]).ravel())
    auc = float("nan") if len(set(y_true.tolist())) < 2 else float(roc_auc_score(y_true, proba))
    score = lambda metric, guess: float(metric(y_true, guess, pos_label=1, zero_division=0))
    return {"precision": score(precision_score, pred), "recall": score(recall_score, pred), "f1": score(f1_score, pred), "roc_auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

# trains a fresh copy of the model on the training rows and returns how likely each
# test row is to be biased
def positive_proba(spec: dict, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = clone(spec["pipe"]).fit(x_train[:, spec["cols"]], y_train)
    return model.predict_proba(x_test[:, spec["cols"]])[:, 1]

# runs every model setting on every development fold and records the scores for each one
def cross_validate(x_dev: np.ndarray, y_dev: np.ndarray, folds: list, specs: list[dict]) -> list[dict]:
    rows = []
    for spec in specs:
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            proba = positive_proba(spec, x_dev[train_idx], y_dev[train_idx], x_dev[test_idx])
            rows.append({"model": spec["model"], "config": spec["config"], "fold": fold_id, **fold_scores(y_dev[test_idx], proba, spec["threshold"])})
    return rows

# picks the best setting for each model by its average f1 across the folds. it also
# averages precision, recall, and roc auc, and counts how many folds had no roc auc
def choose_best(cv_rows: list[dict]) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cv_rows:
        grouped[(row["model"], row["config"])].append(row)
    best: dict[str, dict] = {}
    for (model, config), rows in grouped.items():
        f1 = float(np.mean([row["f1"] for row in rows]))
        if model in best and f1 <= best[model]["cv_f1"]:
            continue
        aucs = np.array([row["roc_auc"] for row in rows], dtype=float)
        undefined = int(np.isnan(aucs).sum())
        best[model] = {"model": model, "config": config, "cv_f1": f1, "n_folds": len(rows), "undefined_auc_folds": undefined,
                       "cv_precision": float(np.mean([row["precision"] for row in rows])), "cv_recall": float(np.mean([row["recall"] for row in rows])),
                       "cv_auc": float(np.nanmean(aucs)) if undefined < len(aucs) else float("nan")}
    return best

# retrains each model's best setting on the whole development set, then scores it on
# the test set. this is the only place the test set gets used
def evaluate_test(x: np.ndarray, y: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, best: dict[str, dict], specs: list[dict]) -> list[dict]:
    by_key = {(spec["model"], spec["config"]): spec for spec in specs}
    results = []
    for chosen in best.values():
        spec = by_key[(chosen["model"], chosen["config"])]
        proba = positive_proba(spec, x[dev_idx], y[dev_idx], x[test_idx])
        results.append({**chosen, **fold_scores(y[test_idx], proba, spec["threshold"])})
    return results

# writes a list of result rows to a csv file, rounding the scores to 6 decimal places
def write_table(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            saved = {key: ("nan" if np.isnan(row[key]) else f"{row[key]:.6f}") if key in METRIC_KEYS else row[key] for key in fields}
            writer.writerow(saved)

# saves all the result files: the cv and test score tables, the confusion matrices,
# the dataset counts, and the settings this run used
def write_outputs(out_dir: Path, cv_rows: list[dict], test_rows: list[dict], stats: dict, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(out_dir / "cv_results.csv", CV_FIELDS, cv_rows)
    write_table(out_dir / "test_results.csv", TEST_FIELDS, test_rows)
    matrix_of = lambda row: [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    payload = {
        "cv": [{"model": r["model"], "config": r["config"], "fold": r["fold"], "matrix": matrix_of(r)} for r in cv_rows],
        "pooled_test": [{"model": r["model"], "config": r["config"], "matrix": matrix_of(r)} for r in test_rows],
    }
    for name, body in (("confusion_matrices.json", payload), ("dataset_stats.json", stats), ("run_config.json", config)):
        (out_dir / name).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

# draws three charts of average cv f1: against C, against the decision threshold
# (both for logistic regression), and against k (for knn)
def plot_curves(out_dir: Path, cv_rows: list[dict]) -> None:
    # for one model, groups the f1 scores by a single setting (like C) and averages
    # each group, so we get one point per value of that setting
    def series(model: str, name: str) -> tuple[list[float], list[float]]:
        buckets: dict[float, list[float]] = defaultdict(list)
        for row in cv_rows:
            value = next((float(part.split("=", 1)[1]) for part in row["config"].split(";") if part.startswith(name + "=")), None) if row["model"] == model else None
            if value is not None:
                buckets[value].append(row["f1"])
        keys = sorted(buckets)
        return keys, [float(np.mean(buckets[key])) for key in keys]

    lr = ("logistic_regression", "logistic_regression_headline")
    jobs = (("cv_f1_vs_C.png", "C", "C", True, lr), ("cv_f1_vs_threshold.png", "threshold", "Decision threshold", False, lr), ("cv_f1_vs_k.png", "k", "k", False, ("knn",)))
    for filename, key, xlabel, log_x, models in jobs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for model in models:
            ax.plot(*series(model, key), marker="o", label=model)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean CV F1 (class 1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=100, metadata={"Software": "", "Creation Time": None})
        plt.close(fig)

# prints a summary table: each model's best setting, its cv scores, and its test scores
def print_summary(test_rows: list[dict]) -> None:
    print(f"{'model':<32} {'config':<22} {'cv_f1':>6} {'cv_auc':>7} {'undef':>5} {'test_f1':>7} {'test_p':>6} {'test_r':>6} {'test_auc':>8}")
    for row in test_rows:
        auc = "nan" if np.isnan(row["cv_auc"]) else f"{row['cv_auc']:.3f}"
        print(f"{row['model']:<32} {row['config']:<22} {row['cv_f1']:.3f} {auc:>7} {row['undefined_auc_folds']:>2}/{row['n_folds']:<2} {row['f1']:.3f} {row['precision']:.3f} {row['recall']:.3f} {row['roc_auc']:.3f}")

# runs the whole evaluation: reads the data, splits off the test set, tunes each
# model with cross-validation, scores the winners on the test set, and saves everything
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--lexicon", required=True)
    for flag, default in (("--headline-col", "headline"), ("--body-col", "body"), ("--label-col", "label"), ("--out", "evaluation/results")):
        parser.add_argument(flag, default=default)
    for flag, kind, default in (("--seed", int, 0), ("--test-size", float, 0.2), ("--folds", int, 3), ("--repeats", int, 5)):
        parser.add_argument(flag, type=kind, default=default)
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1 or args.folds < 2 or args.repeats < 1:
        parser.error("--test-size must be in (0, 1), --folds >= 2, and --repeats >= 1")
    set_seeds(args.seed)
    rows = read_rows(args.data, args.headline_col, args.body_col, args.label_col)
    y = np.array([row["label"] for row in rows])
    if min(class_counts(y).values()) < 2:
        sys.exit(f"Need at least 2 samples of each class; got {class_counts(y)}.")
    matrix = feature_matrix(rows, args.lexicon)
    try:
        dev_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, random_state=args.seed, stratify=y)
    except ValueError as exc:
        sys.exit(f"Stratified split failed: {exc}")
    dev_idx, test_idx = np.asarray(dev_idx), np.asarray(test_idx)
    print(f"Development: n={len(dev_idx)} class counts {class_counts(y[dev_idx])}")
    print(f"Test: n={len(test_idx)} class counts {class_counts(y[test_idx])}")
    names = choose_columns(matrix[dev_idx])
    folds_used, folds = build_folds(y[dev_idx], args.folds, args.repeats, args.seed)
    specs = build_specs(names, args.seed)
    smallest_train = min(len(train_idx) for train_idx, _ in folds)
    if smallest_train < max(K_GRID):
        sys.exit(f"Smallest CV training fold has {smallest_train} rows, which is below k={max(K_GRID)}.")
    print(f"CV: {folds_used} folds x {args.repeats} repeats = {len(folds)} splits, reused for every model.")
    print(f"Features: {names}")
    cv_rows = cross_validate(matrix[dev_idx], y[dev_idx], folds, specs)
    test_rows = evaluate_test(matrix, y, dev_idx, test_idx, choose_best(cv_rows), specs)
    stats = {"n": len(y), "class_counts": class_counts(y), "development": {"n": int(len(dev_idx)), "class_counts": class_counts(y[dev_idx])}, "test": {"n": int(len(test_idx)), "class_counts": class_counts(y[test_idx])}}
    config = {"seed": args.seed, "folds": folds_used, "folds_requested": args.folds, "repeats": args.repeats, "test_size": args.test_size, "feature_set": names, "dropped_features": [name for name in FEATURE_NAMES if name not in names], "headline_ablation_features": [name for name in names if name.startswith("headline_")], "lexicon_path": args.lexicon, "positive_class": 1}
    out_dir = Path(args.out)
    write_outputs(out_dir, cv_rows, test_rows, stats, config)
    plot_curves(out_dir, cv_rows)
    print_summary(test_rows)
    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
