"""Trains the headline framing-bias classifier.

Loads the headline/body dataset and MPQA subjectivity lexicon (subjclues.tff),
extracts the four features from each article and then trains a class-weighted L2-regularized logistic regression model (because we don't
want it to be good just on our training data.)

Saves the learned model weights and training predictions to the output directory. 
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from method.framing.features import FEATURE_NAMES, featurize, featurize_many, load_lexicon, load_nlp


def set_seeds(seed: int) -> None:
    """Same input + same seed -> same output."""
    random.seed(seed)
    np.random.seed(seed)


def read_rows(path, headline_col: str, body_col: str, label_col: str) -> list[dict]:
    """Read and validate the labeled CSV."""
    # utf-8-sig: spreadsheet exports often start with a byte-order mark, which would
    # otherwise end up glued to the first column name.
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {headline_col, body_col, label_col} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"{path}: missing column(s) {sorted(missing)}. Found: {reader.fieldnames}")
        rows = []
        for n, r in enumerate(reader, start=1):
            # Collapse stray newlines and repeated spaces (some headlines contain line breaks).
            headline = " ".join((r[headline_col] or "").split())
            body = " ".join((r[body_col] or "").split())
            label = (r[label_col] or "").strip()
            if not headline or not body:
                sys.exit(f"{path}: data row {n} has an empty headline or body")
            if label not in {"0", "1"}:
                sys.exit(f"{path}: data row {n} has label {label!r}; expected 0 or 1")
            rows.append({
                "id": (r.get("id") or str(n)).strip(),
                "headline": headline,
                "body": body,
                "label": int(label),
            })
    if not rows:
        sys.exit(f"{path}: no data rows")
    return rows


def train(X: np.ndarray, y: np.ndarray, C: float, seed: int):
    """Standardize the features, then fit the logistic regression.
    
    X - Matrix containing the four features for every article
    y - Correct labels for every article 
    C - For controlling the strength of L2 regularization
    seed - random seed for reproducible results 

    Purpose: Takes the extracted features and their labels, then trains the logistic regression classifier.
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def prob_biased(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """P(label = 1) for each row of X.
    
    We use scaler (standardized features) and clf (logistic regression model) to make predictions

    Purpose: Calculates the probability that each article is classified as biased.
    """
    return clf.predict_proba(scaler.transform(X))[:, 1]


def save_outputs(out_dir: Path, rows, probs, preds, scaler, clf, args, counts, lexicon_size) -> None:
    """
    Purpose: Saves the results of our experiment into files so we can inspect them later (weights.json and predictions.csv). 
    """
    
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        "feature_names": FEATURE_NAMES,
        "weights": [round(float(w), 6) for w in clf.coef_[0]], # saves the learned weights for each feature
        "intercept": round(float(clf.intercept_[0]), 6),
        "scaler_mean": [round(float(m), 6) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 6) for s in scaler.scale_],
        "lexicon_path": str(args.lexicon),
        "lexicon_size": lexicon_size,
        "include_weak": args.include_weak,
        "C": args.C,
        "threshold": args.threshold,
        "seed": args.seed,
        "n_rows": len(rows),
        "label_counts": counts,
    }
    with open(out_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "train_prob", "train_pred", "headline"])
        for r, p, yhat in zip(rows, probs, preds):
            writer.writerow([r["id"], r["label"], f"{p:.6f}", int(yhat), r["headline"]])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="labeled CSV")
    parser.add_argument("--lexicon", required=True,
                        help="word bank: MPQA subjclues.tff, or a text file with one lemma per line")
    parser.add_argument("--include-weak", action="store_true",
                        help="also use weaksubj entries from the MPQA lexicon (default: strongsubj only)")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--headline-col", default="headline")
    parser.add_argument("--body-col", default="body")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--C", type=float, default=1.0, help="inverse regularization strength")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold tau")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predict-headline", help="optionally score one new headline (needs --predict-body)")
    parser.add_argument("--predict-body", help="body text for --predict-headline")
    args = parser.parse_args(argv)

    if bool(args.predict_headline) != bool(args.predict_body):
        parser.error("--predict-headline and --predict-body must be given together")

    set_seeds(args.seed)

    rows = read_rows(args.data, args.headline_col, args.body_col, args.label_col)
    y = np.array([r["label"] for r in rows])
    counts = {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())}
    if len(counts) < 2:
        sys.exit(f"Need both labels (0 and 1) in the data to fit the model; got {counts}")

    nlp = load_nlp()
    lexicon = load_lexicon(args.lexicon, include_weak=args.include_weak)
    X = featurize_many([(r["headline"], r["body"]) for r in rows], nlp, lexicon)

    scaler, clf = train(X, y, args.C, args.seed)
    probs = prob_biased(scaler, clf, X)
    preds = (probs >= args.threshold).astype(int)
    save_outputs(Path(args.out), rows, probs, preds, scaler, clf, args, counts, len(lexicon))

    print(f"Word bank: {len(lexicon)} entries from {args.lexicon}")
    print(f"Trained on {len(rows)} rows, label counts {counts}")
    print("Learned weights (standardized features; positive = raises P(biased)):")
    for name, w in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:22s} {w:+.3f}")
    print(f"Saved weights.json and predictions.csv to {args.out}/")
    print("Note: predictions.csv is in-sample (training data), not an evaluation.")

    if args.predict_headline:
        x = featurize(args.predict_headline, args.predict_body, nlp, lexicon).reshape(1, -1)
        p = float(prob_biased(scaler, clf, x)[0])
        print(f"New pair: P(biased) = {p:.3f} -> label {int(p >= args.threshold)}")


if __name__ == "__main__":
    main()
