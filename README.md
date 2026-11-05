# seng-474-csc-503-loaded-framing-detection

## 1. Abstract

Framing in news headlines can influence how readers interpret social issues, particularly when language emphasizes one perspective or intensifies evaluative meaning. This project investigates framing bias in American news headlines covering social issues in other countries, using the corresponding article as contextual information. We operationalize framing bias through one-sided terms and subjective intensifiers and formulate its detection as a binary classification task. The resulting classifier will be used to examine differences in framing patterns across countries and issue types within American international news coverage.

Repository: [Ethan-J-LeeMMV/seng-474-csc-503-loaded-framing-detection](https://github.com/Ethan-J-LeeMMV/seng-474-csc-503-loaded-framing-detection)

## 2. Illustrative Sample

```
data/
  lexicon/subjclues.tff          # MPQA Subjectivity Lexicon (word bank)
  raw/SENG474-CSC503_dataset.csv # labeled headline–body pairs
src/
  main.py                        # training pipeline entry point
  method/
    framing/features.py          # feature extraction φ(h, a)
output/                          # created by main.py (gitignored)
```

Features (see `method/framing/features.py`):

| Feature | Meaning |
|--------|---------|
| `headline_charge` | Share of headline tokens whose lemmas appear in the subjectivity lexicon |
| `headline_attribution` | 1 if the headline has an attribution cue or quotation mark, else 0 |
| `body_evidence` | Grounding rate in the body (numbers, named entities, attribution cues) |
| `tone_gap` | `headline_charge` minus body charge |

The classifier is class-weighted, L2-regularized logistic regression on standardized features.

## Git Clone: 

git clone https://github.com/Ethan-J-LeeMMV/seng-474-csc-503-loaded-framing-detection.git

cd seng-474-csc-503-loaded-framing-detection

## 3. Prerequisite (Requirements)

- Python 3.11 or 3.12 (3.12 tested locally)
- spaCy English model **`en_core_web_sm`** (installed separately; see below)

## 4. Setup/Installation Guide

### Option A — pip and venv (recommended)

```powershell
cd seng-474-csc-503-loaded-framing-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On macOS/Linux, use `source .venv/bin/activate` instead of `Activate.ps1`.

### Option B — Conda (optional)

`environment.yml` is **not required** if you use Option A. It is provided for reviewers who prefer Conda/Mamba:

```powershell
conda env create -f environment.yml
conda activate seng474-framing
python -m spacy download en_core_web_sm
```

## 5. Quick Start/Run training

From the **repository root** (with the virtual environment activated):

```powershell
python src/main.py `
  --data data/raw/SENG474-CSC503_dataset.csv `
  --lexicon data/lexicon/subjclues.tff `
  --headline-col title `
  --label-col bias `
  --out output
```

The raw CSV uses columns `title`, `body`, and `bias` (`0` = not biased, `1` = biased). Defaults `--headline-col headline`, `--body-col body`, and `--label-col label` apply if your file already uses those names.

### Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `--C` | `1.0` | Inverse L2 regularization strength |
| `--threshold` | `0.5` | Decision threshold for label 1 |
| `--seed` | `0` | Random seed |
| `--include-weak` | off | Include MPQA `weaksubj` entries in the lexicon |
| `--predict-headline` / `--predict-body` | — | Score one new pair after training (stdout only) |

Re-running with the same `--out` **overwrites** `weights.json` and `predictions.csv` in that folder.

## 6. Outputs

| File | Contents |
|------|----------|
| `output/weights.json` | Feature names, coefficients, intercept, scaler, run metadata |
| `output/predictions.csv` | Per-row `id`, true `label`, `train_prob` (P(biased)), `train_pred`, `headline` |

`predictions.csv` is **in-sample** (training data only). Held-out evaluation is planned for a later milestone.

## 7. Lexicon/Citation

The word bank is the [MPQA Subjectivity Lexicon](https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/) (`subjclues.tff`). By default only **strongsubj** single-word clues are used. Cite Wilson, Wiebe, and Hoffmann (2005), *HLT/EMNLP*, when reporting results.

## 8. LICENSE 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 


## Smoke test (features only)

```powershell
python src/method/framing/features.py
```

Prints feature values for a small toy headline/body pair (does not use project CSV data).
