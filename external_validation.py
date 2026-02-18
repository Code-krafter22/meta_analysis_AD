#!/usr/bin/env python3
"""
External validation for a fixed gene signature with class-imbalance-aware evaluation.

NO CLI INPUTS: edit the CONFIG section below.

This script supports:
1) Fixed weighted score from a weights file (Excel: gene, weight).
2) Class-weighted logistic regression in repeated stratified CV.

It will:
- Build a samples × genes matrix by merging:
    (a) expression gene×sample CSV (with a gene-symbol column)
    (b) metadata CSV (sample_id + diagnosis/label)
- Run repeated stratified CV
- Print mean ± SD metrics across folds
- Bootstrap 95% CIs on pooled out-of-fold predictions
- Save per-sample out-of-fold predictions as CSV
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG (EDIT THESE ONCE)
# ============================================================

# --- Input files you said you have ---
EXPR_CSV = Path(r"data/GSE125583/DE_data/GSE125583_log2cpm_SELECTED_GENES.csv")      # gene×sample (gene symbol col included)
META_CSV = Path(r"data/GSE125583/DE_data/metadata_200samples.csv")                   # sample metadata

# OPTIONAL weights for fixed-score mode (Excel)
WEIGHTS_XLSX = Path(r"data/GSE184942/meta-analysis-weights-per-gene.xlsx")   # columns like: gene, weight
USE_FIXED_SCORE = True  # True = use weights to compute signature score; False = fit LR in repeated CV

# --- Metadata column names ---
SAMPLE_COL = "geo_accession"
LABEL_COL = "diagnosis:ch1"
POSITIVE_LABEL = "Alzheimer's disease"

# --- Expression file gene symbol column ---
# If you know the exact name, set it. Otherwise keep None to use "last column".
GENE_COL_NAME: Optional[str] = "Gene"  # e.g., "gene" or "symbol" or "Unnamed: 11" ; None = last column

# --- CV / evaluation settings ---
THRESHOLD = 0.5
N_SPLITS = 5
N_REPEATS = 20
BOOTSTRAP = 1000
SEED = 42

# --- Outputs ---
OUTDIR = Path("results")
PREDICTIONS_OUT = OUTDIR / "oof_predictions.csv"
WEIGHTS_JSON_OUT = OUTDIR / "weights_used.json"  # saves the exact weights used (after filtering/intersection)


# ============================================================
# Core code
# ============================================================

@dataclass
class FoldMetrics:
    auc: float
    auprc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    mcc: float
    brier: float


def build_model(seed: int) -> Pipeline:
    # class_weight='balanced' handles imbalance during fitting
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fixed_score(X: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    raw = np.zeros(X.shape[0], dtype=float)
    for g, w in weights.items():
        raw += X[g].to_numpy(dtype=float) * float(w)
    # sigmoid -> pseudo-probability
    return 1.0 / (1.0 + np.exp(-raw))


def metrics_from_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> FoldMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return FoldMetrics(
        auc=roc_auc_score(y_true, y_prob),
        auprc=average_precision_score(y_true, y_prob),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        sensitivity=sensitivity,
        specificity=specificity,
        mcc=matthews_corrcoef(y_true, y_pred),
        brier=brier_score_loss(y_true, y_prob),
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {k: [] for k in ["auc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "mcc", "brier"]}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue

        m = metrics_from_predictions(yt, yp, threshold)
        stats["auc"].append(m.auc)
        stats["auprc"].append(m.auprc)
        stats["balanced_accuracy"].append(m.balanced_accuracy)
        stats["sensitivity"].append(m.sensitivity)
        stats["specificity"].append(m.specificity)
        stats["mcc"].append(m.mcc)
        stats["brier"].append(m.brier)

    ci = {}
    for k, values in stats.items():
        if not values:
            ci[k] = (np.nan, np.nan)
        else:
            lo, hi = np.percentile(values, [2.5, 97.5])
            ci[k] = (float(lo), float(hi))
    return ci


def summarize(name: str, values: List[float]) -> str:
    arr = np.array(values, dtype=float)
    return f"{name}: {np.nanmean(arr):.3f} ± {np.nanstd(arr):.3f}"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    if len(frac_pos) == 0:
        return np.nan
    return float(np.mean(np.abs(frac_pos - mean_pred)))


def load_weights_from_xlsx(xlsx_path: Path) -> Dict[str, float]:
    w = pd.read_excel(xlsx_path)
    # try common column names
    cols_lower = {c.lower(): c for c in w.columns}
    gene_col = cols_lower.get("gene") or cols_lower.get("symbol") or list(w.columns)[0]
    weight_col = cols_lower.get("weight") or cols_lower.get("coef") or cols_lower.get("beta") or list(w.columns)[1]

    w = w[[gene_col, weight_col]].dropna()
    w.columns = ["gene", "weight"]
    w["gene"] = w["gene"].astype(str)
    w["weight"] = w["weight"].astype(float)

    return dict(zip(w["gene"], w["weight"]))


def build_samples_x_genes(expr_csv: Path, meta_csv: Path) -> pd.DataFrame:
    expr = pd.read_csv(expr_csv)
    meta = pd.read_csv(meta_csv)

    # pick gene symbol column
    gene_col = GENE_COL_NAME if GENE_COL_NAME is not None else expr.columns[-1]

    if gene_col not in expr.columns:
        raise ValueError(f"GENE_COL_NAME='{GENE_COL_NAME}' not found in expression CSV columns.")

    if SAMPLE_COL not in meta.columns:
        raise ValueError(f"SAMPLE_COL='{SAMPLE_COL}' not found in metadata CSV columns.")
    if LABEL_COL not in meta.columns:
        raise ValueError(f"LABEL_COL='{LABEL_COL}' not found in metadata CSV columns.")

    # sample columns = all columns except gene_col and (often) first column (ensembl/id)
    exclude = {gene_col}
    first_col = expr.columns[0]
    if first_col != gene_col and (
        first_col.lower() in {"id", "index", "ensembl", "ensg"}
        or expr[first_col].dtype == object
    ):
        exclude.add(first_col)
    sample_cols = [c for c in expr.columns if c not in exclude]

    gx = expr[[gene_col] + sample_cols].dropna(subset=[gene_col]).copy()
    gx[gene_col] = gx[gene_col].astype(str)

    # collapse duplicated symbols
    gx = gx.groupby(gene_col, as_index=True)[sample_cols].mean()

    # transpose -> samples × genes
    sxg = gx.T
    sxg.index.name = SAMPLE_COL
    sxg = sxg.reset_index()

    # merge labels
    df = meta[[SAMPLE_COL, LABEL_COL]].merge(sxg, on=SAMPLE_COL, how="inner")

    if df.empty:
        raise ValueError(
            "After merging metadata and expression, got 0 rows. "
            "Check that metadata sample IDs match the expression sample column names."
        )

    return df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = build_samples_x_genes(EXPR_CSV, META_CSV)

    # create y
    y = (df[LABEL_COL].astype(str) == str(POSITIVE_LABEL)).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Both classes are required for validation (need positives and negatives).")

    # load weights and gene list (if fixed score)
    weights: Optional[Dict[str, float]] = None
    genes: List[str]

    if USE_FIXED_SCORE:
        weights = load_weights_from_xlsx(WEIGHTS_XLSX)
        # keep only genes present in df
        genes = [g for g in weights.keys() if g in df.columns]
        if len(genes) == 0:
            raise ValueError("No overlap between weight genes and expression columns.")
        # filter weights to used genes
        weights = {g: float(weights[g]) for g in genes}

        # save the weights actually used
        WEIGHTS_JSON_OUT.write_text(json.dumps(weights, indent=2))
        print(f"Using fixed-score mode with {len(genes)} genes. Saved weights: {WEIGHTS_JSON_OUT}")
    else:
        # if LR mode, use all gene columns except metadata columns
        genes = [c for c in df.columns if c not in {SAMPLE_COL, LABEL_COL}]
        print(f"Using LR mode with {len(genes)} genes.")

    X = df.loc[:, genes].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Expression matrix contains non-numeric values after coercion.")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    per_fold: List[FoldMetrics] = []
    
    model = None if USE_FIXED_SCORE else build_model(SEED)

    # Replace pooled_prob initialisation and accumulation
    pooled_prob = np.zeros(len(y), dtype=float)
    pooled_count = np.zeros(len(y), dtype=int)
    total_folds = N_SPLITS * N_REPEATS
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if USE_FIXED_SCORE:
            assert weights is not None
            y_prob = fixed_score(X_test, weights)
        else:
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_prob = fold_model.predict_proba(X_test)[:, 1]

        # Inside the loop, replace the assignment line with:
        pooled_prob[test_idx] += y_prob
        pooled_count[test_idx] += 1
        per_fold.append(metrics_from_predictions(y_test, y_prob, THRESHOLD))

        if fold_idx % N_SPLITS == 0:
            print(f"Processed fold {fold_idx}/{total_folds}")

    valid_mask = pooled_count > 0
    y_true_oof = y[valid_mask]
    y_prob_oof = (pooled_prob[valid_mask] / pooled_count[valid_mask])  # average across repeats

    # ── NEW: find optimal threshold from ROC before anything else ──────
    
    fpr, tpr, thresholds = roc_curve(y_true_oof, y_prob_oof)
    balanced_accs = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(balanced_accs)
    best_threshold = float(thresholds[best_idx])

    print(f"\nOptimal threshold (max balanced accuracy): {best_threshold:.3f}")
    print(f"At this threshold — Sensitivity: {tpr[best_idx]:.3f}, Specificity: {1-fpr[best_idx]:.3f}")
    # ───────────────────────────────────────────────────────────────────

    ci = bootstrap_ci(
        y_true=y_true_oof,
        y_prob=y_prob_oof,
        threshold=best_threshold,
        n_bootstrap=BOOTSTRAP,
        seed=SEED,
    )

    m_best = metrics_from_predictions(y_true_oof, y_prob_oof, best_threshold)
    print("\n=== Pooled OOF metrics @ optimal threshold ===")
    print(f"Balanced Accuracy: {m_best.balanced_accuracy:.3f}")
    print(f"Sensitivity:       {m_best.sensitivity:.3f}")
    print(f"Specificity:       {m_best.specificity:.3f}")
    print(f"MCC:               {m_best.mcc:.3f}")
    print(f"Brier:             {m_best.brier:.3f}")

    print("\n=== Repeated Stratified CV performance (mean ± SD) ===")
    print(summarize("AUROC", [m.auc for m in per_fold]))
    print(summarize("AUPRC", [m.auprc for m in per_fold]))
    print(summarize("Balanced Accuracy", [m.balanced_accuracy for m in per_fold]))
    print(summarize("Sensitivity", [m.sensitivity for m in per_fold]))
    print(summarize("Specificity", [m.specificity for m in per_fold]))
    print(summarize("MCC", [m.mcc for m in per_fold]))
    print(summarize("Brier", [m.brier for m in per_fold]))

    print("\n=== Bootstrapped 95% CI (out-of-fold pooled predictions) ===")
    for k, (lo, hi) in ci.items():
        print(f"{k}: [{lo:.3f}, {hi:.3f}]")

    # ── NEW: no-skill baseline ─────────────────────────────────────────
    prevalence = y_true_oof.mean()
    oof_auprc = average_precision_score(y_true_oof, y_prob_oof)
    print(f"\nClass prevalence (AD): {prevalence:.3f}")
    print(f"No-skill AUPRC baseline: {prevalence:.3f}")
    print(f"Signature AUPRC lift over baseline: {oof_auprc - prevalence:.3f}")
    # ──────────────────────────────────────────────────────────────────

    precision, recall, _ = precision_recall_curve(y_true_oof, y_prob_oof)
    ece = expected_calibration_error(y_true_oof, y_prob_oof, bins=10)
    print(f"\nPR curve points: {len(precision)}")
    print(f"Expected calibration error (10-bin): {ece:.3f}")

    #  the calibrator is fitted on the same OOF data used to evaluate it, which is slightly optimistic
    # ── NEW: Platt-scaled calibration ─────────────────────────────────
    from sklearn.linear_model import LogisticRegression as PlattLR

    calibrator = PlattLR()
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_c, test_c = next(sss.split(y_prob_oof.reshape(-1,1), y_true_oof))

    calibrator = PlattLR()
    calibrator.fit(y_prob_oof[train_c].reshape(-1,1), y_true_oof[train_c])
    y_cal = calibrator.predict_proba(y_prob_oof[test_c].reshape(-1,1))[:,1]

    print("Calibration eval on held-out OOF split:")
    print("Brier:", brier_score_loss(y_true_oof[test_c], y_cal))
    print("ECE:", expected_calibration_error(y_true_oof[test_c], y_cal))

    # ──────────────────────────────────────────────────────────────────

    # save predictions
    out = df[[SAMPLE_COL, LABEL_COL]].copy()
    out["y_true"] = y
    oof_prob = np.full(len(y), np.nan)
    mask = pooled_count > 0
    oof_prob[mask] = pooled_prob[mask] / pooled_count[mask]

    out["y_prob_oof"] = oof_prob
    out["y_pred_oof"] = (oof_prob >= best_threshold).astype(int)

    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_OUT, index=False)
    print(f"\nSaved out-of-fold predictions to: {PREDICTIONS_OUT.resolve()}")


if __name__ == "__main__":
    main()
