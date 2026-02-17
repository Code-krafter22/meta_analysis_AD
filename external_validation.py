#!/usr/bin/env python3
"""External validation for AD/control classification using expression + metadata files.

Expected files
--------------
1) Expression CSV (--input): rows are genes, columns are samples.
   - One column must contain gene identifiers (default: first column, or --gene-col).
2) Metadata CSV (--metadata): one row per sample with diagnosis labels.
   - Must contain diagnosis column (default: diagnosis)
   - Must contain sample identifier column (default: sample_id)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldMetrics:
    auc: float
    auprc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    mcc: float
    brier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External validation using expression matrix + metadata diagnosis labels"
    )
    parser.add_argument("--input", required=True, help="Expression CSV (genes x samples)")
    parser.add_argument("--metadata", required=True, help="Metadata CSV with diagnosis labels")
    parser.add_argument(
        "--sample-col",
        default="sample_id",
        help="Sample ID column in metadata (default: sample_id)",
    )
    parser.add_argument(
        "--diagnosis-col",
        default="diagnosis",
        help="Diagnosis column in metadata (default: diagnosis)",
    )
    parser.add_argument(
        "--positive-label",
        default="Ad",
        help="Positive diagnosis label (default: Ad)",
    )
    parser.add_argument(
        "--gene-col",
        default=None,
        help="Gene ID column in expression CSV (default: first column)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for predicted probability",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Stratified CV folds")
    parser.add_argument("--n-repeats", type=int, default=20, help="CV repeats")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional output CSV path for out-of-fold predictions",
    )
    return parser.parse_args()


def load_expression(input_path: str, gene_col: str | None) -> pd.DataFrame:
    expr = pd.read_csv(input_path)
    if expr.empty:
        raise ValueError("Expression file is empty")

    gene_col_name = gene_col if gene_col is not None else expr.columns[0]
    if gene_col_name not in expr.columns:
        raise ValueError(f"Gene column '{gene_col_name}' not found in expression file")

    expr = expr.set_index(gene_col_name)
    if expr.index.duplicated().any():
        raise ValueError("Duplicate gene IDs found in expression file")

    # Convert all sample columns to numeric.
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="any")
    if expr.empty:
        raise ValueError("No numeric gene rows remain after parsing expression matrix")

    # Required orientation for sklearn: samples x genes.
    x = expr.T
    x.index = x.index.astype(str)
    return x


def load_metadata(metadata_path: str, sample_col: str, diagnosis_col: str) -> pd.DataFrame:
    md = pd.read_csv(metadata_path)
    missing = [c for c in [sample_col, diagnosis_col] if c not in md.columns]
    if missing:
        raise ValueError("Missing metadata columns: " + ", ".join(missing))

    md = md[[sample_col, diagnosis_col]].copy()
    md[sample_col] = md[sample_col].astype(str)
    if md[sample_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs found in metadata column '{sample_col}'")
    return md


def build_model(seed: int) -> Pipeline:
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

    out = {}
    for k, values in stats.items():
        if not values:
            out[k] = (np.nan, np.nan)
        else:
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[k] = (float(lo), float(hi))
    return out


def summarize(name: str, values: List[float]) -> str:
    arr = np.array(values, dtype=float)
    return f"{name}: {np.nanmean(arr):.3f} ± {np.nanstd(arr):.3f}"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    if len(frac_pos) == 0:
        return np.nan
    return float(np.mean(np.abs(frac_pos - mean_pred)))


def run_validation(args: argparse.Namespace) -> None:
    x = load_expression(args.input, args.gene_col)
    md = load_metadata(args.metadata, args.sample_col, args.diagnosis_col)

    merged = md.merge(x, left_on=args.sample_col, right_index=True, how="inner")
    if merged.empty:
        raise ValueError("No overlapping sample IDs between metadata and expression matrix")

    y_raw = merged[args.diagnosis_col].astype(str).str.strip().str.lower()
    positive = str(args.positive_label).strip().lower()
    y = (y_raw == positive).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Both classes are required after merging metadata/expression")

    X = merged.drop(columns=[args.sample_col, args.diagnosis_col])

    model = build_model(args.seed)
    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed
    )

    per_fold: List[FoldMetrics] = []
    pooled_prob = np.full(len(y), np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        pooled_prob[test_idx] = y_prob
        per_fold.append(metrics_from_predictions(y_test, y_prob, args.threshold))

        if fold_idx % args.n_splits == 0:
            print(f"Processed fold {fold_idx}/{args.n_splits * args.n_repeats}")

    valid_mask = np.isfinite(pooled_prob)
    y_true_oof = y[valid_mask]
    y_prob_oof = pooled_prob[valid_mask]

    ci = bootstrap_ci(y_true_oof, y_prob_oof, args.threshold, args.bootstrap, args.seed)

    print("\n=== Repeated Stratified CV performance (mean ± SD) ===")
    print(summarize("AUROC", [m.auc for m in per_fold]))
    print(summarize("AUPRC", [m.auprc for m in per_fold]))
    print(summarize("Balanced Accuracy", [m.balanced_accuracy for m in per_fold]))
    print(summarize("Sensitivity", [m.sensitivity for m in per_fold]))
    print(summarize("Specificity", [m.specificity for m in per_fold]))
    print(summarize("MCC", [m.mcc for m in per_fold]))
    print(summarize("Brier", [m.brier for m in per_fold]))

    print("\n=== Bootstrapped 95% CI (pooled OOF predictions) ===")
    for k, (lo, hi) in ci.items():
        print(f"{k}: [{lo:.3f}, {hi:.3f}]")

    precision, _, _ = precision_recall_curve(y_true_oof, y_prob_oof)
    ece = expected_calibration_error(y_true_oof, y_prob_oof, bins=10)
    print(f"\nPR curve points: {len(precision)}")
    print(f"Expected calibration error (10-bin): {ece:.3f}")

    if args.predictions_out:
        out = merged[[args.sample_col, args.diagnosis_col]].copy()
        out["y_true"] = y
        out["y_prob_oof"] = pooled_prob
        out["y_pred_oof"] = (pooled_prob >= args.threshold).astype(int)
        Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.predictions_out, index=False)
        print(f"Saved out-of-fold predictions to: {args.predictions_out}")


def main() -> None:
    args = parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
