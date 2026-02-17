#!/usr/bin/env python3
"""External validation for a fixed gene signature with class-imbalance-aware evaluation.

Input assumptions:
- CSV has one row per sample.
- A label column contains case/control classes.
- Expression columns are log2CPM values for genes.

The script supports two validation modes:
1) Fixed weighted score from a JSON file mapping gene -> coefficient.
2) Class-weighted logistic regression fitted in repeated stratified CV.

Outputs:
- Console summary of mean +/- SD metrics across repeats.
- Bootstrap 95% CIs for pooled out-of-fold predictions.
- Optional CSV with per-sample out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
        description="External validation of AD gene signature under class imbalance"
    )
    parser.add_argument("--input", required=True, help="CSV with labels + gene expression")
    parser.add_argument("--label-col", required=True, help="Column containing AD/control labels")
    parser.add_argument(
        "--positive-label",
        default="AD",
        help="Value in --label-col treated as positive class (default: AD)",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="List of genes in the fixed signature",
    )
    parser.add_argument(
        "--score-weights",
        default=None,
        help=(
            "Optional JSON file with gene coefficients for a fixed score. "
            "If omitted, script fits class-weighted logistic regression in repeated CV."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on predicted probability/score for class calls",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Stratified CV folds")
    parser.add_argument("--n-repeats", type=int, default=20, help="CV repeats")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations for 95%% CIs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional output CSV path for out-of-fold predictions",
    )
    return parser.parse_args()


def validate_inputs(df: pd.DataFrame, label_col: str, genes: Sequence[str]) -> None:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input file")

    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(
            "Missing genes in input file: " + ", ".join(missing) + ". "
            "Ensure gene symbols match CSV column names exactly."
        )


def build_model(seed: int) -> Pipeline:
    # class_weight='balanced' handles AD/control imbalance during model fitting.
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
    # Linear weighted signature score; sigmoid transforms to pseudo-probability.
    raw = np.zeros(X.shape[0], dtype=float)
    for g, w in weights.items():
        raw += X[g].to_numpy(dtype=float) * float(w)
    return 1.0 / (1.0 + np.exp(-raw))


def metrics_from_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> FoldMetrics:
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
    stats = {
        "auc": [],
        "auprc": [],
        "balanced_accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "mcc": [],
        "brier": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]

        # Skip pathological resamples with only one class.
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
    # Equal weighting across bins produced by quantile strategy.
    return float(np.mean(np.abs(frac_pos - mean_pred)))


def run_validation(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    validate_inputs(df, args.label_col, args.genes)

    X = df.loc[:, args.genes].copy()
    y = (df[args.label_col].astype(str) == str(args.positive_label)).astype(int).to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Both classes are required for validation")

    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed
    )

    per_fold: List[FoldMetrics] = []
    pooled_prob = np.full(len(y), np.nan, dtype=float)

    use_fixed_score = args.score_weights is not None
    weights = None
    model = None
    if use_fixed_score:
        with open(args.score_weights, "r", encoding="utf-8") as f:
            weights = json.load(f)
        missing_weights = [g for g in args.genes if g not in weights]
        if missing_weights:
            raise ValueError(
                "Missing coefficients in weight file for genes: "
                + ", ".join(missing_weights)
            )
    else:
        model = build_model(args.seed)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_fixed_score:
            y_prob = fixed_score(X_test, weights)
        else:
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_prob = fold_model.predict_proba(X_test)[:, 1]

        pooled_prob[test_idx] = y_prob
        m = metrics_from_predictions(y_test, y_prob, args.threshold)
        per_fold.append(m)

        if fold_idx % args.n_splits == 0:
            print(f"Processed fold {fold_idx}/{args.n_splits * args.n_repeats}")

    valid_mask = np.isfinite(pooled_prob)
    y_true_oof = y[valid_mask]
    y_prob_oof = pooled_prob[valid_mask]

    ci = bootstrap_ci(
        y_true=y_true_oof,
        y_prob=y_prob_oof,
        threshold=args.threshold,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

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

    precision, recall, _ = precision_recall_curve(y_true_oof, y_prob_oof)
    ece = expected_calibration_error(y_true_oof, y_prob_oof, bins=10)
    print(f"\nPR curve points: {len(precision)}")
    print(f"Expected calibration error (10-bin): {ece:.3f}")

    if args.predictions_out:
        out = df.copy()
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
