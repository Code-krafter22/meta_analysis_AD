#!/usr/bin/env python3
"""
Multiple Evaluation Methods for 37-Gene Signature
==================================================

This script provides 5 different evaluation approaches for your 37-gene signature.
Simply change the METHOD variable to try each one.

Methods:
1. Simple Train/Test Split (80/20)
2. Single Cross-Validation (5-fold)
3. Leave-One-Out Cross-Validation (LOOCV)
4. Stratified K-Fold with Fixed Parameters
5. Full Cohort Training (training performance only)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - CHANGE THIS TO SWITCH METHODS
# ============================================================================
METHOD = 5  # Change this to: 1, 2, 3, 4, or 5

# Elastic-net parameters (used in methods 1, 3, 4, 5)
L1_RATIO = 0.5  # 0.0 = Ridge, 1.0 = Lasso, 0.5 = Elastic-net
C_VALUE = 1.0   # Inverse regularization strength

# Cross-validation folds (used in methods 2, 4)
N_FOLDS = 5

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================

METHOD_NAMES = {
    1: "Train/Test Split (80/20)",
    2: "Single Cross-Validation (5-fold)",
    3: "Leave-One-Out Cross-Validation (LOOCV)",
    4: "Stratified K-Fold with Fixed Parameters",
    5: "Full Cohort Training"
}

print("=" * 80)
print(f"EVALUATION METHOD: {METHOD_NAMES[METHOD]}")
print("=" * 80)
print()

# Load the 37-gene signature
genes_37 = [
    "ADAM33", "AEBP1", "CCDC102A", "CLDN9", "GFAP", "HSPB1", "HSPB7", "KANK2", 
    "KLF15", "MRGPRF", "NUPR1", "PIK3R5", "PRELP", "PRX", "TCEA3", "TMPRSS5", 
    "CHML", "ELOVL4", "GAD1", "GAD2", "HPRT1", "ITFG1", "MAS1", "NAP1L5", 
    "NCALD", "NEUROD6", "NRN1", "OPN3", "RAB3B", "RAB3C", "RGS4", "RPH3A", 
    "SCG2", "SERPINI1", "STAT4", "TRIM36"
]

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("1. Loading data...")
print("-" * 80)

try:
    expr_data = pd.read_csv('data/GSE125583/DE_data/GSE125583_log2cpm_SELECTED_GENES.csv', index_col=0)
    print(f"✓ Loaded expression data: {expr_data.shape}")
    print(f"  Index (rows): {expr_data.shape[0]} - appears to be: ", end="")
    print("GENES" if expr_data.shape[0] < 100 else "SAMPLES")
    print(f"  Columns: {expr_data.shape[1]} - appears to be: ", end="")
    print("SAMPLES" if expr_data.shape[0] < 100 else "GENES")
    
    # Auto-detect if we need to transpose
    # If rows < 100 and columns > 100, likely genes are rows (needs transpose)
    if expr_data.shape[0] < 100 and expr_data.shape[1] > 100:
        print(f"\n⚠️  Expression data appears to be transposed (genes as rows)")
        print(f"  Transposing to samples × genes format...")
        expr_data = expr_data.T
        print(f"  After transpose: {expr_data.shape}")
    
except FileNotFoundError:
    print("ERROR: 'external_cohort_expression.csv' not found")
    exit(1)

try:
    metadata = pd.read_csv('data/GSE125583/DE_data/metadata_200samples.csv', index_col=0)
    print(f"✓ Loaded metadata: {metadata.shape}")
except FileNotFoundError:
    print("ERROR: 'data/GSE125583/DE_data/metadata_200samples.csv' not found")
    exit(1)

print(f"✓ Loaded 37-gene signature: {len(genes_37)} genes")
print()

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("2. Preparing data...")
print("-" * 80)

# CRITICAL: Align samples first!
print("Aligning samples between expression and metadata...")
common_samples = expr_data.index.intersection(metadata.index)
print(f"  Expression samples: {len(expr_data)}")
print(f"  Metadata samples: {len(metadata)}")
print(f"  Common samples: {len(common_samples)}")

if len(common_samples) == 0:
    print()
    print("ERROR: No common samples found between expression and metadata!")
    print("Sample ID examples:")
    print(f"  Expression: {list(expr_data.index[:5])}")
    print(f"  Metadata:   {list(metadata.index[:5])}")
    print()
    print("Please check that sample IDs match exactly between files.")
    exit(1)

# Keep only common samples
expr_data = expr_data.loc[common_samples]
metadata = metadata.loc[common_samples]
print(f"✓ Aligned to {len(common_samples)} common samples")
print()

# Filter for 37 genes
available_genes = [g for g in genes_37 if g in expr_data.columns]
missing_genes = [g for g in genes_37 if g not in expr_data.columns]

print(f"Available genes: {len(available_genes)}/{len(genes_37)}")
if missing_genes:
    print(f"Missing genes: {len(missing_genes)}")
    print(f"  {', '.join(missing_genes[:5])}{'...' if len(missing_genes) > 5 else ''}")

X = expr_data[available_genes].copy()

# Prepare labels
diagnosis_col = 'diagnosis:ch1' if 'diagnosis:ch1' in metadata.columns else metadata.columns[0]
print(f"Using diagnosis column: '{diagnosis_col}'")
y = metadata[diagnosis_col].copy()

# Map to binary labels
if y.dtype == 'object' or isinstance(y.iloc[0], str):
    unique_labels = y.unique()
    print(f"Unique labels: {unique_labels}")
    
    label_map = {}
    for label in unique_labels:
        label_str = str(label)
        label_lower = label_str.lower()
        
        # Check for AD/Alzheimer patterns (case-insensitive)
        if ("alzheimer's disease" in label_lower or 
            'alzheimer' in label_lower or 
            'ad' == label_lower or
            'disease' in label_lower):
            label_map[label] = 1
        # Check for control patterns
        elif ('control' in label_lower or 
              'normal' in label_lower or 
              'cn' == label_lower):
            label_map[label] = 0
        else:
            # Unknown label
            label_map[label] = None
            print(f"  ⚠️  Warning: Unknown label '{label}' will be set to None")
    
    y = y.map(label_map)
    print(f"Label mapping: {label_map}")
    
    # Check for unmapped labels
    if y.isna().any():
        n_unmapped = y.isna().sum()
        print(f"  ⚠️  Warning: {n_unmapped} samples have unmapped labels (will be removed)")
else:
    y = (y > 0).astype(int)

# Remove NaN
valid_idx = ~(X.isna().any(axis=1) | y.isna())
n_removed = (~valid_idx).sum()
if n_removed > 0:
    print(f"Removing {n_removed} samples with NaN values")
X = X[valid_idx]
y = y[valid_idx]

# Final sanity check
assert X.shape[0] == y.shape[0], f"SHAPE MISMATCH: X has {X.shape[0]} samples, y has {y.shape[0]}"

print(f"\nFinal dataset:")
print(f"  Samples: {X.shape[0]}")
print(f"  Genes: {X.shape[1]}")
print(f"  AD cases: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
print(f"  Controls: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print()

# ============================================================================
# 3. RUN SELECTED METHOD
# ============================================================================
print(f"3. Running: {METHOD_NAMES[METHOD]}")
print("-" * 80)

def create_pipeline():
    """Create a standard pipeline with scaling and elastic-net"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=L1_RATIO,
            C=C_VALUE,
            class_weight='balanced',
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ])

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all metrics"""
    return {
        'auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score: {metrics['f1']:.4f}")

# Storage for results
all_y_true = []
all_y_pred = []
all_y_proba = []
fold_results = []
trained_model = None
feature_coefs = []

# ============================================================================
# METHOD 1: TRAIN/TEST SPLIT
# ============================================================================
if METHOD == 1:
    print(f"Splitting data: 80% train, 20% test")
    print(f"Parameters: L1_ratio={L1_RATIO}, C={C_VALUE}")
    print()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"  AD: {(y_train == 1).sum()}, Control: {(y_train == 0).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  AD: {(y_test == 1).sum()}, Control: {(y_test == 0).sum()}")
    print()
    
    # Train
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    trained_model = pipeline
    
    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
    
    print("📊 TRAINING PERFORMANCE:")
    print_metrics(train_metrics, "  ")
    print()
    print("📊 TEST PERFORMANCE:")
    print_metrics(test_metrics, "  ")
    
    all_y_true = y_test.values
    all_y_pred = y_pred_test
    all_y_proba = y_proba_test
    
    feature_coefs.append(pd.Series(
        pipeline.named_steps['classifier'].coef_[0], 
        index=X.columns
    ))

# ============================================================================
# METHOD 2: SINGLE CROSS-VALIDATION
# ============================================================================
elif METHOD == 2:
    print(f"Cross-validation: {N_FOLDS} folds")
    print(f"Using sklearn's cross_validate with automatic hyperparameter tuning")
    print()
    
    from sklearn.linear_model import LogisticRegressionCV
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"Fold {fold_idx}/{N_FOLDS}:", end=" ")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Use LogisticRegressionCV for automatic hyperparameter selection
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=10,
                cv=5,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[L1_RATIO],
                max_iter=10000,
                random_state=RANDOM_STATE,
                scoring='roc_auc',
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        print(f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        fold_results.append(metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        feature_coefs.append(pd.Series(
            pipeline.named_steps['classifier'].coef_[0], 
            index=X.columns
        ))
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    print()
    print("📊 MEAN PERFORMANCE ACROSS FOLDS:")
    results_df = pd.DataFrame(fold_results)
    for metric in ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

# ============================================================================
# METHOD 3: LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# ============================================================================
elif METHOD == 3:
    print(f"Leave-One-Out Cross-Validation")
    print(f"Parameters: L1_ratio={L1_RATIO}, C={C_VALUE}")
    print(f"Total iterations: {len(X)} (one per sample)")
    print()
    
    loo = LeaveOneOut()
    
    for i, (train_idx, test_idx) in enumerate(loo.split(X), 1):
        if i % 20 == 0 or i == len(X):
            print(f"Progress: {i}/{len(X)}", end="\r")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        if i % 50 == 0:  # Save coefficients every 50 iterations
            feature_coefs.append(pd.Series(
                pipeline.named_steps['classifier'].coef_[0], 
                index=X.columns
            ))
    
    print()
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    metrics = calculate_metrics(all_y_true, all_y_pred, all_y_proba)
    print()
    print("📊 LOOCV PERFORMANCE:")
    print_metrics(metrics, "  ")

# ============================================================================
# METHOD 4: STRATIFIED K-FOLD WITH FIXED PARAMETERS
# ============================================================================
elif METHOD == 4:
    print(f"Stratified K-Fold: {N_FOLDS} folds")
    print(f"Fixed parameters: L1_ratio={L1_RATIO}, C={C_VALUE}")
    print()
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"Fold {fold_idx}/{N_FOLDS}:", end=" ")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        print(f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        fold_results.append(metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        feature_coefs.append(pd.Series(
            pipeline.named_steps['classifier'].coef_[0], 
            index=X.columns
        ))
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    print()
    print("📊 MEAN PERFORMANCE ACROSS FOLDS:")
    results_df = pd.DataFrame(fold_results)
    for metric in ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

# ============================================================================
# METHOD 5: FULL COHORT TRAINING
# ============================================================================
elif METHOD == 5:
    print(f"Training on entire cohort")
    print(f"Parameters: L1_ratio={L1_RATIO}, C={C_VALUE}")
    print()
    print("⚠️  WARNING: Training and evaluating on the same data will give")
    print("    optimistically biased results. Use for feature analysis only.")
    print()
    
    pipeline = create_pipeline()
    pipeline.fit(X, y)
    trained_model = pipeline
    
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    
    metrics = calculate_metrics(y, y_pred, y_proba)
    
    print("📊 TRAINING PERFORMANCE (BIASED):")
    print_metrics(metrics, "  ")
    
    all_y_true = y.values
    all_y_pred = y_pred
    all_y_proba = y_proba
    
    feature_coefs.append(pd.Series(
        pipeline.named_steps['classifier'].coef_[0], 
        index=X.columns
    ))

# ============================================================================
# 4. AGGREGATE RESULTS
# ============================================================================
print()
print("=" * 80)
print("AGGREGATE RESULTS")
print("=" * 80)

overall_metrics = calculate_metrics(all_y_true, all_y_pred, all_y_proba)
print()
print("📈 OVERALL PERFORMANCE:")
print_metrics(overall_metrics, "  ")

# Confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"\n📋 Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Control  AD")
print(f"Actual Control    {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"       AD         {cm[1,0]:4d}   {cm[1,1]:4d}")

print("\n📊 Classification Report:")
print(classification_report(all_y_true, all_y_pred, 
                          target_names=['Control', 'AD'],
                          digits=4))

# Statistical test (if applicable)
if METHOD in [2, 4]:  # Methods with multiple folds
    auc_scores = [r['auc'] for r in fold_results]
    t_stat, p_value = stats.ttest_1samp(auc_scores, 0.5)
    print(f"🔬 STATISTICAL SIGNIFICANCE:")
    print(f"  One-sample t-test (H0: AUC = 0.5)")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    if p_value < 0.05:
        print(f"  ✓ Significantly better than random (p < 0.05)")
    else:
        print(f"  ✗ NOT significantly better than random (p ≥ 0.05)")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print()
print("=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

if len(feature_coefs) > 0:
    coef_df = pd.DataFrame(feature_coefs)
    mean_coefs = coef_df.mean().sort_values(key=abs, ascending=False)
    std_coefs = coef_df.std()
    
    print(f"\nTop 15 genes by |coefficient|:")
    print("-" * 60)
    for i, (gene, coef) in enumerate(mean_coefs.head(15).items(), 1):
        if len(feature_coefs) > 1:
            std = std_coefs[gene]
            print(f"{i:2d}. {gene:15s}  {coef:8.4f} ± {std:.4f}")
        else:
            print(f"{i:2d}. {gene:15s}  {coef:8.4f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print()
print("=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(14, 10))

# 1. ROC Curve
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {overall_metrics["auc"]:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=10)
ax1.set_ylabel('True Positive Rate', fontsize=10)
ax1.set_title(f'ROC Curve\n{METHOD_NAMES[METHOD]}', fontsize=11, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# 2. Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['Control', 'AD'], yticklabels=['Control', 'AD'])
ax2.set_ylabel('True Label', fontsize=10)
ax2.set_xlabel('Predicted Label', fontsize=10)
ax2.set_title('Confusion Matrix', fontsize=11, fontweight='bold')

# 3. Performance metrics bar chart
ax3 = plt.subplot(2, 3, 3)
metrics_to_plot = ['auc', 'accuracy', 'balanced_accuracy', 'f1']
metric_values = [overall_metrics[m] for m in metrics_to_plot]
colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple']
bars = ax3.bar(range(len(metrics_to_plot)), metric_values, color=colors, alpha=0.7)
ax3.set_xticks(range(len(metrics_to_plot)))
ax3.set_xticklabels([m.upper().replace('_', '\n') for m in metrics_to_plot], fontsize=9)
ax3.set_ylabel('Score', fontsize=10)
ax3.set_title('Performance Metrics', fontsize=11, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, metric_values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Predicted probability distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(all_y_proba[all_y_true == 0], bins=20, alpha=0.6, label='Control', color='green')
ax4.hist(all_y_proba[all_y_true == 1], bins=20, alpha=0.6, label='AD', color='red')
ax4.set_xlabel('Predicted Probability (AD)', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Predicted Probability Distribution', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# 5. Top features
ax5 = plt.subplot(2, 3, 5)
if len(feature_coefs) > 0:
    top_features = mean_coefs.head(15)
    colors_feat = ['red' if x < 0 else 'steelblue' for x in top_features.values]
    ax5.barh(range(len(top_features)), top_features.values, color=colors_feat, alpha=0.7)
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features.index, fontsize=8)
    ax5.set_xlabel('Coefficient', fontsize=10)
    ax5.set_title('Top 15 Features', fontsize=11, fontweight='bold')
    ax5.axvline(x=0, color='black', linewidth=1)
    ax5.grid(alpha=0.3, axis='x')

# 6. Fold-wise performance (if applicable)
ax6 = plt.subplot(2, 3, 6)
if METHOD in [2, 4] and len(fold_results) > 0:
    fold_aucs = [r['auc'] for r in fold_results]
    x_pos = np.arange(len(fold_aucs))
    ax6.bar(x_pos, fold_aucs, alpha=0.7, color='steelblue')
    ax6.axhline(y=np.mean(fold_aucs), color='green', linestyle='-', 
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(fold_aucs):.3f}')
    ax6.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label='Random')
    ax6.set_xlabel('Fold', fontsize=10)
    ax6.set_ylabel('AUC', fontsize=10)
    ax6.set_title('AUC Across Folds', fontsize=11, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'{i+1}' for i in range(len(fold_aucs))])
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3, axis='y')
    ax6.set_ylim([0, 1])
elif METHOD == 1:
    # Show train vs test for train/test split
    train_test_metrics = ['AUC', 'Accuracy', 'Bal. Acc', 'F1']
    if 'train_metrics' in locals():
        train_vals = [train_metrics['auc'], train_metrics['accuracy'], 
                     train_metrics['balanced_accuracy'], train_metrics['f1']]
        test_vals = [test_metrics['auc'], test_metrics['accuracy'], 
                    test_metrics['balanced_accuracy'], test_metrics['f1']]
        
        x = np.arange(len(train_test_metrics))
        width = 0.35
        ax6.bar(x - width/2, train_vals, width, label='Train', alpha=0.7, color='lightblue')
        ax6.bar(x + width/2, test_vals, width, label='Test', alpha=0.7, color='steelblue')
        ax6.set_xlabel('Metric', fontsize=10)
        ax6.set_ylabel('Score', fontsize=10)
        ax6.set_title('Train vs Test Performance', fontsize=11, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(train_test_metrics, fontsize=9)
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')
        ax6.set_ylim([0, 1])
else:
    ax6.text(0.5, 0.5, 'N/A for this method', 
            ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    ax6.axis('off')

plt.tight_layout()
output_filename = f'results/method_{METHOD}_results.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved: method_{METHOD}_results.png")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': all_y_true,
    'y_pred': all_y_pred,
    'y_proba': all_y_proba
})
predictions_df.to_csv(f'results/method_{METHOD}_predictions.csv', index=False)
print(f"✓ Saved: method_{METHOD}_predictions.csv")

# Save feature coefficients
if len(feature_coefs) > 0:
    coef_output = pd.DataFrame({
        'gene': mean_coefs.index,
        'mean_coefficient': mean_coefs.values,
        'abs_coefficient': np.abs(mean_coefs.values)
    })
    if len(feature_coefs) > 1:
        coef_output['std_coefficient'] = [std_coefs[g] for g in mean_coefs.index]
    coef_output = coef_output.sort_values('abs_coefficient', ascending=False)
    coef_output.to_csv(f'results/method_{METHOD}_coefficients.csv', index=False)
    print(f"✓ Saved: method_{METHOD}_coefficients.csv")

# Save summary
with open(f'results/method_{METHOD}_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write(f"METHOD: {METHOD_NAMES[METHOD]}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET:\n")
    f.write(f"  Total samples: {X.shape[0]}\n")
    f.write(f"  Genes: {X.shape[1]}\n")
    f.write(f"  AD: {(y == 1).sum()}, Control: {(y == 0).sum()}\n\n")
    
    if METHOD in [1, 3, 5] or len(fold_results) == 0:
        f.write("OVERALL PERFORMANCE:\n")
        for metric, value in overall_metrics.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")
    else:
        f.write("PERFORMANCE (Mean ± Std across folds):\n")
        results_df = pd.DataFrame(fold_results)
        for metric in ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            f.write(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    f.write(f"\nTOP 10 GENES:\n")
    for i, (gene, coef) in enumerate(mean_coefs.head(10).items(), 1):
        f.write(f"  {i:2d}. {gene:15s}  {coef:8.4f}\n")

print(f"✓ Saved: method_{METHOD}_summary.txt")

print()
print("=" * 80)
print("COMPLETE!")
print("=" * 80)
print()
print(f"To try a different method, change METHOD = {METHOD} to another value (1-5)")
print("at the top of the script and run again.")
print()