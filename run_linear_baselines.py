"""Phase 3 — 선형·생성적 베이스라인 (6개 모델 × 하이퍼파라미터 그리드 × 5-fold CV).

전처리: StandardScaler + SMOTE (LR/LDA/QDA) | StandardScaler + class_weight (SVM)
결과:   results/tables/baseline_results.csv
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import pandas as pd

from utils import set_seed
from data import load_raw, get_fold, generate_splits
from preprocess import get_feature_cols, fit_transform_fold
from evaluate import compute_metrics, aggregate_fold_metrics

from models.linear import (
    LR_L1_GRID, LR_L2_GRID, LDA_GRID, QDA_GRID,
    make_lr, make_lda, make_qda,
)
from models.svm import SVM_LINEAR_GRID, SVM_RBF_GRID, make_svm_linear, make_svm_rbf

set_seed(42)

TABLES_DIR = Path('results/tables')
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values
generate_splits(X, y)

N_FOLDS = 5
print(f"Data: {X.shape}, fail%={y.mean()*100:.2f}%\n")

# ── Generic fold runner ───────────────────────────────────────────────────
def run_folds(make_clf_fn, scaler='standard', resample='smote'):
    """Run 5-fold CV, return list of per-fold metric dicts."""
    fold_results = []
    for fold_i in range(N_FOLDS):
        X_tr, X_va, y_tr, y_va = get_fold(fold_i, X, y)
        X_tr_s, X_va_s, y_tr_r = fit_transform_fold(
            X_tr, X_va, y_tr, scaler, resample
        )
        clf = make_clf_fn()
        clf.fit(X_tr_s, y_tr_r)
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_va_s)[:, 1]
        else:
            y_prob = clf.decision_function(X_va_s)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        fold_results.append(compute_metrics(y_va, y_prob))
    return fold_results


# ── Model families ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    # (family_tag, param_grid, make_fn, scaler, resample)
    ("LR_L1",      LR_L1_GRID,      lambda p: make_lr(p),           "standard", "smote"),
    ("LR_L2",      LR_L2_GRID,      lambda p: make_lr(p),           "standard", "smote"),
    ("LDA",        LDA_GRID,        lambda p: make_lda(p),          "standard", "smote"),
    ("QDA",        QDA_GRID,        lambda p: make_qda(p),          "standard", "smote"),
    ("SVM_linear", SVM_LINEAR_GRID, lambda p: make_svm_linear(p),   "standard", "none"),  # class_weight inside model
    ("SVM_RBF",    SVM_RBF_GRID,    lambda p: make_svm_rbf(p),      "standard", "none"),  # class_weight inside model
]

all_rows = []
done = 0

for family, grid, make_fn, scaler, resample in EXPERIMENTS:
    t_family = time.time()
    family_rows = []

    for params in grid:
        t0 = time.time()
        try:
            # Need closure over params
            p_copy = dict(params)
            results = run_folds(lambda p=p_copy: make_fn(p), scaler=scaler, resample=resample)
            agg = aggregate_fold_metrics(results)
        except Exception as e:
            print(f"  ERROR {family} {params}: {e}")
            agg = {k: float('nan') for k in ['roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std','f1_mean','f1_std']}

        row = {"family": family, "scaler": scaler, "resample": resample, **params, **agg}
        family_rows.append(row)
        done += 1
        elapsed = time.time() - t0
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{done:3d}] {family:10s} {param_str:25s} | "
              f"ROC={agg.get('roc_auc_mean', float('nan')):.4f}+/-{agg.get('roc_auc_std', float('nan')):.4f}  "
              f"PR={agg.get('pr_auc_mean', float('nan')):.4f}+/-{agg.get('pr_auc_std', float('nan')):.4f}  "
              f"({elapsed:.1f}s)")

    # Keep best per family (by ROC-AUC)
    family_df = pd.DataFrame(family_rows)
    family_df['is_best'] = family_df['roc_auc_mean'] == family_df['roc_auc_mean'].max()
    all_rows.extend(family_df.to_dict('records'))
    print(f"  --> {family} done in {time.time()-t_family:.1f}s\n")

# ── Save full grid results ─────────────────────────────────────────────────
full_df = pd.DataFrame(all_rows)
full_df.to_csv(TABLES_DIR / 'baseline_grid_full.csv', index=False)

# ── Best-per-family summary ────────────────────────────────────────────────
best_rows = full_df[full_df['is_best'] == True].copy()

# Guidebook reference column (from KAMP guidebook §2.3 — DNN/SVM/AE values)
GUIDEBOOK_REF = {
    "LR_L1":      {"guidebook_roc_auc": "N/A", "guidebook_pr_auc": "N/A"},
    "LR_L2":      {"guidebook_roc_auc": "N/A", "guidebook_pr_auc": "N/A"},
    "LDA":        {"guidebook_roc_auc": "N/A", "guidebook_pr_auc": "N/A"},
    "QDA":        {"guidebook_roc_auc": "N/A", "guidebook_pr_auc": "N/A"},
    "SVM_linear": {"guidebook_roc_auc": "N/A (guidebook SVM=best per §2.3)", "guidebook_pr_auc": "N/A"},
    "SVM_RBF":    {"guidebook_roc_auc": "N/A (guidebook SVM=best per §2.3)", "guidebook_pr_auc": "N/A"},
}

summary_rows = []
for fam in ["LR_L1", "LR_L2", "LDA", "QDA", "SVM_linear", "SVM_RBF"]:
    subset = best_rows[best_rows['family'] == fam]
    if len(subset) == 0:
        continue
    r = subset.iloc[0]
    summary_rows.append({
        "model": fam,
        "best_params": {k: v for k, v in r.items() if k not in
                        {'family','scaler','resample','is_best'} and
                        not k.endswith(('_mean','_std'))},
        "roc_auc": f"{r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}",
        "pr_auc":  f"{r['pr_auc_mean']:.4f}+/-{r['pr_auc_std']:.4f}",
        "f1":      f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}",
        "roc_auc_mean": r['roc_auc_mean'],
        "roc_auc_std":  r['roc_auc_std'],
        "pr_auc_mean":  r['pr_auc_mean'],
        "pr_auc_std":   r['pr_auc_std'],
        "f1_mean":      r['f1_mean'],
        "f1_std":       r['f1_std'],
        **GUIDEBOOK_REF.get(fam, {}),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES_DIR / 'baseline_results.csv', index=False)

print("\n=== Phase 3 베이스라인 결과 (모델별 best, ROC-AUC 정렬) ===")
print(summary_df[['model','roc_auc','pr_auc','f1']].sort_values('roc_auc_mean', ascending=False).to_string(index=False))

best_model = summary_df.loc[summary_df['roc_auc_mean'].idxmax(), 'model']
best_roc   = summary_df['roc_auc_mean'].max()
best_pr    = summary_df.loc[summary_df['roc_auc_mean'].idxmax(), 'pr_auc_mean']
print(f"\nBest overall: {best_model}  ROC-AUC={best_roc:.4f}  PR-AUC={best_pr:.4f}")
print(f"Saved: results/tables/baseline_results.csv")
print(f"       results/tables/baseline_grid_full.csv")
