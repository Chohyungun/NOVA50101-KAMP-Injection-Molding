"""Phase 2 전처리 Ablation — LR-L2 고정, 전처리 옵션 5종 × 5-fold CV.

결과: results/tables/preproc_ablation.csv
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from utils import set_seed, setup_korean_font
from data import load_raw, get_fold, generate_splits
from preprocess import get_feature_cols, fit_transform_fold

set_seed(42)
setup_korean_font()

TABLES_DIR = Path('results/tables')
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
print("Loading labeled_data ...")
df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values

print(f"  shape={df.shape}, features={len(FEAT_COLS)}, fail%={y.mean()*100:.2f}%")

generate_splits(X, y)

# ── Ablation grid ──────────────────────────────────────────────────────────
SCALERS    = ['standard', 'robust']
RESAMPLERS = ['none', 'class_weight', 'smote', 'adasyn', 'undersample']
N_FOLDS    = 5
C_LR       = 1.0   # fixed for ablation


def precision_at_recall(y_true, y_prob, target_recall=0.90):
    """Find precision at the operating point where recall >= target_recall."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    mask = rec >= target_recall
    if not mask.any():
        return 0.0
    return float(prec[mask][-1])


results = []
total = len(SCALERS) * len(RESAMPLERS)
done = 0

for scaler_name, resample_name in product(SCALERS, RESAMPLERS):
    t0 = time.time()
    fold_metrics = {k: [] for k in ['roc_auc', 'pr_auc', 'f1', 'prec_at_rec90']}

    for fold_i in range(N_FOLDS):
        X_tr, X_va, y_tr, y_va = get_fold(fold_i, X, y)

        # Preprocessing INSIDE the fold (no leakage)
        X_tr_s, X_va_s, y_tr_r = fit_transform_fold(
            X_tr, X_va, y_tr, scaler_name, resample_name
        )

        cw = 'balanced' if resample_name == 'class_weight' else None
        clf = LogisticRegression(
            penalty='l2', C=C_LR, class_weight=cw,
            solver='saga', max_iter=2000, random_state=42,
        )
        clf.fit(X_tr_s, y_tr_r)
        y_prob = clf.predict_proba(X_va_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_metrics['roc_auc'].append(roc_auc_score(y_va, y_prob))
        fold_metrics['pr_auc'].append(average_precision_score(y_va, y_prob))
        fold_metrics['f1'].append(f1_score(y_va, y_pred, zero_division=0))
        fold_metrics['prec_at_rec90'].append(precision_at_recall(y_va, y_prob, 0.90))

    elapsed = time.time() - t0
    done += 1
    row = {
        'scaler': scaler_name,
        'resample': resample_name,
        'roc_auc_mean': round(np.mean(fold_metrics['roc_auc']), 4),
        'roc_auc_std':  round(np.std(fold_metrics['roc_auc']),  4),
        'pr_auc_mean':  round(np.mean(fold_metrics['pr_auc']),  4),
        'pr_auc_std':   round(np.std(fold_metrics['pr_auc']),   4),
        'f1_mean':      round(np.mean(fold_metrics['f1']),      4),
        'f1_std':       round(np.std(fold_metrics['f1']),       4),
        'prec_at_rec90_mean': round(np.mean(fold_metrics['prec_at_rec90']), 4),
        'prec_at_rec90_std':  round(np.std(fold_metrics['prec_at_rec90']),  4),
        'n_train_after_resample': int(np.mean([
            get_fold(i, X, y)[2].shape[0] for i in range(N_FOLDS)
        ])),
    }
    results.append(row)
    print(f"  [{done:2d}/{total}] {scaler_name:8s} × {resample_name:12s} | "
          f"ROC-AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}  "
          f"PR-AUC={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f}  "
          f"({elapsed:.1f}s)")

# ── Save ───────────────────────────────────────────────────────────────────
abl_df = pd.DataFrame(results)
# Sort by PR-AUC desc (most relevant for imbalanced data)
abl_df = abl_df.sort_values('pr_auc_mean', ascending=False).reset_index(drop=True)
abl_df.to_csv(TABLES_DIR / 'preproc_ablation.csv', index=False)
print(f"\nSaved: results/tables/preproc_ablation.csv")

# ── Summary ────────────────────────────────────────────────────────────────
best = abl_df.iloc[0]
print("\n=== 전처리 Ablation 결과 (PR-AUC 기준 정렬) ===")
display_cols = ['scaler', 'resample', 'roc_auc_mean', 'roc_auc_std',
                'pr_auc_mean', 'pr_auc_std', 'f1_mean', 'f1_std']
print(abl_df[display_cols].to_string(index=False))
print(f"\n🏆 Best: scaler={best['scaler']}, resample={best['resample']} "
      f"| ROC-AUC={best['roc_auc_mean']:.4f} PR-AUC={best['pr_auc_mean']:.4f}")
