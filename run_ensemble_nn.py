"""Phase 4-B — 앙상블 + MLP (PyTorch) 5-fold CV.

모델: RandomForest, AdaBoost, GradientBoosting, MLP, Guidebook-DNN
전처리: StandardScaler + SMOTE (RF/AdaBoost/GBM/MLP)
출력: results/tables/ensemble_nn_results.csv
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seed
from data import load_raw, get_fold
from preprocess import get_feature_cols, fit_transform_fold
from evaluate import compute_metrics, aggregate_fold_metrics
from models.tree import (
    RF_GRID, ADABOOST_GRID, GBM_GRID,
    make_rf, make_adaboost, make_gbm,
)
from models.nn import MLP_GRID, GUIDEBOOK_DNN_PARAMS, make_mlp

set_seed(42)
TABLES_DIR  = Path('results/tables');  TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path('results/figures'); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')

df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values
N_FOLDS = 5

print(f"Data: {X.shape}, fail%={y.mean()*100:.2f}%\n")


def run_folds(make_fn, scaler='standard', resample='smote'):
    fold_results = []
    for fold_i in range(N_FOLDS):
        X_tr, X_va, y_tr, y_va = get_fold(fold_i, X, y)
        X_tr_s, X_va_s, y_tr_r = fit_transform_fold(X_tr, X_va, y_tr, scaler, resample)
        clf = make_fn()
        clf.fit(X_tr_s, y_tr_r)
        y_prob = clf.predict_proba(X_va_s)[:, 1]
        fold_results.append(compute_metrics(y_va, y_prob))
    return aggregate_fold_metrics(fold_results)


# ── Experiments ────────────────────────────────────────────────────────────
EXPERIMENTS = [
    ("RandomForest",    RF_GRID,           lambda p: make_rf(p)),
    ("AdaBoost",        ADABOOST_GRID,     lambda p: make_adaboost(p)),
    ("GradientBoosting",GBM_GRID,          lambda p: make_gbm(p)),
    ("MLP_PyTorch",     MLP_GRID,          lambda p: make_mlp({**p, 'random_state': 42})),
    ("Guidebook_DNN",   [GUIDEBOOK_DNN_PARAMS], lambda p: make_mlp({**p, 'random_state': 42})),
]

all_rows = []
done = 0

for family, grid, make_fn in EXPERIMENTS:
    t_fam = time.time()
    family_rows = []
    for params in grid:
        t0 = time.time()
        p_copy = dict(params)
        try:
            agg = run_folds(lambda p=p_copy: make_fn(p))
        except Exception as e:
            print(f"  ERROR {family} {params}: {e}")
            agg = {k: float('nan') for k in
                   ['roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std','f1_mean','f1_std']}
        row = {"family": family, **params, **agg}
        family_rows.append(row)
        done += 1
        param_str = " ".join(f"{k}={v}" for k, v in params.items())[:40]
        print(f"  [{done:3d}] {family:20s} {param_str:40s} | "
              f"ROC={agg.get('roc_auc_mean',float('nan')):.4f}  "
              f"PR={agg.get('pr_auc_mean',float('nan')):.4f}  "
              f"({time.time()-t0:.1f}s)")

    fam_df = pd.DataFrame(family_rows)
    fam_df['is_best'] = fam_df['roc_auc_mean'] == fam_df['roc_auc_mean'].max()
    all_rows.extend(fam_df.to_dict('records'))
    print(f"  --> {family} done in {time.time()-t_fam:.1f}s\n")

# ── Save full grid ─────────────────────────────────────────────────────────
full_df = pd.DataFrame(all_rows)
full_df.to_csv(TABLES_DIR / 'ensemble_nn_grid_full.csv', index=False)

# ── Best-per-family summary ────────────────────────────────────────────────
phase3_best = {
    "QDA (Phase 3 best)": (0.9344, 0.0175, 0.3526, 0.1026),
    "LR-L2 (Phase 3 ref)": (0.9343, 0.0118, 0.2690, 0.0537),
}

summary_rows = []
for fam in ["RandomForest","AdaBoost","GradientBoosting","MLP_PyTorch","Guidebook_DNN"]:
    sub = full_df[full_df['family']==fam].dropna(subset=['roc_auc_mean'])
    if len(sub) == 0: continue
    r = sub.sort_values('roc_auc_mean', ascending=False).iloc[0]
    summary_rows.append({
        'model': fam,
        'roc_auc_mean': r['roc_auc_mean'], 'roc_auc_std': r['roc_auc_std'],
        'pr_auc_mean':  r['pr_auc_mean'],  'pr_auc_std':  r['pr_auc_std'],
        'f1_mean':      r['f1_mean'],       'f1_std':      r['f1_std'],
        'guidebook_note': 'Guidebook DNN §2.3 재현' if fam == 'Guidebook_DNN' else 'N/A',
    })

# Add Phase 3 best as reference rows
for name, (rm, rs, pm, ps) in phase3_best.items():
    summary_rows.append({
        'model': name, 'roc_auc_mean': rm, 'roc_auc_std': rs,
        'pr_auc_mean': pm, 'pr_auc_std': ps, 'f1_mean': None, 'f1_std': None,
        'guidebook_note': 'Phase 3 reference',
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES_DIR / 'ensemble_nn_results.csv', index=False)

print("\n=== Phase 4-B 앙상블+NN 결과 (ROC-AUC 정렬) ===")
print(summary_df[['model','roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std']].sort_values('roc_auc_mean', ascending=False).to_string(index=False))

# ── ROC/PR comparison figure ───────────────────────────────────────────────
best_models = summary_df[~summary_df['model'].str.contains('Phase')].sort_values('pr_auc_mean', ascending=False)

COLORS = ['#2ca02c','#ff7f0e','#d62728','#9467bd','#8c564b']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, ylabel in [(axes[0],'roc_auc','ROC-AUC'),(axes[1],'pr_auc','PR-AUC')]:
    models = best_models['model'].tolist()
    means  = best_models[f'{metric}_mean'].values
    stds   = best_models[f'{metric}_std'].values
    x = range(len(models))
    bars = ax.bar(x, means, color=COLORS[:len(models)], alpha=0.8)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)
    # Phase 3 baselines
    ax.axhline(phase3_best['QDA (Phase 3 best)'][0 if metric=='roc_auc' else 2],
               color='blue', linestyle='--', linewidth=1.2, label='Phase3 QDA')
    ax.axhline(phase3_best['LR-L2 (Phase 3 ref)'][0 if metric=='roc_auc' else 2],
               color='gray', linestyle=':', linewidth=1.0, label='Phase3 LR-L2')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{ylabel} 비교 (Phase 4 앙상블·NN)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim([0, min(1.0, means.max()+0.12)])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Phase 4-B: 앙상블·NN best vs Phase 3 베이스라인', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ensemble_nn_comparison.png', bbox_inches='tight')
plt.close()
print("\nSaved: ensemble_nn_comparison.png")
print(f"Saved: ensemble_nn_results.csv  ensemble_nn_grid_full.csv")
