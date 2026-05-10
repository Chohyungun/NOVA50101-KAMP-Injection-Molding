"""Phase 4-A — 차원축소 Ablation (LR-L2 고정, StandardScaler+SMOTE).

Methods: None / VarianceThreshold / PCA(80/90/95/99%) / TreeImportance Top-K
Output:  results/tables/dim_reduction_ablation.csv
         results/figures/eda_pca_2d_scatter.png
         results/figures/dim_reduction_feature_importance.png
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

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

from utils import set_seed
from data import load_raw, get_fold
from preprocess import get_feature_cols, fit_transform_fold
from evaluate import compute_metrics, aggregate_fold_metrics

set_seed(42)
TABLES_DIR  = Path('results/tables');  TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path('results/figures'); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')

df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values
N_FOLDS = 5

LR_BEST = dict(penalty='l2', C=10, solver='saga', max_iter=3000, random_state=42)

print(f"Data: {X.shape}, fail%={y.mean()*100:.2f}%\n")


def run_folds_with_dr(dr_fn):
    """5-fold CV with preprocessing + optional DR applied inside each fold."""
    fold_results = []
    n_dims_list = []
    for fold_i in range(N_FOLDS):
        X_tr, X_va, y_tr, y_va = get_fold(fold_i, X, y)
        X_tr_s, X_va_s, y_tr_r = fit_transform_fold(X_tr, X_va, y_tr, 'standard', 'smote')
        X_tr_d, X_va_d = dr_fn(X_tr_s, X_va_s, y_tr_r)
        n_dims_list.append(X_tr_d.shape[1])
        clf = LogisticRegression(**LR_BEST)
        clf.fit(X_tr_d, y_tr_r)
        y_prob = clf.predict_proba(X_va_d)[:, 1]
        fold_results.append(compute_metrics(y_va, y_prob))
    return aggregate_fold_metrics(fold_results), int(np.mean(n_dims_list))


# ── DR methods ─────────────────────────────────────────────────────────────
def dr_none(Xtr, Xva, ytr): return Xtr, Xva

def make_var_thresh(threshold=0.01):
    def dr(Xtr, Xva, ytr):
        sel = VarianceThreshold(threshold=threshold)
        return sel.fit_transform(Xtr), sel.transform(Xva)
    return dr

def make_pca(var_ratio):
    def dr(Xtr, Xva, ytr):
        pca = PCA(n_components=var_ratio, random_state=42)
        return pca.fit_transform(Xtr), pca.transform(Xva)
    return dr

def make_tree_topk(k):
    def dr(Xtr, Xva, ytr):
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        idx = np.argsort(rf.feature_importances_)[-k:]
        return Xtr[:, idx], Xva[:, idx]
    return dr


EXPERIMENTS = [
    ("None (full=25)", dr_none),
    ("VarianceThreshold(0.01)", make_var_thresh(0.01)),
    ("PCA 80%",  make_pca(0.80)),
    ("PCA 90%",  make_pca(0.90)),
    ("PCA 95%",  make_pca(0.95)),
    ("PCA 99%",  make_pca(0.99)),
    ("TreeTop-5",  make_tree_topk(5)),
    ("TreeTop-10", make_tree_topk(10)),
    ("TreeTop-15", make_tree_topk(15)),
]

rows = []
for name, dr_fn in EXPERIMENTS:
    t0 = time.time()
    agg, n_dim = run_folds_with_dr(dr_fn)
    elapsed = time.time() - t0
    row = {"method": name, "n_dims": n_dim, **agg}
    rows.append(row)
    print(f"  {name:30s} dims={n_dim:2d} | "
          f"ROC={agg['roc_auc_mean']:.4f}+/-{agg['roc_auc_std']:.4f}  "
          f"PR={agg['pr_auc_mean']:.4f}+/-{agg['pr_auc_std']:.4f}  ({elapsed:.1f}s)")

dr_df = pd.DataFrame(rows)
dr_df.to_csv(TABLES_DIR / 'dim_reduction_ablation.csv', index=False)
print(f"\nSaved: dim_reduction_ablation.csv")

# ── PCA 2D scatter plot ────────────────────────────────────────────────────
print("\nGenerating PCA 2D scatter...")
from sklearn.preprocessing import StandardScaler as SS
sc = SS()
X_s = sc.fit_transform(X)
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_s)
ev = pca2.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
colors = {0: '#4C72B0', 1: '#DD8452'}
labels = {0: f'양품 (n={int((y==0).sum())})', 1: f'불량 (n={int((y==1).sum())})'}
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors[cls], label=labels[cls],
               alpha=0.5 if cls == 0 else 0.9,
               s=10 if cls == 0 else 40,
               edgecolors='none' if cls == 0 else 'black', linewidths=0.5)
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% 분산)', fontsize=12)
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% 분산)', fontsize=12)
ax.set_title('PCA 2D 시각화 — 양품 vs 불량 분포', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_pca_2d_scatter.png', bbox_inches='tight')
plt.close()
print("Saved: eda_pca_2d_scatter.png")

# ── Feature importance (RF) ────────────────────────────────────────────────
print("Generating feature importance plot...")
from sklearn.preprocessing import StandardScaler as SS2
sc2 = SS2()
X_s2 = sc2.fit_transform(X)
rf_imp = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                random_state=42, n_jobs=-1)
rf_imp.fit(X_s2, y)
importances = rf_imp.feature_importances_
order = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(len(FEAT_COLS)), importances[order], color='#4C72B0', alpha=0.8)
ax.set_xticks(range(len(FEAT_COLS)))
ax.set_xticklabels([FEAT_COLS[i] for i in order], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Feature Importance (RF)', fontsize=11)
ax.set_title('RF Feature Importance — TreeTop-K 선택 기준 (n_estimators=300)', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dim_reduction_feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: dim_reduction_feature_importance.png")

# ── DR result chart ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = range(len(dr_df))
for ax, metric, ylabel in [(axes[0],'roc_auc','ROC-AUC'),(axes[1],'pr_auc','PR-AUC')]:
    means = dr_df[f'{metric}_mean'].values
    stds  = dr_df[f'{metric}_std'].values
    bars  = ax.bar(x, means, color='#4C72B0', alpha=0.75)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(dr_df['method'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'차원축소 방법별 {ylabel}', fontsize=12)
    ax.set_ylim([0, min(1.0, means.max()+0.12)])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
plt.suptitle('Phase 4-A 차원축소 Ablation (LR-L2 C=10 고정, 5-fold)', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dim_reduction_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: dim_reduction_comparison.png")

print("\n=== DR Ablation 결과 ===")
print(dr_df[['method','n_dims','roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std']].to_string(index=False))
best = dr_df.loc[dr_df['roc_auc_mean'].idxmax()]
print(f"\nBest ROC-AUC: {best['method']} ({best['roc_auc_mean']:.4f})")
best_pr = dr_df.loc[dr_df['pr_auc_mean'].idxmax()]
print(f"Best PR-AUC : {best_pr['method']} ({best_pr['pr_auc_mean']:.4f})")
