"""Generate ROC and PR curve figures for Phase 3 baselines."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from utils import set_seed, setup_korean_font
from data import load_raw, get_fold
from preprocess import get_feature_cols, fit_transform_fold

set_seed(42)
setup_korean_font()
FIGURES_DIR = Path('results/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
setup_korean_font()
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values

# Best configs per model
MODELS = {
    'QDA (reg=0.01)': (
        lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        'standard', 'smote',
    ),
    'LR-L1 (C=10)': (
        lambda: LogisticRegression(penalty='l1', C=10, solver='saga',
                                   max_iter=3000, random_state=42),
        'standard', 'smote',
    ),
    'LR-L2 (C=10)': (
        lambda: LogisticRegression(penalty='l2', C=10, solver='saga',
                                   max_iter=3000, random_state=42),
        'standard', 'smote',
    ),
    'LDA': (
        lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'standard', 'smote',
    ),
    'SVM-linear (C=10)': (
        lambda: CalibratedClassifierCV(
            LinearSVC(C=10, class_weight='balanced', max_iter=5000, random_state=42),
            cv=3, method='sigmoid'
        ),
        'standard', 'none',
    ),
    'SVM-RBF (C=10,g=0.1)': (
        lambda: SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced',
                    probability=True, random_state=42),
        'standard', 'none',
    ),
}

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

# Collect fold-level curves
roc_data = {}
pr_data  = {}

for (name, (make_fn, scaler, resample)), color in zip(MODELS.items(), COLORS):
    fold_fpr_list, fold_tpr_list = [], []
    fold_rec_list, fold_prec_list = [], []
    auc_rocs, auc_prs = [], []

    for fold_i in range(5):
        X_tr, X_va, y_tr, y_va = get_fold(fold_i, X, y)
        X_tr_s, X_va_s, y_tr_r = fit_transform_fold(X_tr, X_va, y_tr, scaler, resample)
        clf = make_fn()
        clf.fit(X_tr_s, y_tr_r)
        y_prob = clf.predict_proba(X_va_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_va, y_prob)
        prec, rec, _ = precision_recall_curve(y_va, y_prob)

        fold_fpr_list.append(fpr)
        fold_tpr_list.append(tpr)
        fold_rec_list.append(rec)
        fold_prec_list.append(prec)
        auc_rocs.append(auc(fpr, tpr))
        auc_prs.append(auc(rec[::-1], prec[::-1]))

    mean_roc = np.mean(auc_rocs)
    mean_pr  = np.mean(auc_prs)

    # Interpolate to common grid
    fpr_grid = np.linspace(0, 1, 200)
    tprs_interp = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr in zip(fold_fpr_list, fold_tpr_list)]
    mean_tpr = np.mean(tprs_interp, axis=0)

    rec_grid = np.linspace(0, 1, 200)
    precs_interp = [np.interp(rec_grid, rec[::-1], prec[::-1])
                    for rec, prec in zip(fold_rec_list, fold_prec_list)]
    mean_prec = np.mean(precs_interp, axis=0)
    std_prec  = np.std(precs_interp,  axis=0)

    roc_data[name] = (fpr_grid, mean_tpr, mean_roc, color)
    pr_data[name]  = (rec_grid, mean_prec, std_prec, mean_pr, color)
    print(f'  {name:25s} ROC-AUC={mean_roc:.4f}  PR-AUC={mean_pr:.4f}')

# ── ROC curve ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for name, (fpr_g, tpr_g, roc_val, color) in roc_data.items():
    ax.plot(fpr_g, tpr_g, color=color, linewidth=2, label=f'{name} (AUC={roc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Phase 3 — ROC 곡선 비교 (5-fold 평균)', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_roc_curves.png', bbox_inches='tight')
plt.close()
print('Saved: baseline_roc_curves.png')

# ── PR curve ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
baseline_pr = y.mean()
for name, (rec_g, prec_g, std_g, pr_val, color) in pr_data.items():
    ax.plot(rec_g, prec_g, color=color, linewidth=2, label=f'{name} (AUC={pr_val:.4f})')
    ax.fill_between(rec_g, prec_g - std_g, prec_g + std_g, color=color, alpha=0.1)
ax.axhline(baseline_pr, color='gray', linestyle='--', linewidth=1,
           label=f'Random ({baseline_pr:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Phase 3 — PR 곡선 비교 (5-fold 평균 ± 1std)', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_pr_curves.png', bbox_inches='tight')
plt.close()
print('Saved: baseline_pr_curves.png')

# ── Bar chart: ROC-AUC vs PR-AUC comparison ───────────────────────────────
summary = pd.read_csv('results/tables/baseline_results.csv').sort_values('pr_auc_mean', ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = range(len(summary))
for ax, metric, ylabel in [
    (axes[0], 'roc_auc', 'ROC-AUC'),
    (axes[1], 'pr_auc',  'PR-AUC'),
]:
    means = summary[f'{metric}_mean'].values
    stds  = summary[f'{metric}_std'].values
    bars = ax.bar(x, means, color=COLORS[:len(summary)], alpha=0.8)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['model'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{ylabel} 비교 (Phase 3)', fontsize=12)
    ax.set_ylim([0, min(1.0, means.max() + 0.15)])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Phase 3 선형·생성적 베이스라인 결과 (best per family, 5-fold)', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'baseline_comparison_bar.png', bbox_inches='tight')
plt.close()
print('Saved: baseline_comparison_bar.png')
