"""Regenerate Phase 4 comparison figures with AdaBoost included."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from utils import set_seed, setup_korean_font
from data import load_raw, get_fold
from preprocess import get_feature_cols, fit_transform_fold
from models.nn import make_mlp, GUIDEBOOK_DNN_PARAMS

set_seed(42)
setup_korean_font()
FIGURES_DIR = Path('results/figures')
TABLES_DIR  = Path('results/tables')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values

# Best configs per model
MODELS = {
    'MLP[256,128,64]': (
        lambda: make_mlp({**{'hidden_dims':(256,128,64),'dropout':0.3,'epochs':30,'lr':1e-3},'random_state':42}),
        'standard', 'smote'
    ),
    'RandomForest(n=500)': (
        lambda: RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1),
        'standard', 'smote'
    ),
    'Guidebook DNN': (
        lambda: make_mlp({**GUIDEBOOK_DNN_PARAMS, 'random_state':42}),
        'standard', 'smote'
    ),
    'GBM(n=100,lr=0.1,d=5)': (
        lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'standard', 'smote'
    ),
    'AdaBoost(n=50)': (
        lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42),
            n_estimators=50, learning_rate=0.5, random_state=42
        ),
        'standard', 'none'
    ),
    'QDA (Phase3)': (
        lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        'standard', 'smote'
    ),
}

COLORS = ['#1f77b4','#2ca02c','#9467bd','#d62728','#ff7f0e','#8c8c8c']

print("Generating ROC/PR curves for Phase 4 models...")
roc_data, pr_data = {}, {}

for (name, (make_fn, scaler, resample)), color in zip(MODELS.items(), COLORS):
    fold_fpr, fold_tpr = [], []
    fold_rec, fold_prec = [], []
    roc_aucs, pr_aucs = [], []

    for fi in range(5):
        Xtr, Xva, ytr, yva = get_fold(fi, X, y)
        Xtr_s, Xva_s, ytr_r = fit_transform_fold(Xtr, Xva, ytr, scaler, resample)
        clf = make_fn()
        clf.fit(Xtr_s, ytr_r)
        yp = clf.predict_proba(Xva_s)[:, 1]
        fpr, tpr, _ = roc_curve(yva, yp)
        prec, rec, _ = precision_recall_curve(yva, yp)
        fold_fpr.append(fpr); fold_tpr.append(tpr)
        fold_rec.append(rec); fold_prec.append(prec)
        roc_aucs.append(auc(fpr, tpr))
        pr_aucs.append(auc(rec[::-1], prec[::-1]))

    fpr_g = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(fpr_g, f, t) for f, t in zip(fold_fpr, fold_tpr)], axis=0)
    rec_g = np.linspace(0, 1, 200)
    interp_precs = [np.interp(rec_g, r[::-1], p[::-1]) for r, p in zip(fold_rec, fold_prec)]
    mean_prec = np.mean(interp_precs, axis=0)
    std_prec  = np.std(interp_precs,  axis=0)

    roc_data[name] = (fpr_g, mean_tpr, np.mean(roc_aucs), color)
    pr_data[name]  = (rec_g, mean_prec, std_prec, np.mean(pr_aucs), color)
    print(f"  {name:25s} ROC={np.mean(roc_aucs):.4f}  PR={np.mean(pr_aucs):.4f}")

# ROC curve
fig, ax = plt.subplots(figsize=(9, 7))
for name, (fg, mt, rv, col) in roc_data.items():
    ls = '--' if 'Phase3' in name else '-'
    ax.plot(fg, mt, color=col, linewidth=2, linestyle=ls, label=f'{name} ({rv:.4f})')
ax.plot([0,1],[0,1],'k--',linewidth=1,label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Phase 4 — ROC 곡선 (앙상블·NN, 5-fold 평균)', fontsize=13)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(FIGURES_DIR/'ensemble_roc_curves.png', bbox_inches='tight')
plt.close()
print("Saved: ensemble_roc_curves.png")

# PR curve
fig, ax = plt.subplots(figsize=(9, 7))
for name, (rg, mp, sp, pv, col) in pr_data.items():
    ls = '--' if 'Phase3' in name else '-'
    ax.plot(rg, mp, color=col, linewidth=2, linestyle=ls, label=f'{name} ({pv:.4f})')
    if 'Phase3' not in name:
        ax.fill_between(rg, mp-sp, mp+sp, color=col, alpha=0.1)
ax.axhline(y.mean(), color='gray', linestyle=':', linewidth=1, label=f'Random({y.mean():.3f})')
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Phase 4 — PR 곡선 (앙상블·NN, 5-fold 평균 ±1std)', fontsize=13)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(FIGURES_DIR/'ensemble_pr_curves.png', bbox_inches='tight')
plt.close()
print("Saved: ensemble_pr_curves.png")

# Bar comparison (full summary incl. AdaBoost)
summary = pd.read_csv(TABLES_DIR/'ensemble_nn_results.csv')
phase_models = summary[~summary['model'].str.contains('Phase')]
phase_models = phase_models.sort_values('pr_auc_mean', ascending=False)
COLORS_BAR = ['#1f77b4','#2ca02c','#9467bd','#d62728','#ff7f0e','#bcbd22'][:len(phase_models)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, ylabel in [(axes[0],'roc_auc','ROC-AUC'),(axes[1],'pr_auc','PR-AUC')]:
    means = phase_models[f'{metric}_mean'].values
    stds  = phase_models[f'{metric}_std'].values
    x = range(len(phase_models))
    bars = ax.bar(x, means, color=COLORS_BAR, alpha=0.82)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)
    ref_idx = 0 if metric=='roc_auc' else 2
    ax.axhline(0.9344 if metric=='roc_auc' else 0.3526,
               color='blue', linestyle='--', linewidth=1.2, label='Phase3 QDA')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_models['model'], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{ylabel} 비교 (Phase 4)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim([0, min(1.0, means.max()+0.12)])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Phase 4: 앙상블·NN vs Phase 3 베이스라인', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'ensemble_nn_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: ensemble_nn_comparison.png (updated with AdaBoost)")
print("Done.")
