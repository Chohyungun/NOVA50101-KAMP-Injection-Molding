"""Phase 5 — 1D-CNN + Stacking + 운용점 분석.

CNN   : 25 피처를 1D 시퀀스로 처리 (PyTorch Conv1d × 2 → FC)
Stack : LR-L2/QDA/RF/GBM/MLP OOF → 메타 LR-L2
운용점 : Precision=0.95, 0.99 고정 시 Recall
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from utils import set_seed, setup_korean_font
from data import load_raw, get_fold
from preprocess import get_feature_cols, fit_transform_fold
from evaluate import compute_metrics, aggregate_fold_metrics
from models.stacking import get_base_configs, get_meta_learner

set_seed(42)
setup_korean_font()
TABLES_DIR  = Path('results/tables');  TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path('results/figures'); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

df = load_raw('labeled_data')
FEAT_COLS = get_feature_cols(df)
X = df[FEAT_COLS].values
y = df['PassOrFail'].values
N_FOLDS = 5
N_FEATS = X.shape[1]

print(f"Data: {X.shape}, fail%={y.mean()*100:.2f}%\n")


# ═══════════════════════════════════════════════════════════════════════════
# Part A — 1D-CNN
# ═══════════════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """25 features → (1, 25) → Conv1d × 2 → FC → binary logit."""

    def __init__(self, n_feats: int = 25, ch1: int = 16, ch2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch1, ch2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        conv_out = ch2 * (n_feats // 4)   # two MaxPool1d(2): 25→12→6
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, 1, F)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


class CNN1DTrainer:
    """sklearn-style interface for CNN1D."""

    def __init__(self, ch1=16, ch2=32, dropout=0.3, lr=1e-3,
                 epochs=30, batch_size=256, random_state=42):
        self.ch1, self.ch2, self.dropout = ch1, ch2, dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.random_state = random_state
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        self.model_ = CNN1D(X.shape[1], self.ch1, self.ch2, self.dropout)
        n_pos = y.sum(); n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(self.random_state))
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward(); optimizer.step()
        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = torch.sigmoid(self.model_(torch.tensor(X, dtype=torch.float32))).numpy()
        return np.column_stack([1 - probs, probs])


CNN_GRID = [
    {"ch1": 16, "ch2": 32, "dropout": 0.3, "epochs": 30},
    {"ch1": 32, "ch2": 64, "dropout": 0.3, "epochs": 50},
]

print("=" * 60)
print("Part A - 1D-CNN")
print("=" * 60)
cnn_rows = []
for params in CNN_GRID:
    t0 = time.time()
    folds = []
    for fi in range(N_FOLDS):
        Xtr, Xva, ytr, yva = get_fold(fi, X, y)
        Xtr_s, Xva_s, ytr_r = fit_transform_fold(Xtr, Xva, ytr, 'standard', 'smote')
        clf = CNN1DTrainer(**params, random_state=42)
        clf.fit(Xtr_s, ytr_r)
        yp = clf.predict_proba(Xva_s)[:, 1]
        folds.append(compute_metrics(yva, yp))
    agg = aggregate_fold_metrics(folds)
    cnn_rows.append({"model": f"CNN1D({params['ch1']},{params['ch2']})", **agg})
    print(f"  CNN({params['ch1']},{params['ch2']}) ep={params['epochs']} | "
          f"ROC={agg['roc_auc_mean']:.4f}+/-{agg['roc_auc_std']:.4f}  "
          f"PR={agg['pr_auc_mean']:.4f}+/-{agg['pr_auc_std']:.4f}  ({time.time()-t0:.1f}s)")

cnn_df = pd.DataFrame(cnn_rows)
print()


# ═══════════════════════════════════════════════════════════════════════════
# Part B — Stacking
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Part B - Stacking (OOF -> meta LR)")
print("=" * 60)

BASE_CONFIGS = get_base_configs()   # [(name, factory_fn), ...]

# SMOTE-based models vs non-SMOTE
SMOTE_BASES = {"LR_L2", "QDA", "RF", "GBM", "MLP"}

# Step 1: generate OOF predictions for all base models
oof_preds = np.zeros((len(X), len(BASE_CONFIGS)))

from data import SPLITS_DIR

for j, (name, make_fn) in enumerate(BASE_CONFIGS):
    t0 = time.time()
    resample = 'smote' if name in SMOTE_BASES else 'none'
    for fi in range(N_FOLDS):
        split = np.load(SPLITS_DIR / f"fold_{fi}.npy", allow_pickle=True).item()
        tr_idx, va_idx = split["train"], split["val"]
        Xtr_s, Xva_s, ytr_r = fit_transform_fold(
            X[tr_idx], X[va_idx], y[tr_idx], 'standard', resample)
        clf = make_fn()
        clf.fit(Xtr_s, ytr_r)
        oof_preds[va_idx, j] = clf.predict_proba(Xva_s)[:, 1]
    print(f"  [{j+1}/{len(BASE_CONFIGS)}] {name:8s} OOF done ({time.time()-t0:.1f}s)")

# Step 2: 5-fold CV of meta-LR on OOF meta-features
meta_X = oof_preds
meta_y = y

print("\nMeta-LR 5-fold CV on OOF features:")
meta_folds = []
for fi in range(N_FOLDS):
    split = np.load(SPLITS_DIR / f"fold_{fi}.npy", allow_pickle=True).item()  # noqa: F821
    tr_idx, va_idx = split["train"], split["val"]
    meta_clf = get_meta_learner()
    meta_clf.fit(meta_X[tr_idx], meta_y[tr_idx])
    yp = meta_clf.predict_proba(meta_X[va_idx])[:, 1]
    m = compute_metrics(meta_y[va_idx], yp)
    meta_folds.append(m)
    print(f"  fold {fi}: ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}")

meta_agg = aggregate_fold_metrics(meta_folds)
print(f"\nStacking (meta-LR) 5-fold 평균: "
      f"ROC={meta_agg['roc_auc_mean']:.4f}+/-{meta_agg['roc_auc_std']:.4f}  "
      f"PR={meta_agg['pr_auc_mean']:.4f}+/-{meta_agg['pr_auc_std']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Part C — 운용점 분석 (Precision @ 0.95 / 0.99)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Part C - Operating Point Analysis")
print("=" * 60)

# Use Stacking OOF predictions for operating point analysis
# Also compare with MLP best

def get_operating_points(y_true, y_prob, precisions=(0.95, 0.99)):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    ops = {}
    for target_p in precisions:
        mask = prec >= target_p
        if mask.any():
            idx = np.argmax(mask)  # first threshold where precision >= target
            ops[f"prec>={target_p}"] = {
                "precision": float(prec[idx]),
                "recall":    float(rec[idx]),
                "threshold": float(thr[idx]) if idx < len(thr) else 1.0,
            }
        else:
            ops[f"prec>={target_p}"] = {"precision": None, "recall": 0.0, "threshold": None}
    return ops

# Collect OOF probs for Stacking + MLP for operating point analysis
stacking_probs = np.zeros(len(X))
for fi in range(N_FOLDS):
    split = np.load(SPLITS_DIR / f"fold_{fi}.npy", allow_pickle=True).item()
    tr_idx, va_idx = split["train"], split["val"]
    meta_clf2 = get_meta_learner()
    meta_clf2.fit(meta_X[tr_idx], meta_y[tr_idx])
    stacking_probs[va_idx] = meta_clf2.predict_proba(meta_X[va_idx])[:, 1]

op_rows = []
models_for_op = {
    "Stacking": (stacking_probs, y),
}

# Also collect MLP OOF
from models.nn import make_mlp
mlp_oof = np.zeros(len(X))
for fi in range(N_FOLDS):
    split = np.load(SPLITS_DIR / f"fold_{fi}.npy", allow_pickle=True).item()
    tr_idx, va_idx = split["train"], split["val"]
    Xtr_s, Xva_s, ytr_r = fit_transform_fold(X[tr_idx], X[va_idx], y[tr_idx], 'standard', 'smote')
    clf = make_mlp({"hidden_dims": (256, 128, 64), "dropout": 0.3, "epochs": 30, "lr": 1e-3, "random_state": 42})
    clf.fit(Xtr_s, ytr_r); mlp_oof[va_idx] = clf.predict_proba(Xva_s)[:, 1]
models_for_op["MLP[256,128,64]"] = (mlp_oof, y)

for model_name, (probs, ytrue) in models_for_op.items():
    ops = get_operating_points(ytrue, probs)
    for label, vals in ops.items():
        op_rows.append({
            "model": model_name, "target_precision": label,
            "achieved_precision": round(vals["precision"] or 0, 4),
            "recall_at_target":   round(vals["recall"], 4),
            "threshold":          round(vals["threshold"] or 0, 4) if vals["threshold"] else None,
        })

op_df = pd.DataFrame(op_rows)
print(op_df.to_string(index=False))
op_df.to_csv(TABLES_DIR / 'operating_points.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════

# Combined Phase 5 results
phase5_rows = list(cnn_df.to_dict('records')) + [{
    "model": "Stacking(LR-meta)",
    **meta_agg,
}]
phase5_df = pd.DataFrame(phase5_rows)
phase5_df.to_csv(TABLES_DIR / 'phase5_results.csv', index=False)

print("\n=== Phase 5 전체 결과 ===")
print(phase5_df[['model','roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std']].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

# All-phase PR curve comparison
all_summary = {
    "LR-L2 (Ph3)":       (0.9343, 0.2690),
    "QDA (Ph3)":          (0.9344, 0.3526),
    "RF (Ph4)":           (0.9478, 0.4481),
    "MLP[256,128,64](Ph4)":(0.9497, 0.4710),
    f"CNN1D (Ph5)":       (cnn_df['roc_auc_mean'].max(), cnn_df.loc[cnn_df['roc_auc_mean'].idxmax(),'pr_auc_mean']),
    "Stacking (Ph5)":     (meta_agg['roc_auc_mean'], meta_agg['pr_auc_mean']),
}

# PR curve with operating point markers
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: all-phase summary bar
ax = axes[0]
models = list(all_summary.keys())
pr_vals = [v[1] for v in all_summary.values()]
roc_vals = [v[0] for v in all_summary.values()]
COLS = ['#8c8c8c','#8c8c8c','#2ca02c','#1f77b4','#ff7f0e','#d62728']
bars = ax.barh(models, pr_vals, color=COLS, alpha=0.82)
ax.axvline(0.3526, color='blue', linestyle='--', linewidth=1, label='Phase3 QDA PR-AUC')
for bar, val in zip(bars, pr_vals):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('PR-AUC (5-fold 평균)', fontsize=11)
ax.set_title('전체 Phase PR-AUC 비교', fontsize=12)
ax.legend(fontsize=8)

# Right: Stacking PR curve with operating point markers
ax2 = axes[1]
prec_st, rec_st, _ = precision_recall_curve(y, stacking_probs)
prec_mlp, rec_mlp, _ = precision_recall_curve(y, mlp_oof)

ax2.plot(rec_st,  prec_st,  color='#d62728', linewidth=2, label='Stacking')
ax2.plot(rec_mlp, prec_mlp, color='#1f77b4', linewidth=2, label='MLP[256,128,64]')
ax2.axhline(y.mean(), color='gray', linestyle=':', linewidth=1, label=f'Random({y.mean():.3f})')

# Mark operating points
for target_p, marker, color in [(0.95, 's', '#d62728'), (0.99, '^', '#d62728')]:
    mask = prec_st >= target_p
    if mask.any():
        idx = np.argmax(mask)
        ax2.scatter(rec_st[idx], prec_st[idx], marker=marker, s=120,
                    color=color, zorder=5, label=f'Stacking Prec≥{target_p}(Rec={rec_st[idx]:.3f})')

for target_p, marker, color in [(0.95, 's', '#1f77b4'), (0.99, '^', '#1f77b4')]:
    mask = prec_mlp >= target_p
    if mask.any():
        idx = np.argmax(mask)
        ax2.scatter(rec_mlp[idx], prec_mlp[idx], marker=marker, s=120,
                    color=color, zorder=5, label=f'MLP Prec≥{target_p}(Rec={rec_mlp[idx]:.3f})')

ax2.set_xlabel('Recall', fontsize=11); ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title('운용점 분석 — Stacking vs MLP (PR 곡선)', fontsize=12)
ax2.legend(fontsize=7.5, loc='upper right')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])

plt.suptitle('Phase 5 — CNN·Stacking 결과 + 운용점 분석', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase5_summary.png', bbox_inches='tight')
plt.close()
print("\nSaved: phase5_summary.png")

print(f"\nSaved: phase5_results.csv, operating_points.csv")
print("Phase 5 완료")
