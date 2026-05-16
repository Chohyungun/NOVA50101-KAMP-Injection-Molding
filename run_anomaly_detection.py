"""비지도 이상탐지 비교 실험 — One-Class SVM vs K-Means 거리 기반.

학습 데이터:
  OC-SVM  : labeled 양품(fold train) 만 사용
  K-Means : unlabeled 795K 전체 사용 (레이블 불필요)

평가 기준: labeled_data 5-fold StratifiedKFold val fold
지표: ROC-AUC, PR-AUC (불균형 데이터 적합)

Output figures:
  results/figures/anomaly_kmeans_k_ablation.png
  results/figures/anomaly_detection_roc.png
  results/figures/anomaly_score_distribution.png
Output table:
  results/tables/anomaly_detection_results.csv
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

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

from utils import set_seed, setup_korean_font
from data import load_raw, get_fold
from preprocess import get_feature_cols

set_seed(42)
TABLES_DIR  = Path('results/tables');  TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path('results/figures'); FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
setup_korean_font()

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("데이터 로드 중...")
df_lab   = load_raw('labeled_data')
df_unlab = load_raw('unlabeled_data')

FEAT_COLS = get_feature_cols(df_lab)
# unlabeled에 없는 피처 제거 (안전하게 교집합)
FEAT_COLS = [c for c in FEAT_COLS if c in df_unlab.columns]

X_lab   = df_lab[FEAT_COLS].values
y_lab   = df_lab['PassOrFail'].values
X_unlab = df_unlab[FEAT_COLS].values

print(f"Labeled : {X_lab.shape}, 불량률={y_lab.mean()*100:.2f}%")
print(f"Unlabeled: {X_unlab.shape}")
print(f"사용 피처: {len(FEAT_COLS)}개\n")

N_FOLDS  = 5
K_VALUES = [5, 10, 20, 50]

# ── unlabeled 기준 StandardScaler (K-Means용) ────────────────────────────────
scaler_unlab = StandardScaler()
X_unlab_sc   = scaler_unlab.fit_transform(X_unlab)
X_lab_sc     = scaler_unlab.transform(X_lab)   # K-Means 이상 점수 계산에 사용

# ── K-Means 학습 (unlabeled 전체) ────────────────────────────────────────────
print("K-Means 학습 (unlabeled 795K)...")
km_models = {}
for k in K_VALUES:
    t0 = time.time()
    # 795K 행 → MiniBatchKMeans로 속도 개선, 결과는 유사
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                         batch_size=4096, max_iter=300)
    km.fit(X_unlab_sc)
    km_models[k] = km
    print(f"  k={k:2d}: {time.time()-t0:.1f}s")

# ── K-Means: 5-fold 이상 점수 평가 ──────────────────────────────────────────
print("\nK-Means 평가 (labeled 5-fold val)...")
km_fold_rows = []
for fold_i in range(N_FOLDS):
    _, X_va, _, y_va = get_fold(fold_i, X_lab, y_lab)
    X_va_sc = scaler_unlab.transform(X_va)
    for k in K_VALUES:
        dists = km_models[k].transform(X_va_sc)   # (n_va, k) 각 centroid까지 거리
        score = dists.min(axis=1)                  # 최근접 centroid 거리 = 이상 점수
        roc = roc_auc_score(y_va, score)
        pr  = average_precision_score(y_va, score)
        km_fold_rows.append({'k': k, 'fold': fold_i, 'roc_auc': roc, 'pr_auc': pr})

km_df = pd.DataFrame(km_fold_rows)
km_summary = (km_df.groupby('k')
              .agg(roc_mean=('roc_auc','mean'), roc_std=('roc_auc','std'),
                   pr_mean=('pr_auc','mean'),  pr_std=('pr_auc','std'))
              .reset_index())

print("K-Means 결과:")
for _, row in km_summary.iterrows():
    print(f"  k={int(row.k):2d}  ROC={row.roc_mean:.4f}±{row.roc_std:.4f}  "
          f"PR={row.pr_mean:.4f}±{row.pr_std:.4f}")

best_k_roc = int(km_summary.loc[km_summary.roc_mean.idxmax(), 'k'])
best_k_pr  = int(km_summary.loc[km_summary.pr_mean.idxmax(),  'k'])
print(f"  Best ROC-AUC: k={best_k_roc}")
print(f"  Best PR-AUC : k={best_k_pr}")

# ── One-Class SVM: 5-fold 평가 ───────────────────────────────────────────────
print("\nOne-Class SVM 평가 (fold-wise, 양품 train-fold 기준)...")
ocsvm_rows = []
for fold_i in range(N_FOLDS):
    X_tr, X_va, y_tr, y_va = get_fold(fold_i, X_lab, y_lab)
    # train fold 양품만으로 OC-SVM 학습
    X_tr_normal = X_tr[y_tr == 0]
    sc = StandardScaler()
    X_tr_n_sc = sc.fit_transform(X_tr_normal)
    X_va_sc   = sc.transform(X_va)

    oc = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
    oc.fit(X_tr_n_sc)
    score = -oc.score_samples(X_va_sc)   # 높을수록 이상치
    roc = roc_auc_score(y_va, score)
    pr  = average_precision_score(y_va, score)
    ocsvm_rows.append({'fold': fold_i, 'roc_auc': roc, 'pr_auc': pr})
    print(f"  fold {fold_i}: ROC={roc:.4f}  PR={pr:.4f}")

ocsvm_df = pd.DataFrame(ocsvm_rows)
ocsvm_roc = ocsvm_df.roc_auc.mean()
ocsvm_pr  = ocsvm_df.pr_auc.mean()
print(f"  OC-SVM 평균: ROC={ocsvm_roc:.4f}±{ocsvm_df.roc_auc.std():.4f}  "
      f"PR={ocsvm_pr:.4f}±{ocsvm_df.pr_auc.std():.4f}")

# ── 결과 통합 CSV 저장 ────────────────────────────────────────────────────────
summary_rows = []
for _, row in km_summary.iterrows():
    summary_rows.append({
        'method': f'K-Means (k={int(row.k)})',
        'train_data': 'unlabeled 795K',
        'roc_auc_mean': round(row.roc_mean, 4), 'roc_auc_std': round(row.roc_std, 4),
        'pr_auc_mean':  round(row.pr_mean, 4),  'pr_auc_std':  round(row.pr_std, 4),
    })
summary_rows.append({
    'method': 'One-Class SVM (nu=0.02)',
    'train_data': 'labeled 양품 7,925',
    'roc_auc_mean': round(ocsvm_roc, 4),
    'roc_auc_std':  round(ocsvm_df.roc_auc.std(), 4),
    'pr_auc_mean':  round(ocsvm_pr, 4),
    'pr_auc_std':   round(ocsvm_df.pr_auc.std(), 4),
})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES_DIR / 'anomaly_detection_results.csv', index=False)
print("\nSaved: anomaly_detection_results.csv")

# ── Figure 1: K-Means k ablation (ROC-AUC + PR-AUC bar chart) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(K_VALUES))
width = 0.6

for ax, metric, ylabel, col in [
    (axes[0], 'roc', 'ROC-AUC', '#4C72B0'),
    (axes[1], 'pr',  'PR-AUC',  '#DD8452'),
]:
    means = km_summary[f'{metric}_mean'].values
    stds  = km_summary[f'{metric}_std'].values
    bars  = ax.bar(x, means, width, color=col, alpha=0.75)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'K-Means 클러스터 수(k) vs {ylabel}', fontsize=12)
    ax.set_ylim(0, min(1.0, means.max() + 0.15))
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('K-Means 거리 기반 이상탐지 — k 튜닝 비교 (unlabeled 학습, labeled 5-fold 평가)',
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'anomaly_kmeans_k_ablation.png', bbox_inches='tight')
plt.close()
print("Saved: anomaly_kmeans_k_ablation.png")

# ── Figure 2: ROC 곡선 비교 (best K-Means vs OC-SVM, fold 0) ────────────────
_, X_va0, _, y_va0 = get_fold(0, X_lab, y_lab)

# K-Means best k score (fold 0)
X_va0_unsc = scaler_unlab.transform(X_va0)
km_best = km_models[best_k_roc]
km_score0 = km_best.transform(X_va0_unsc).min(axis=1)

# OC-SVM score (fold 0, 재학습)
X_tr0, _, y_tr0, _ = get_fold(0, X_lab, y_lab)
sc0 = StandardScaler()
X_tr0_n = sc0.fit_transform(X_tr0[y_tr0 == 0])
X_va0_sc = sc0.transform(X_va0)
oc0 = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
oc0.fit(X_tr0_n)
oc_score0 = -oc0.score_samples(X_va0_sc)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curves
for ax_idx, (ax, title) in enumerate([(axes[0], 'ROC 곡선'), (axes[1], 'PR 곡선')]):
    if ax_idx == 0:
        fpr_km, tpr_km, _ = roc_curve(y_va0, km_score0)
        fpr_oc, tpr_oc, _ = roc_curve(y_va0, oc_score0)
        ax.plot(fpr_km, tpr_km,
                label=f'K-Means (k={best_k_roc}, AUC={roc_auc_score(y_va0, km_score0):.3f})',
                color='#2196F3', lw=2)
        ax.plot(fpr_oc, tpr_oc,
                label=f'One-Class SVM (AUC={roc_auc_score(y_va0, oc_score0):.3f})',
                color='#E91E63', lw=2)
        ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
    else:
        prec_km, rec_km, _ = precision_recall_curve(y_va0, km_score0)
        prec_oc, rec_oc, _ = precision_recall_curve(y_va0, oc_score0)
        ax.plot(rec_km, prec_km,
                label=f'K-Means (k={best_k_roc}, AP={average_precision_score(y_va0, km_score0):.3f})',
                color='#2196F3', lw=2)
        ax.plot(rec_oc, prec_oc,
                label=f'One-Class SVM (AP={average_precision_score(y_va0, oc_score0):.3f})',
                color='#E91E63', lw=2)
        baseline = y_va0.mean()
        ax.axhline(baseline, color='gray', linestyle='--', lw=1,
                   label=f'Random (AP={baseline:.3f})')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle('비지도 이상탐지 비교 — K-Means 거리 vs One-Class SVM (Fold 0)',
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'anomaly_detection_roc.png', bbox_inches='tight')
plt.close()
print("Saved: anomaly_detection_roc.png")

# ── Figure 3: 이상 점수 분포 (양품 vs 불량) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

km_all_score = km_models[best_k_roc].transform(X_lab_sc).min(axis=1)
sc_all = StandardScaler()
X_lab_n = X_lab[y_lab == 0]
sc_all.fit(X_lab_n)
oc_all = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
oc_all.fit(sc_all.transform(X_lab_n))
oc_all_score = -oc_all.score_samples(sc_all.transform(X_lab))

titles = [f'K-Means (k={best_k_roc}) 이상 점수 분포', 'One-Class SVM 이상 점수 분포']
scores = [km_all_score, oc_all_score]

for ax, score, title in zip(axes, scores, titles):
    for cls, label, color in [(0, f'양품 (n={int((y_lab==0).sum())})', '#4C72B0'),
                               (1, f'불량 (n={int((y_lab==1).sum())})', '#DD8452')]:
        ax.hist(score[y_lab == cls], bins=60, alpha=0.65,
                label=label, color=color, density=True)
    ax.set_xlabel('이상 점수 (높을수록 불량 가능성 높음)', fontsize=10)
    ax.set_ylabel('밀도', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)

plt.suptitle('양품 vs 불량 이상 점수 분포 비교 (labeled_data 전체)', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'anomaly_score_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: anomaly_score_distribution.png")

# ── 최종 요약 출력 ────────────────────────────────────────────────────────────
print("\n=== 비지도 이상탐지 결과 요약 ===")
print(f"{'방법':<30} {'학습 데이터':<20} {'ROC-AUC':>10} {'PR-AUC':>10}")
print("-" * 75)
for _, row in km_summary.iterrows():
    print(f"K-Means (k={int(row.k):<2}){'':16} {'unlabeled 795K':<20} "
          f"{row.roc_mean:.4f}±{row.roc_std:.3f}  {row.pr_mean:.4f}±{row.pr_std:.3f}")
print(f"{'One-Class SVM (nu=0.02)':<30} {'labeled 양품 7,925':<20} "
      f"{ocsvm_roc:.4f}±{ocsvm_df.roc_auc.std():.3f}  "
      f"{ocsvm_pr:.4f}±{ocsvm_df.pr_auc.std():.3f}")
print(f"\n→ K-Means Best ROC-AUC: k={best_k_roc}")
print(f"→ K-Means Best PR-AUC : k={best_k_pr}")
