"""Phase 6 — 전 Phase 통합 결과 시각화 및 요약 스크립트.

생성 파일:
  results/figures/phase6_all_phases_roc.png
  results/figures/phase6_all_phases_pr.png
  results/figures/phase6_improvement_trajectory.png
  results/figures/phase6_final_operating_point.png
  results/tables/phase6_unified_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import set_seed, setup_korean_font  # noqa: E402

set_seed(42)
setup_korean_font()

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. 상수 정의
# ---------------------------------------------------------------------------
GUIDEBOOK_ROC = 0.9468
GUIDEBOOK_PR  = 0.4655

# ---------------------------------------------------------------------------
# 1. 통합 결과표 (phase6_unified_results.csv) 생성
# ---------------------------------------------------------------------------

unified_rows = [
    # Phase 2 (전처리 ablation best)
    {
        "Phase": "Phase 2",
        "모델": "LR-L2",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9311,
        "ROC-AUC(std)": 0.0107,
        "PR-AUC(mean)": 0.2413,
        "PR-AUC(std)": 0.0554,
        "가이드북비교": "미비교 (전처리 ablation 단계)",
        "비고": "불균형 처리 +0.024 ROC vs none",
    },
    # Phase 3 — QDA best
    {
        "Phase": "Phase 3",
        "모델": "QDA(reg=0.01)",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9344,
        "ROC-AUC(std)": 0.0175,
        "PR-AUC(mean)": 0.3526,
        "PR-AUC(std)": 0.1026,
        "가이드북비교": "가이드북 미시도 모델이 SVM 초과",
        "비고": "선형·생성적 모델 최고",
    },
    # Phase 3 — SVM-linear (가이드북 비교)
    {
        "Phase": "Phase 3",
        "모델": "SVM-linear(C=10)",
        "전처리": "StandardScaler",
        "ROC-AUC(mean)": 0.9074,
        "ROC-AUC(std)": 0.0225,
        "PR-AUC(mean)": 0.3434,
        "PR-AUC(std)": 0.1094,
        "가이드북비교": "가이드북 §2.3 SVM best",
        "비고": "QDA 대비 ROC -0.027",
    },
    # Phase 4-A — 차원축소 best
    {
        "Phase": "Phase 4-A",
        "모델": "LR-L2+TreeTop-15",
        "전처리": "StandardScaler+SMOTE+TreeTop15",
        "ROC-AUC(mean)": 0.9380,
        "ROC-AUC(std)": 0.0204,
        "PR-AUC(mean)": 0.2899,
        "PR-AUC(std)": 0.0806,
        "가이드북비교": "가이드북 미포함 단계",
        "비고": "full(25) 대비 ROC +0.004",
    },
    # Phase 4-B — MLP best
    {
        "Phase": "Phase 4-B",
        "모델": "MLP[256,128,64]+Dropout",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9497,
        "ROC-AUC(std)": 0.0166,
        "PR-AUC(mean)": 0.4710,
        "PR-AUC(std)": 0.0968,
        "가이드북비교": "가이드북 DNN 초과 +0.0029 ROC / +0.0055 PR",
        "비고": "전 Phase 최고 모델",
    },
    # Phase 4-B — Guidebook DNN
    {
        "Phase": "Phase 4-B",
        "모델": "Guidebook DNN(§2.3 재현)",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": GUIDEBOOK_ROC,
        "ROC-AUC(std)": 0.0164,
        "PR-AUC(mean)": GUIDEBOOK_PR,
        "PR-AUC(std)": 0.1269,
        "가이드북비교": "가이드북 DNN 기준선",
        "비고": "Dropout 없음",
    },
    # Phase 4-B — RandomForest
    {
        "Phase": "Phase 4-B",
        "모델": "RandomForest(n=500)",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9478,
        "ROC-AUC(std)": 0.0123,
        "PR-AUC(mean)": 0.4481,
        "PR-AUC(std)": 0.1121,
        "가이드북비교": "가이드북 미포함",
        "비고": "앙상블 2위",
    },
    # Phase 5 — CNN1D
    {
        "Phase": "Phase 5",
        "모델": "CNN1D(32,64)",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9472,
        "ROC-AUC(std)": 0.0152,
        "PR-AUC(mean)": 0.4501,
        "PR-AUC(std)": 0.1194,
        "가이드북비교": "가이드북 미포함",
        "비고": "1D-CNN 최고",
    },
    # Phase 5 — Stacking
    {
        "Phase": "Phase 5",
        "모델": "Stacking(LR-meta)",
        "전처리": "StandardScaler+SMOTE",
        "ROC-AUC(mean)": 0.9462,
        "ROC-AUC(std)": 0.0108,
        "PR-AUC(mean)": 0.4484,
        "PR-AUC(std)": 0.1021,
        "가이드북비교": "가이드북 미포함",
        "비고": "단일 MLP 미초과 — 앙상블 반례",
    },
]

df_unified = pd.DataFrame(unified_rows)
df_unified["ROC-AUC(mean±std)"] = df_unified.apply(
    lambda r: f"{r['ROC-AUC(mean)']:.4f}±{r['ROC-AUC(std)']:.4f}", axis=1
)
df_unified["PR-AUC(mean±std)"] = df_unified.apply(
    lambda r: f"{r['PR-AUC(mean)']:.4f}±{r['PR-AUC(std)']:.4f}", axis=1
)

save_cols = ["Phase", "모델", "전처리", "ROC-AUC(mean±std)", "PR-AUC(mean±std)", "가이드북비교", "비고"]
df_unified[save_cols].to_csv(TABLES_DIR / "phase6_unified_results.csv", index=False, encoding="utf-8-sig")
print("[OK] phase6_unified_results.csv 저장 완료")

# ---------------------------------------------------------------------------
# 2. Figure A-1: 전 Phase ROC-AUC 막대 그래프
# ---------------------------------------------------------------------------

phase_labels = [
    "Ph2\nLR+SMOTE",
    "Ph3\nQDA",
    "Ph3\nSVM-linear\n(가이드북)",
    "Ph4A\nLR+Tree15",
    "Ph4B\nRF",
    "Ph4B\nMLP[256,128,64]",
    "Ph4B\nGuidebook DNN",
    "Ph5\nCNN1D(32,64)",
    "Ph5\nStacking",
]

roc_means = [0.9311, 0.9344, 0.9074, 0.9380, 0.9478, 0.9497, 0.9468, 0.9472, 0.9462]
roc_stds  = [0.0107, 0.0175, 0.0225, 0.0204, 0.0123, 0.0166, 0.0164, 0.0152, 0.0108]

is_guidebook = [False, False, True, False, False, False, True, False, False]

colors = ["#4C72B0" if not g else "#C44E52" for g in is_guidebook]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(phase_labels))
bars = ax.bar(x, roc_means, yerr=roc_stds, color=colors, capsize=4, width=0.6, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(phase_labels, fontsize=9)
ax.set_ylim(0.85, 0.975)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("전 Phase 통합 ROC-AUC 비교", fontsize=14)
ax.axhline(GUIDEBOOK_ROC, color="#C44E52", linestyle="--", linewidth=1.2, label=f"가이드북 DNN ({GUIDEBOOK_ROC:.4f})")
ax.axhline(0.9497, color="#2ca02c", linestyle=":", linewidth=1.2, label="최고 모델 MLP (0.9497)")

patch_our = mpatches.Patch(color="#4C72B0", label="우리 모델")
patch_gb  = mpatches.Patch(color="#C44E52", label="가이드북 관련")
ax.legend(handles=[patch_our, patch_gb,
                   plt.Line2D([0], [0], color="#C44E52", linestyle="--", label=f"가이드북 DNN ({GUIDEBOOK_ROC:.4f})"),
                   plt.Line2D([0], [0], color="#2ca02c", linestyle=":", label="최고 MLP (0.9497)")],
          fontsize=9, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "phase6_all_phases_roc.png", dpi=120)
plt.close(fig)
print("[OK] phase6_all_phases_roc.png 저장 완료")

# ---------------------------------------------------------------------------
# 3. Figure A-2: 전 Phase PR-AUC 막대 그래프
# ---------------------------------------------------------------------------

pr_means = [0.2413, 0.3526, 0.3434, 0.2899, 0.4481, 0.4710, 0.4655, 0.4501, 0.4484]
pr_stds  = [0.0554, 0.1026, 0.1094, 0.0806, 0.1121, 0.0968, 0.1269, 0.1194, 0.1021]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, pr_means, yerr=pr_stds, color=colors, capsize=4, width=0.6, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(phase_labels, fontsize=9)
ax.set_ylim(0.0, 0.65)
ax.set_ylabel("PR-AUC", fontsize=12)
ax.set_title("전 Phase 통합 PR-AUC 비교", fontsize=14)
ax.axhline(GUIDEBOOK_PR, color="#C44E52", linestyle="--", linewidth=1.2, label=f"가이드북 DNN ({GUIDEBOOK_PR:.4f})")
ax.axhline(0.4710, color="#2ca02c", linestyle=":", linewidth=1.2, label="최고 MLP (0.4710)")

ax.legend(handles=[patch_our, patch_gb,
                   plt.Line2D([0], [0], color="#C44E52", linestyle="--", label=f"가이드북 DNN ({GUIDEBOOK_PR:.4f})"),
                   plt.Line2D([0], [0], color="#2ca02c", linestyle=":", label="최고 MLP (0.4710)")],
          fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "phase6_all_phases_pr.png", dpi=120)
plt.close(fig)
print("[OK] phase6_all_phases_pr.png 저장 완료")

# ---------------------------------------------------------------------------
# 4. Figure B: Phase별 PR-AUC 향상 누적 라인 차트
# ---------------------------------------------------------------------------

trajectory_phases = ["Phase 2\n(전처리)", "Phase 3\n(선형·QDA)", "Phase 4\n(앙상블·NN)", "Phase 5\n(CNN+Stacking)"]
trajectory_pr     = [0.2413, 0.3526, 0.4710, 0.4501]  # best-of-phase PR-AUC
trajectory_models = ["LR+SMOTE", "QDA", "MLP[256,128,64]", "CNN1D(32,64)"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(trajectory_phases, trajectory_pr, marker="o", markersize=9,
        linewidth=2.2, color="#4C72B0", label="Phase별 최고 PR-AUC")

for i, (p, v, m) in enumerate(zip(trajectory_phases, trajectory_pr, trajectory_models)):
    ax.annotate(f"{v:.4f}\n({m})", xy=(i, v),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

ax.axhline(GUIDEBOOK_PR, color="#C44E52", linestyle="--", linewidth=1.5,
           label=f"가이드북 DNN PR-AUC = {GUIDEBOOK_PR:.4f}")
ax.fill_between([0, 1, 2, 3], trajectory_pr, GUIDEBOOK_PR,
                where=[v > GUIDEBOOK_PR for v in trajectory_pr],
                alpha=0.12, color="#2ca02c", label="가이드북 초과 영역")

ax.set_ylim(0.15, 0.60)
ax.set_ylabel("PR-AUC", fontsize=12)
ax.set_title("Phase별 PR-AUC 향상 궤적", fontsize=14)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "phase6_improvement_trajectory.png", dpi=120)
plt.close(fig)
print("[OK] phase6_improvement_trajectory.png 저장 완료")

# ---------------------------------------------------------------------------
# 5. Figure C: 최종 MLP 운용점 PR 곡선 (OOF 예측 재계산)
# ---------------------------------------------------------------------------
print("[INFO] MLP OOF 예측 계산 시작 (5-fold)...")

from data import load_raw, get_fold  # noqa: E402
from preprocess import get_feature_cols, fit_transform_fold  # noqa: E402
from models.nn import MLPTrainer  # noqa: E402

try:
    df_raw = load_raw("labeled_data")
    feature_cols = get_feature_cols(df_raw)
    X_all = df_raw[feature_cols].values.astype(np.float32)
    y_all = df_raw["PassOrFail"].values.astype(np.float32)

    oof_probs = np.zeros(len(y_all))
    oof_labels = np.zeros(len(y_all))

    for fold in range(5):
        X_tr, X_va, y_tr, y_va = get_fold(fold, X_all, y_all)
        X_tr_s, X_va_s, y_tr_s = fit_transform_fold(X_tr, X_va, y_tr, "standard", "smote")

        trainer = MLPTrainer(
            hidden_dims=(256, 128, 64),
            dropout=0.3,
            lr=1e-3,
            epochs=30,
            batch_size=256,
            random_state=42,
        )
        trainer.fit(X_tr_s, y_tr_s)
        proba = trainer.predict_proba(X_va_s)[:, 1]

        # 검증 인덱스 복원
        fold_path = PROJECT_ROOT / "data" / "splits" / f"fold_{fold}.npy"
        indices = np.load(str(fold_path), allow_pickle=True).item()
        va_idx = indices["val"]
        oof_probs[va_idx]  = proba
        oof_labels[va_idx] = y_va

    roc_oof = roc_auc_score(oof_labels, oof_probs)
    pr_oof  = average_precision_score(oof_labels, oof_probs)
    print(f"[INFO] OOF ROC-AUC={roc_oof:.4f}, PR-AUC={pr_oof:.4f}")

    prec_arr, rec_arr, thr_arr = precision_recall_curve(oof_labels, oof_probs)

    # 운용점 탐색
    def find_op(prec_arr, rec_arr, thr_arr, target_prec):
        valid = np.where(prec_arr >= target_prec)[0]
        if len(valid) == 0:
            return None, None, None
        idx = valid[np.argmax(rec_arr[valid])]
        thr_val = thr_arr[idx] if idx < len(thr_arr) else thr_arr[-1]
        return prec_arr[idx], rec_arr[idx], thr_val

    p95, r95, t95 = find_op(prec_arr, rec_arr, thr_arr, 0.95)
    p99, r99, t99 = find_op(prec_arr, rec_arr, thr_arr, 0.99)

    n_defects = int(oof_labels.sum())  # 71
    detected_99 = int(round(r99 * n_defects)) if r99 is not None else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, color="#4C72B0", linewidth=2, label=f"PR 곡선 (OOF AUC={pr_oof:.4f})")
    ax.axhline(GUIDEBOOK_PR, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"가이드북 DNN PR-AUC = {GUIDEBOOK_PR:.4f}")

    if p95 is not None:
        ax.scatter(r95, p95, color="#e67e22", zorder=5, s=100,
                   label=f"Prec≥0.95 운용점 (Recall={r95:.3f})")
        ax.annotate(f"Prec={p95:.2f}\nRecall={r95:.3f}",
                    xy=(r95, p95), xytext=(r95 + 0.05, p95 - 0.08),
                    fontsize=8.5, color="#e67e22",
                    arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1))

    if p99 is not None:
        ax.scatter(r99, p99, color="#c0392b", zorder=5, s=120, marker="*",
                   label=f"Prec≥0.99 운용점 (Recall={r99:.3f})")
        ax.annotate(f"Prec={p99:.2f}\nRecall={r99:.3f}",
                    xy=(r99, p99), xytext=(r99 + 0.05, p99 + 0.05),
                    fontsize=8.5, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
        ax.text(0.50, 0.30,
                f"Prec=0.99 운용 시\n불량 {n_defects}건 중 {detected_99}건 탐지",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7", alpha=0.85))

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("최종 모델(MLP) OOF Precision-Recall 곡선 및 운용점", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase6_final_operating_point.png", dpi=120)
    plt.close(fig)
    print("[OK] phase6_final_operating_point.png 저장 완료")

except Exception as e:
    print(f"[WARN] OOF 재계산 실패: {e}")
    print("[WARN] 저장된 수치로 대체 PR 곡선 생성...")

    # 저장 수치로 대체 시각화
    fig, ax = plt.subplots(figsize=(8, 6))

    # 운용점 2개 표시 (수치 기반)
    op_pts = {
        "Prec≥0.95": (0.338, 0.960, "#e67e22"),
        "Prec≥0.99": (0.324, 1.000, "#c0392b"),
    }
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("최종 모델(MLP) 운용점 분석", fontsize=13)

    for label, (rec, prec, col) in op_pts.items():
        ax.scatter(rec, prec, color=col, zorder=5, s=140, marker="*", label=f"{label}: Recall={rec:.3f}")
        ax.annotate(f"Prec={prec:.2f}\nRecall={rec:.3f}",
                    xy=(rec, prec), xytext=(rec - 0.18, prec - 0.12),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col))

    ax.axhline(GUIDEBOOK_PR, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"가이드북 DNN PR-AUC = {GUIDEBOOK_PR:.4f}")
    ax.text(0.45, 0.25,
            "Prec=0.99 운용 시\n불량 71건 중 23건 탐지",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7", alpha=0.85))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.1])
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase6_final_operating_point.png", dpi=120)
    plt.close(fig)
    print("[OK] phase6_final_operating_point.png (대체) 저장 완료")

# ---------------------------------------------------------------------------
# 6. 최종 요약 출력
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Phase 6 요약 스크립트 완료")
print("=" * 60)
print(f"  통합 결과표 : {TABLES_DIR / 'phase6_unified_results.csv'}")
print(f"  ROC 비교도  : {FIGURES_DIR / 'phase6_all_phases_roc.png'}")
print(f"  PR 비교도   : {FIGURES_DIR / 'phase6_all_phases_pr.png'}")
print(f"  향상 궤적   : {FIGURES_DIR / 'phase6_improvement_trajectory.png'}")
print(f"  운용점 PR   : {FIGURES_DIR / 'phase6_final_operating_point.png'}")
print("=" * 60)
