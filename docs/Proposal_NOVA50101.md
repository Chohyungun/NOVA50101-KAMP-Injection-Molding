# Final Project Proposal
## NOVA50101 Introduction to Artificial Intelligence for Industrial AI

---

**Team Number:** [팀 번호 — Blackboard 공지 확인 후 기재]  
**Team Members:** Hyungun Cho (Leader), [Member B], [Member C]  
**Project Title:** Defect Prediction in Injection Molding: Step-by-Step Ablation Study and Unsupervised Anomaly Detection on KAMP Manufacturing Data

---

## 1. Dataset Information and Baselines

### Dataset

| Item | Detail |
|---|---|
| **Name** | KAMP Injection Molding Machine AI Dataset |
| **Source** | KAMP Platform (https://www.kamp-ai.kr) — KAIST · UNIST · EPM Solutions Co., Ltd. |
| **Kaggle URL** | Not available on Kaggle (KAMP platform only) |
| **Size** | 886,227 total rows across 8 CSV files |
| **Labeled data** | `labeled_data.csv` — 7,996 rows × 45 columns (9 meta + 36 process variables) |
| **Unlabeled data** | `unlabeled_data.csv` — 795,315 rows (no per-item label) |
| **Effective features** | 25 (removed 11 zero-variance + 1 ID column via EDA) |
| **Target** | `PassOrFail` — 0: defect (71 items, 0.89%), 1: good (7,925 items, 99.11%) |
| **Type** | Tabular, manufacturing process variables (temperature, pressure, speed, position, time) |

> Since no public code is available for this dataset, baselines' URLs are not submitted per the assignment guidelines. Instead, we use the **KAMP Official Guidebook** (KAIST·UNIST, 2020, §2.2–2.3) as our primary comparison baseline, which implements Denoising AutoEncoder, SVM, and DNN on this exact dataset.

### Baseline Comparison (Guidebook, §2.3)

| Model | ROC-AUC | PR-AUC | Note |
|---|---|---|---|
| Guidebook DNN [32→64→32] | 0.9468 | 0.4655 | Reproduced exactly |
| Guidebook SVM (claimed "best") | N/A | N/A | Phase 3 in our study |

---

## 2. Project Description and Approach

### 2.1 Problem and Motivation

Injection molding defect prediction is a critical quality-control task in automotive manufacturing. The KAMP guidebook addresses this by sequentially training an AutoEncoder, SVM, and DNN, then selecting the single best performer. This approach has two limitations that motivate our work: (1) it does not isolate which processing step contributes most to performance improvement, and (2) it leaves 795,315 unlabeled samples (89.8% of the total dataset) entirely unused. The extreme class imbalance (0.89% defect rate) makes the task additionally challenging.

### 2.2 Initial Experimental Results (Phase 1–6, Completed)

We have completed a six-phase ablation study using only the 7,996 labeled samples. All experiments use Stratified 5-fold cross-validation with random_state=42, reporting ROC-AUC and PR-AUC as primary metrics.

**Phase 2 — Preprocessing Ablation (2 scalers × 5 imbalance methods = 10 combinations):**  
StandardScaler + SMOTE achieved the best ROC-AUC (0.9311), improving over no resampling (+0.024). Critically, SMOTE halved the PR-AUC standard deviation (0.1145 → 0.0554), confirming its role in stabilizing results under extreme imbalance — a dimension entirely absent from the guidebook.

**Phase 3 — Linear Baselines (LR-L1/L2, LDA, QDA, SVM-linear, SVM-RBF):**  
QDA (reg=0.01) achieved ROC-AUC 0.9344 and PR-AUC 0.3526, outperforming the guidebook's claimed "best" SVM-RBF (PR-AUC 0.0890) by a factor of 4. Systematic search revealed a strong model that the guidebook's sequential approach missed entirely.

**Phase 4 — Dimensionality Reduction + Ensemble·NN:**  
MLP[256,128,64]+Dropout(0.3) achieved ROC-AUC 0.9497 and PR-AUC 0.4710, exceeding the reproduced guidebook DNN (0.9468 / 0.4655). The non-linear model transition from QDA to MLP contributed the single largest performance gain: PR-AUC +0.119. PCA hurt performance on this dataset, confirming that defect patterns are not well-captured by linear principal components.

**Phase 5 — 1D-CNN and Stacking:**  
CNN1D(32,64) achieved ROC-AUC 0.9472. Stacking (meta-LR over LR+QDA+RF+GBM+MLP) reached ROC-AUC 0.9462 — failing to exceed the single best model (MLP 0.9497), demonstrating that ensemble diversity, not merely combination, drives stacking gains.

**Phase 6 — Operating Point Analysis:**  
At Precision ≥ 0.99, MLP detects 23 of 71 defects (Recall 32.4%). This defines the practical deployment boundary and reveals that detecting the remaining 48 defects requires either more labeled defect data or alternative modeling approaches.

**Step-by-step contribution summary:**

| Phase | Change | Contribution |
|---|---|---|
| Phase 2 | None → SMOTE | ROC-AUC +0.024, PR-AUC std halved |
| Phase 3 | LR → QDA | PR-AUC +0.084 |
| Phase 4 | QDA → MLP | PR-AUC +0.119 (largest gain) |
| Phase 5 | Single → Stacking | PR-AUC −0.023 (counterexample) |

**Anomaly detection prototype (1st experiment, 2026-05-16):**  
One-Class SVM trained on labeled good samples (7,925 rows) achieved ROC-AUC 0.8814 without any defect labels during training. K-Means (k=5, trained on unlabeled 795K) achieved ROC-AUC 0.4697 — near random — because the unlabeled data spans many machines and products while labeled data is product-specific (CN7/RG3, single machine). This domain mismatch is the key finding motivating Method C's refinement.

### 2.3 Proposed Methods (Future Work)

**Method B — Autoencoder-based Anomaly Detection (Guidebook §2.2.1 reproduction + extension)**

The guidebook trains a Denoising AutoEncoder on labeled good samples only, using reconstruction error as the anomaly score with a 3–5σ threshold. We propose two extensions: (1) training the AE jointly on labeled good samples and all unlabeled 795K rows to build richer normal-pattern representations (no label leakage, as AE requires no labels), and (2) replacing the σ-heuristic threshold with an operating-point-based threshold selected from the labeled validation fold's PR curve. We compare One-Class SVM (already implemented, ROC 0.8814), AE (labeled-only), and AE (labeled + unlabeled) across both threshold strategies — a 3×2 ablation not present in the guidebook.

*Lecture connections: One-Class SVM (L7-8), Denoising AE / MLP+Dropout (L11-14)*

**Method C — K-Means Cluster-Distance Anomaly Detection (Novel)**

The guidebook describes distance- and density-based anomaly scores as valid unsupervised approaches (§2.2.1) but implements only AE. We propose using K-Means centroid distance as the anomaly score: train K-Means on unlabeled data (no labels needed) and assign each sample's minimum centroid distance as its anomaly score. The first prototype (K-Means on all 795K, k=5–50) produced ROC-AUC 0.47 due to domain mismatch. The refined experiment will filter unlabeled data to the same machine and product type as the test set before K-Means training. This addresses a gap in the guidebook and directly compares with Method B's reconstruction-error scores.

*Lecture connections: K-Means (L9-10) / Future extensions: DBSCAN, GMM, UMAP (L9-11)*

---

## 3. Limitations and Future Work

1. **K-Means domain mismatch** — Unlabeled data filtered to matching machine/product will be re-evaluated.
2. **Denoising AE not yet implemented** — Guidebook §2.2.1 reproduction with unlabeled expansion is planned.
3. **Temporal structure unused** — Process drift visible in EDA (anomalous batch near 2020-10-27) not yet modeled via sliding window CNN.
4. **CN7/RG3 not separated** — KS test confirmed 22/24 variables differ significantly between products; separate models not attempted.
5. **Semi-supervised pseudo-labeling** — Guidebook §2.2.2 (pseudo-labeling + entropy minimization) not yet reproduced; planned as future work.

---

## 4. Team Member Roles

| Member | Phase | Primary Contribution |
|---|---|---|
| **Hyungun Cho** (Leader) | 0, 2, 5, 6, 7 | Project design, preprocessing ablation, Stacking, anomaly detection (OC-SVM, K-Means), Proposal |
| [Member B] | 1, 3 | EDA visualization, linear baselines (LR, LDA, QDA, SVM) |
| [Member C] | 4, 5 | Ensemble (RF, GBM, AdaBoost), MLP+Dropout, 1D-CNN, guidebook DNN reproduction |

*(If 4-member team, Phase 4-A and Phase 7 AE implementation are split between Members B and C.)*

---

*Submitted by: Hyungun Cho (Team Leader) | NOVA50101 | 2026-05-17*
