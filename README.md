# KAMP Injection Molding Defect Prediction — Step-by-Step Ablation + Unsupervised Anomaly Detection

> **NOVA50101 Introduction to Artificial Intelligence for Industrial AI**  
> Step-by-step ablation study on KAMP injection molding data, with unsupervised anomaly detection using unlabeled 795K samples.

---

## Project Overview

Using the KAMP injection molding AI dataset (KAIST·UNIST), we (1) systematically measure the contribution of each processing stage to defect prediction performance, and (2) explore unsupervised anomaly detection methods that exploit the large unlabeled dataset — neither of which is addressed in the official KAMP guidebook.

**Key originality vs. guidebook:**

| Item | Guidebook | Ours |
|---|---|---|
| Imbalance handling | Concept only (no code) | 4 methods × 10 combos, 5-fold ablation |
| Evaluation | accuracy + recall | ROC-AUC + PR-AUC + operating point |
| Model selection | AE → SVM → DNN, pick best | Systematic ablation across 6 linear / 4 ensemble / CNN / Stacking |
| SVM claim | SVM = best | QDA outperforms SVM on PR-AUC |
| Unlabeled 795K | AE (good-only) + pseudo-labeling | OC-SVM ROC 0.88 / K-Means ROC 0.47 + failure analysis |

---

## Results

### Phase 1–6 (Supervised Classification)

| Phase | Model | ROC-AUC | PR-AUC |
|---|---|---|---|
| Phase 2 (preprocessing) | LR-L2 + SMOTE | 0.9311 | 0.2413 |
| Phase 3 (linear) | **QDA** (reg=0.01) | 0.9344 | 0.3526 |
| Phase 4-A (dim. reduction) | TreeTop-15 + LR | 0.9380 | 0.2899 |
| **Phase 4-B (ensemble·NN)** | **MLP [256,128,64]** | **0.9497** | **0.4710** |
| Phase 5 (CNN) | CNN1D (32→64) | 0.9472 | 0.4501 |
| Phase 5 (Stacking) | meta-LR | 0.9462 | 0.4484 |
| Guidebook DNN (reproduced) | DNN [128,64,32] | 0.9468 | 0.4655 |

**Operating point (Precision ≥ 0.99):** MLP Recall = 32.4% (≈23 of 71 defects detected)

### Phase 7 (Unsupervised Anomaly Detection, 1st prototype)

| Method | Training data | ROC-AUC (5-fold) | PR-AUC |
|---|---|---|---|
| One-Class SVM (nu=0.02) | labeled good 7,925 | **0.8814 ± 0.027** | 0.1765 |
| K-Means (k=5, best) | unlabeled 795,315 | 0.4697 ± 0.063 | 0.1444 |
| Denoising AE | planned | — | — |

K-Means failure cause: unlabeled data spans many machines/products while labeled data is CN7/RG3 single-machine only → domain mismatch. Refinement (filter unlabeled to same machine/product) planned.

---

## Dataset

| Item | Value |
|---|---|
| Source | [KAMP Platform](https://www.kamp-ai.kr) — Injection Molding Machine AI Dataset |
| Total rows | 886,227 (8 CSVs) |
| Labeled (supervised) | `labeled_data.csv` 7,996 rows |
| Unlabeled | `unlabeled_data.csv` 795,315 rows |
| Effective features | 25 (removed 11 zero-variance + 1 ID) |
| Defect rate | 0.89% (71 / 7,996) |

> **Note:** `data/raw/` CSVs are not included (KAMP copyright). Download from [KAMP platform](https://www.kamp-ai.kr) and place under `data/raw/`.

---

## Directory Structure

```
IAAI_Term_project/
├── src/
│   ├── utils.py                # set_seed, setup_korean_font
│   ├── data.py                 # load_raw, get_fold, generate_splits
│   ├── preprocess.py           # scaler, resampler, fit_transform_fold
│   ├── evaluate.py             # ROC-AUC, PR-AUC, operating point
│   └── models/
│       ├── linear.py           # LR, LDA, QDA
│       ├── svm.py              # SVM-linear, SVM-RBF
│       ├── tree.py             # RF, AdaBoost, GBM
│       ├── nn.py               # MLP (PyTorch), CNN1D
│       └── stacking.py         # OOF stacking meta-learner
├── notebooks/
│   ├── 00_EDA.ipynb            # EDA + class distribution
│   ├── 01_preprocessing_ablation.ipynb
│   ├── 02_linear_baselines.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_ensemble_nn.ipynb
│   ├── 05_cnn_aux_stacking.ipynb
│   ├── 06_results_summary.ipynb
│   └── 07_anomaly_detection.ipynb  # OC-SVM / K-Means / AE
├── run_eda.py
├── run_eda_enhanced.py
├── run_dim_reduction.py
├── run_baseline_figures.py
├── run_phase4_figures.py
├── run_phase5.py
├── run_anomaly_detection.py    # OC-SVM + K-Means (Phase 7)
├── tests/                      # pytest unit tests
├── results/
│   ├── figures/                # NB{nn}_fig{n}_*.png + anomaly_*.png
│   ├── tables/                 # ablation CSVs + anomaly_detection_results.csv
│   ├── ablation_summary.md
│   └── decisions.md
├── docs/
│   ├── Proposal_NOVA50101.md       # ← Proposal submission draft
│   ├── Proposal_방향성_팀가이드.md  # internal team guide
│   ├── NOVA50101_Default_Final_Project_EN.md
│   ├── 04.-Guidebook_InjectionMolding_EN.md
│   └── TERM_PROJECT_GUIDELINE.md
├── reports/
│   ├── project_report_full.md
│   ├── presentation_outline.md
│   ├── notion_draft.md
│   └── 데이터셋_최종_결정_보고서_v3.md
├── data/
│   ├── raw/                    # ← .gitignore (download required)
│   └── splits/                 # 5-fold indices (fold_0~4.npy)
├── AGENT_INSTRUCTIONS.md
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

Download 8 CSVs from [KAMP platform](https://www.kamp-ai.kr) → "사출성형기 AI 데이터셋":

```
data/raw/labeled_data.csv
data/raw/unlabeled_data.csv
data/raw/moldset_labeled.csv
data/raw/supervised_label_cn7.csv
data/raw/moldset_labeled_cn7.csv
data/raw/moldset_unlabeled_cn7.csv
data/raw/moldset_labeled_rg3.csv
data/raw/moldset_unlabeled_rg3.csv
```

### 3. Run Phase 1–6

```bash
python run_eda.py
python run_eda_enhanced.py
python run_dim_reduction.py
python run_baseline_figures.py
python run_phase4_figures.py
python run_phase5.py
# or run notebooks/00~06 in order
```

### 4. Run anomaly detection (Phase 7)

```bash
python run_anomaly_detection.py
# Generates: results/figures/NB07_fig{1-3}_*.png
#            results/tables/anomaly_detection_results.csv
```

### 5. Tests

```bash
pytest tests/ -v
```

---

## Algorithms Used (lecture-covered only)

| Group | Algorithm | Lecture |
|---|---|---|
| Linear / Generative | LR (L1/L2), LDA, QDA | L4–6 |
| Margin | SVM (linear, RBF), **One-Class SVM** | L7–8 |
| Tree / Ensemble | RF, AdaBoost, GBM, Stacking | L7–10 |
| Clustering | **K-Means**, GMM, DBSCAN (planned) | L9–10 |
| Dim. Reduction | PCA, VarianceThreshold, TreeTop-K, **UMAP** | L9–11 |
| Neural Net | MLP (Dropout, ReLU, Adam), **Denoising AE** | L11–15 |
| Deep Learning | 1D-CNN (PyTorch) | L13–16 |
| Evaluation | ROC-AUC, PR-AUC, F1, operating point, Stratified 5-fold CV | L5–6 |

---

## License

Data: [KAMP platform terms of use](https://www.kamp-ai.kr)  
Code: MIT
