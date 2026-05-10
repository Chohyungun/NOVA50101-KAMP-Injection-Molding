# KAMP 사출성형 공정 불량 예측 — 단계별 Ablation 분석

> **NOVA50101 인공지능학개론 텀프로젝트**  
> KAMP 사출성형기 AI 데이터셋으로 공정 불량 예측 모델을 개발하고 단계별 Ablation을 수행했다.

---

## 프로젝트 개요

KAIST·UNIST 공동 가이드북(§2.3)의 "AE→SVM→DNN 순차 학습 후 best 선택" 방식 대신,
**각 처리 단계를 분리해 어느 단계가 성능을 좌우하는지 정량적으로 측정**하는 것이 핵심 차별점이다.

### 데이터셋

| 항목 | 값 |
|---|---|
| 출처 | [KAMP 플랫폼](https://www.kamp-ai.kr) — 사출성형기 AI 데이터셋 |
| 총 행 수 | 886,227행 (8개 CSV 합산) |
| 주 지도학습 데이터 | `labeled_data.csv` 7,996행 |
| 유효 변수 | 25개 (분산=0 제거 포함) |
| 불량률 | 0.89% (71/7,996) — 극심한 클래스 불균형 |

> **주의:** `data/raw/` 원본 CSV는 KAMP 저작권으로 인해 포함되지 않는다.  
> [KAMP 플랫폼](https://www.kamp-ai.kr)에서 "사출성형기 AI 데이터셋"을 직접 다운로드할 것.

---

## 핵심 결과

| Phase | 대표 모델 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Phase 2 (전처리 기준선) | LR-L2 + SMOTE | 0.9311 | 0.2413 |
| Phase 3 (선형 베이스라인) | **QDA** (reg=0.01) | 0.9344 | 0.3526 |
| Phase 4-A (차원축소) | TreeTop-15 + LR | 0.9380 | 0.2899 |
| **Phase 4-B (앙상블·NN)** | **MLP [256,128,64]** | **0.9497** | **0.4710** |
| Phase 5 (CNN) | CNN1D (32→64) | 0.9472 | 0.4501 |
| Phase 5 (Stacking) | meta-LR | 0.9462 | 0.4484 |
| 가이드북 DNN 재현 | DNN [128,64,32] | 0.9468 | 0.4655 |

**운용점 (Precision ≥ 0.99):** MLP Recall = 32.4% (불량 71건 중 약 23건 탐지)

---

## 디렉토리 구조

```
IAAI_Term_project/
├── src/                        # 재사용 모듈
│   ├── utils.py                # set_seed, setup_korean_font
│   ├── data.py                 # load_raw, get_fold, generate_splits
│   ├── preprocess.py           # scaler, resampler, fit_transform_fold
│   ├── evaluate.py             # ROC-AUC, PR-AUC, 운용점 분석
│   └── models/
│       ├── linear.py           # LR, LDA, QDA
│       ├── svm.py              # SVM-linear, SVM-RBF
│       ├── tree.py             # RF, AdaBoost, GBM
│       ├── nn.py               # MLP (PyTorch), CNN1D
│       └── stacking.py         # OOF 스태킹 메타학습기
├── notebooks/
│   ├── 00_EDA.ipynb            # Phase 1+2: 탐색적 데이터 분석
│   ├── 01_preprocessing_ablation.ipynb
│   ├── 02_linear_baselines.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_ensemble_nn.ipynb
│   ├── 05_cnn_aux_stacking.ipynb
│   └── 06_results_summary.ipynb  # 전 Phase 통합 결과
├── tests/                      # pytest 단위 테스트 (29개)
├── results/
│   ├── figures/                # 분석 figure (PNG)
│   ├── tables/                 # ablation 결과 CSV
│   ├── ablation_summary.md     # Phase별 결과 누적
│   └── decisions.md            # 기술 결정 로그 (D-001~D-013)
├── reports/
│   ├── html/                   # 노트북 HTML 변환본 (00~06)
│   ├── project_report_full.md
│   ├── final_report_outline.md
│   ├── limitation_section.md
│   └── presentation_outline.md
├── data/
│   ├── raw/                    # ← .gitignore (직접 다운로드 필요)
│   └── splits/                 # 5-fold 인덱스 (fold_0~4.npy)
├── run_eda.py                  # EDA 파이프라인
├── run_preproc_ablation.py     # Phase 2
├── run_linear_baselines.py     # Phase 3
├── run_dim_reduction.py        # Phase 4-A
├── run_ensemble_nn.py          # Phase 4-B
├── run_phase5.py               # Phase 5
├── run_phase6_summary.py       # Phase 6 통합
├── requirements.txt
└── AGENT_INSTRUCTIONS.md       # 개발 에이전트 지침
```

---

## 실행 방법

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

[KAMP 플랫폼](https://www.kamp-ai.kr)에서 사출성형기 AI 데이터셋 8개 CSV를 다운로드 후:

```
data/raw/labeled_data.csv
data/raw/moldset_labeled.csv
data/raw/unlabeled_data.csv
data/raw/supervised_label_cn7.csv
data/raw/moldset_labeled_cn7.csv
data/raw/moldset_unlabeled_cn7.csv
data/raw/moldset_labeled_rg3.csv
data/raw/moldset_unlabeled_rg3.csv
```

### 3. 순서대로 실행

```bash
python run_eda.py                  # EDA + 5-fold split 생성
python run_eda_enhanced.py         # EDA 심화 (Box plot, 이상치, 드리프트)
python run_preproc_ablation.py     # Phase 2: 전처리 10종 비교
python run_linear_baselines.py     # Phase 3: 선형 모델 6종
python run_dim_reduction.py        # Phase 4-A: 차원축소 9종
python run_ensemble_nn.py          # Phase 4-B: 앙상블·NN
python run_phase5.py               # Phase 5: CNN + Stacking
python run_phase6_summary.py       # Phase 6: 전체 통합 요약
```

### 4. 테스트

```bash
pytest tests/ -v
# 29 passed
```

---

## 사용 알고리즘 (강의 범위 내)

| 그룹 | 알고리즘 |
|---|---|
| 선형·생성적 | Logistic Regression (L1/L2), LDA, QDA |
| 거리·마진 | SVM (linear, RBF) |
| 트리·앙상블 | Random Forest, AdaBoost, Gradient Boosting |
| 메타 학습 | Stacking (OOF 메타학습기) |
| 차원 축소 | PCA, VarianceThreshold, TreeImportance Top-K |
| 신경망 | MLP (Dropout, ReLU, Adam), 1D-CNN (PyTorch) |
| 평가 | ROC-AUC, PR-AUC, F1, 운용점 (Precision-fixed Recall) |

---

## 라이선스

데이터: [KAMP 플랫폼 이용약관](https://www.kamp-ai.kr) 준수  
코드: MIT
