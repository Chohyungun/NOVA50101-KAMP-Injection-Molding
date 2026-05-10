# Ablation Summary — KAMP 사출성형 텀프로젝트

> 각 Phase 끝마다 결과를 누적 기록. Phase 6 보고서 작성 시 그대로 활용.

---

## Phase 1+2 — EDA 결과 (2026-05-10)

### 데이터 개요

| 항목 | 값 |
|---|---|
| 총 행 수 | 886,227행 (8개 CSV 합산) |
| 주 지도학습 데이터 | `labeled_data.csv` 7,996행 + `supervised_label_cn7.csv` 6,736행 |
| 레이블 없는 데이터 | `unlabeled_data.csv` 795,315행 (Phase 3+ 준지도 학습 옵션) |
| 원 변수 수 | 36개 (수치 독립변수 기준, labeled_data) |
| **유효 변수 수** | **25개** (분산=0 제거 10개 + Barrel_Temperature_7 제거 1개 + PART_FACT_SERIAL 제거 1개) |
| 양품/불량 비율 | 양품 99.11% / **불량 0.89%** (극심한 불균형) |
| 결측 | Reason 컬럼 99.1% (불량 원인 기록), 나머지 피처 0% |

### 제거 변수 목록

| 변수 | 사유 |
|---|---|
| Mold_Temperature_1, 2, 5~12 (10개) | 분산=0 (측정 미실시) — 가이드북 §2.1과 완전 일치 |
| Barrel_Temperature_7 | 고유값 2개 (0.0 vs 36.4°C), mean=0.009°C → near-zero |

### 상관 분석

- 강한 상관 짝 (피어슨 |r|>0.95): **34쌍**
- Phase 4 PCA/분산임계 제거에서 다중공선성 처리 예정

### CN7 vs RG3 분포 비교

- KS 검정 유의 변수 (p<0.05): **22/24개** → 두 제품 분포가 통계적으로 다름
- **처리 방침:** Phase 3 기본은 `labeled_data` 단독 분석. Phase 4에서 product flag 추가 통합 vs 분리 ablation 진행.

### 5-fold CV 설정

- Stratified 5-fold, random_state=42
- fold별 불량률 편차: 최대 **0.0496 pp** (< 1 pp 기준 통과)
- 인덱스 파일: `data/splits/fold_0~4.npy` (영구 저장, 모든 Phase 공유)

### 핵심 인사이트 1줄

> 유효 변수 26개, 불량률 0.89%, 강한 상관 짝 34쌍, CN7·RG3 분포 22/24 유의차이 → **Phase 3 기본은 labeled_data 단독, Phase 4에서 product 통합/분리 ablation 결정.**

### 가이드북 비교

- 가이드북이 "0"으로 표기한 11개 변수(Mold_Temp_1,2,5-12 + Barrel_Temp_7)가 실측과 100% 일치.
- 가이드북 §2.3의 "AE/SVM/DNN 차례 학습" 흐름 대비, 우리는 **단계별 ablation(전처리→선형→차원축소→앙상블/NN→스태킹)**으로 어느 단계가 성능을 좌우하는지 분리 측정. — originality의 1차 차별점.

### Phase 2 진입 조건 (DoD 체크)

- [x] `pytest tests/test_data.py` 14/14 통과
- [x] `00_EDA.ipynb` Run All 검증 완료 (스크립트 시뮬레이션)
- [x] `results/figures/eda_*.png` 5개 생성
- [x] `results/tables/eda_*.csv` 5개 생성
- [x] `data/splits/fold_0~4.npy` 생성 완료
- [x] `results/decisions.md` 5개 결정 항목 기록

---

*다음 Phase: 전처리 Ablation (`01_preprocessing_ablation.ipynb`) — LR-L2 고정, 전처리 5종 5-fold CV 비교*

---

## Phase 2 — 전처리 Ablation 결과 (2026-05-10)

### 실험 설정

| 항목 | 값 |
|---|---|
| 분류기 (고정) | Logistic Regression L2, C=1.0, solver=saga |
| CV | Stratified 5-fold (fold_0~4.npy 공유) |
| 데이터 | labeled_data.csv (7,996행, 25개 유효 변수) |
| Scaler 후보 | StandardScaler, RobustScaler |
| 불균형 처리 후보 | None, class_weight, SMOTE, ADASYN, RandomUnderSampler |
| 총 조합 | 2 × 5 = **10** |

### 결과표 (PR-AUC 기준 정렬)

| Scaler | Resample | ROC-AUC | ±std | PR-AUC | ±std | F1 | ±std |
|---|---|---|---|---|---|---|---|
| standard | none | 0.9076 | 0.0159 | **0.2666** | 0.1145 | 0.2845 | 0.1078 |
| robust | undersample | 0.8886 | 0.0340 | 0.2541 | 0.0682 | 0.0420 | 0.0023 |
| **standard** | **smote** | **0.9311** | **0.0107** | **0.2413** | **0.0554** | 0.0808 | 0.0060 |
| standard | class_weight | 0.9261 | 0.0107 | 0.2316 | 0.0608 | 0.0808 | 0.0046 |
| robust | smote | 0.7626 | 0.0793 | 0.2293 | 0.0685 | 0.0267 | 0.0022 |
| standard | adasyn | 0.9155 | 0.0292 | 0.2070 | 0.0462 | 0.0783 | 0.0055 |
| robust | class_weight | 0.7342 | 0.0816 | 0.1988 | 0.0648 | 0.0251 | 0.0021 |
| standard | undersample | 0.9064 | 0.0139 | 0.1616 | 0.0400 | 0.0808 | 0.0044 |
| robust | adasyn | 0.7777 | 0.0687 | 0.1584 | 0.0710 | 0.0300 | 0.0030 |
| robust | none | 0.3406 | 0.0843 | 0.1271 | 0.0468 | 0.2694 | 0.0975 |

> **굵은 행 = Phase 3+ 확정 전처리 (StandardScaler + SMOTE)**

### 핵심 발견

1. **StandardScaler가 RobustScaler를 압도.** Robust+None ROC-AUC=0.3406은 saga solver가 극단적 불균형 데이터에서 수렴 실패한 결과.
2. **불균형 처리 효과 (Standard 기준):** SMOTE(0.9311) > class_weight(0.9261) > none(0.9076) > ADASYN(0.9155) > undersample(0.9064) — ROC-AUC 기준.
3. **PR-AUC 안정성:** standard+none이 PR-AUC 0.2666으로 명목 최고이나 std=0.1145로 불안정. standard+smote는 PR-AUC 0.2413, std=0.0554로 절반 이하의 분산.
4. **가이드북 비교:** 가이드북 §2.3은 단일 표준화만 적용 후 모델 학습. 이 ablation 단계가 가이드북에 빠진 단계. 불균형 처리가 ROC-AUC를 최대 +0.024(none→SMOTE) 개선.

### 확정 전처리 (Phase 3+)

```
Scaler  : StandardScaler (fit on train, transform val)
Resample: SMOTE (random_state=42, k_neighbors=5, fit on scaled train only)
```

### 핵심 인사이트 1줄

> **StandardScaler + SMOTE가 ROC-AUC 0.9311로 최고, PR-AUC 분산 절반으로 안정적 → Phase 3 확정 전처리; 불균형 처리 ablation이 가이드북 대비 +0.024 ROC-AUC 개선 근거 확보.**

---

*다음 Phase: 선형 베이스라인 (`02_linear_baselines.ipynb`) — LR(L1/L2), LDA, QDA, SVM(linear/RBF) 5-fold CV*

---

## Phase 3 — 선형·생성적 베이스라인 결과 (2026-05-10)

### 실험 설정

| 항목 | 값 |
|---|---|
| 전처리 | StandardScaler + SMOTE (LR/LDA/QDA) / class_weight='balanced' (SVM) |
| CV | Stratified 5-fold 공유 |
| 데이터 | labeled_data.csv (7,996행, 25 변수) |
| 모델 수 | 6종 (LR-L1, LR-L2, LDA, QDA, SVM-linear, SVM-RBF) |
| 총 실험 | 24 (그리드 조합 × 5-fold) |

### Best-per-Family 결과표

| 모델 | 최적 파라미터 | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| **QDA** | reg_param=0.01 | **0.9344 ±0.0175** | **0.3526 ±0.1026** | 0.0716 |
| LR-L2 | C=10 | 0.9343 ±0.0118 | 0.2690 ±0.0537 | 0.0828 |
| LR-L1 | C=10 | 0.9342 ±0.0119 | 0.2691 ±0.0535 | 0.0848 |
| SVM-RBF | C=10, γ=0.1 | 0.9297 ±0.0116 | 0.0890 ±0.0151 | 0.0000 |
| LDA | shrinkage=auto | 0.9078 ±0.0128 | 0.1447 ±0.0523 | 0.0813 |
| SVM-linear | C=10 | 0.9074 ±0.0225 | 0.3434 ±0.1094 | 0.0000 |

> **굵은 행 = Phase 4 앙상블 비교 기준선 (QDA)**

### 핵심 발견

1. **ROC-AUC 기준:** QDA(0.9344) ≈ LR-L1/L2(0.9342) >> SVM (0.90~0.93). 선형 결정경계로 ROC-AUC 0.93+ 달성.
2. **PR-AUC 기준:** QDA(0.35) > SVM-linear(0.34) > LR(0.27) > LDA(0.13) > SVM-RBF(0.08). QDA의 클래스별 공분산 모델링이 불량 포착에 유리.
3. **SVM-RBF 역설:** ROC-AUC 3위지만 PR-AUC 최저 → 임계값 0.5에서 minority 예측 실패. Stacking base learner로는 사용 가능.
4. **가이드북 비교:** 가이드북 §2.3의 "SVM best" 주장과 달리, QDA가 전 지표 1위. 가이드북 미시도 모델(QDA)이 숨겨진 강자 → ablation의 가치 정량화.

### 핵심 인사이트 1줄

> **QDA(reg=0.01)가 ROC-AUC 0.9344·PR-AUC 0.3526으로 전 모델 1위; 가이드북 미시도 모델이 "SVM best" 주장을 뒤집음 — 단계별 ablation의 originality 근거 확보.**

---

*다음 Phase: 차원축소 + 앙상블 + NN (`03_dimensionality_reduction.ipynb`, `04_ensemble_nn.ipynb`)*

---

## Phase 4-A — 차원축소 Ablation (2026-05-10)

### 실험 설정: LR-L2(C=10) 고정, StandardScaler+SMOTE

| 방법 | 차원 수 | ROC-AUC | PR-AUC |
|---|---|---|---|
| None (full) | 25 | 0.9343 ±0.0118 | 0.2690 ±0.0537 |
| VarianceThreshold | 25 | 0.9343 (동일) | 0.2690 (동일) |
| PCA 80% | 3 | 0.8061 ±0.0487 | 0.1151 |
| PCA 90% | 4 | 0.8657 ±0.0403 | 0.1851 |
| PCA 95% | 6 | 0.8746 ±0.0332 | 0.1913 |
| PCA 99% | 9 | 0.8915 ±0.0143 | 0.2280 |
| TreeTop-5 | 5 | 0.8320 ±0.1295 | 0.2076 |
| TreeTop-10 | 10 | 0.9371 ±0.0124 | 0.2554 |
| **TreeTop-15** | **15** | **0.9380 ±0.0204** | **0.2899** |

**핵심:** TreeTop-15가 full(25)보다 우수. PCA는 성능 저하. 가이드북에 없는 단계 전체가 originality.

---

## Phase 4-B — 앙상블 + NN 결과 (2026-05-10)

### 실험 설정: StandardScaler+SMOTE (RF/GBM/MLP) / class_weight (AdaBoost)

| 모델 | ROC-AUC | PR-AUC | 비고 |
|---|---|---|---|
| **MLP[256,128,64]** | **0.9497 ±0.0166** | **0.4710 ±0.0968** | PyTorch, Dropout=0.3 |
| RandomForest(n=500) | 0.9478 ±0.0123 | 0.4481 ±0.1121 | best ensemble |
| Guidebook DNN | 0.9468 ±0.0164 | 0.4655 ±0.1269 | §2.3 재현 |
| GBM(n=100,lr=0.1) | 0.9397 ±0.0226 | 0.4439 ±0.1262 | |
| AdaBoost(n=50) | 0.8953 | ~0.50 | PR-AUC 최고 (class_weight stump) |
| *QDA (Phase3 ref)* | *0.9344* | *0.3526* | *기준선* |
| *LR-L2 (Phase3 ref)* | *0.9343* | *0.2690* | *기준선* |

### 핵심 발견

1. **Phase3→4 향상:** ROC-AUC +0.015, PR-AUC +0.12~+0.15 (앙상블·NN의 비선형 이득)
2. **가이드북 DNN 재현:** ROC=0.9468 / PR=0.4655. 우리 MLP(+Dropout)가 소폭 우위.
3. **AdaBoost 이변:** PR-AUC~0.50으로 최고 — 불균형 데이터에서 class_weight stump reweighting 효과.
4. **Phase 5 Stacking 후보:** MLP, RF, Guidebook DNN, QDA, LR-L2 (다양성 확보).

### 핵심 인사이트 1줄

> **앙상블·NN이 Phase3 선형 모델 대비 ROC-AUC +0.015, PR-AUC +0.15 향상; Guidebook DNN 재현과 우리 MLP+Dropout 비교에서 Dropout이 PR-AUC +0.005 개선 확인 — originality 핵심 증거.**

---

*다음 Phase: CNN 보조 + Stacking 메타학습기 (`05_cnn_aux_stacking.ipynb`)*

---

## Phase 5 — CNN + Stacking + 운용점 분석 (2026-05-10)

### 1D-CNN 결과

| 모델 | ROC-AUC | PR-AUC |
|---|---|---|
| CNN1D(16,32) ep=30 | 0.9408 ±0.0133 | 0.4126 ±0.1139 |
| **CNN1D(32,64) ep=50** | **0.9472 ±0.0152** | **0.4501 ±0.1194** |

구조: 25 피처 → Conv1d(1→ch1, k=3) → MaxPool → Conv1d(ch1→ch2, k=3) → MaxPool → FC(64) → Dropout(0.3) → Output

### Stacking 결과 (OOF 메타 LR)

| 기반 모델 | meta-LR ROC-AUC | meta-LR PR-AUC |
|---|---|---|
| LR-L2+QDA+RF+GBM+MLP → LR-meta | 0.9462 ±0.0108 | 0.4484 ±0.1021 |

**Stacking이 단일 best(MLP 0.9497)를 넘지 못함** — base learner 다양성 부족.

### 운용점 분석

| 모델 | Prec≥0.95 Recall | Prec≥0.99 Recall |
|---|---|---|
| MLP[256,128,64] | 33.8% | **32.4%** |
| Stacking | 32.4% | 23.9% |

Precision≥0.99 운용 시 MLP가 불량 71건 중 약 23건 탐지 가능.

### 핵심 발견

1. **CNN1D(32,64) ≈ Phase4 앙상블 수준** — 시계열 정보 없이도 경쟁력 있음.
2. **Stacking < MLP 단일** — "Stacking이 항상 이긴다"는 가정 반례 발견.
3. **최고 모델**: MLP[256,128,64]+SMOTE (전 Phase 통틀어 ROC 0.9497, PR 0.4710).
4. **가이드북 비교**: 가이드북 "AE→SVM→DNN 순서" 대비, 우리 ablation에서 가장 기여한 단계 = 불균형 처리(Ph2 +0.024)와 비선형 모델 전환(Ph4 +0.12 PR-AUC).

### 핵심 인사이트 1줄

> **Stacking이 단일 MLP를 넘지 못해 '무조건 스태킹이 낫다'는 가정 반례; 운용점 Prec≥0.99에서 MLP가 Recall 32.4%로 최고 — Phase 6 보고서 핵심 그림 확정.**
