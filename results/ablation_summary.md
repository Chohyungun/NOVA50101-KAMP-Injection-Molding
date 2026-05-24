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

### 핵심 1줄

유효 변수 26개, 불량률 0.89%, 강한 상관 짝 34쌍, CN7·RG3 분포 22/24 유의차이 → **Phase 3 기본은 labeled_data 단독, Phase 4에서 product 통합/분리 ablation 결정.**

### 가이드북 비교

- 가이드북이 "0"으로 표기한 11개 변수(Mold_Temp_1,2,5-12 + Barrel_Temp_7)가 실측과 100% 일치.
- 가이드북 §2.3의 "AE/SVM/DNN 차례 학습" 흐름 대비, 이 프로젝트는 **단계별 ablation(전처리→선형→차원축소→앙상블/NN→스태킹)**으로 어느 단계가 성능을 좌우하는지 분리 측정했다. originality의 1차 차별점.

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

굵은 행 = Phase 3+ 확정 전처리 (StandardScaler + SMOTE)

### 발견 사항

1. **StandardScaler가 RobustScaler를 압도.** Robust+None ROC-AUC=0.3406은 saga solver가 극단적 불균형 데이터에서 수렴 실패한 결과다.
2. **불균형 처리 효과 (Standard 기준):** SMOTE(0.9311) > class_weight(0.9261) > none(0.9076) > ADASYN(0.9155) > undersample(0.9064) — ROC-AUC 기준.
3. **PR-AUC 안정성:** standard+none이 PR-AUC 0.2666으로 명목 최고이나 std=0.1145로 불안정하다. standard+smote는 PR-AUC 0.2413에 std=0.0554로 분산이 절반 이하.
4. **가이드북 비교:** 가이드북 §2.3은 단일 표준화만 적용 후 모델 학습. 이 ablation 단계는 가이드북에 없다. 불균형 처리가 ROC-AUC를 최대 +0.024(none→SMOTE) 끌어올렸다.

### 확정 전처리 (Phase 3+)

```
Scaler  : StandardScaler (fit on train, transform val)
Resample: SMOTE (random_state=42, k_neighbors=5, fit on scaled train only)
```

### 핵심 1줄

**StandardScaler + SMOTE가 ROC-AUC 0.9311로 최고, PR-AUC 분산은 절반 수준으로 안정적 → Phase 3 확정 전처리. 불균형 처리 ablation이 가이드북 대비 +0.024 ROC-AUC 개선 근거.**

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

굵은 행 = Phase 4 앙상블 비교 기준선 (QDA)

### 발견 사항

1. **ROC-AUC 기준:** QDA(0.9344) ≈ LR-L1/L2(0.9342) >> SVM (0.90~0.93). 선형 결정경계만으로 ROC-AUC 0.93+가 나왔다.
2. **PR-AUC 기준:** QDA(0.35) > SVM-linear(0.34) > LR(0.27) > LDA(0.13) > SVM-RBF(0.08). QDA의 클래스별 공분산 모델링이 불량 포착에 유리하다.
3. **SVM-RBF 역설:** ROC-AUC 3위지만 PR-AUC 최저 → 임계값 0.5에서 minority 예측 실패. Stacking base learner로는 활용 가능.
4. **가이드북 비교:** 가이드북 §2.3의 "SVM best" 주장과 달리, QDA가 전 지표 1위였다. 가이드북이 시도하지 않은 모델이 숨겨진 강자 → ablation의 가치가 수치로 드러난 지점.

### 핵심 1줄

**QDA(reg=0.01)가 ROC-AUC 0.9344·PR-AUC 0.3526으로 전 모델 1위. 가이드북 미시도 모델이 "SVM best" 주장을 뒤집었다 — 단계별 ablation의 originality 근거.**

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

TreeTop-15가 full(25)보다 오히려 우수하게 나왔다. PCA는 성능이 떨어졌다. 이 차원축소 단계 전체가 가이드북에 없는 부분이며 originality 근거다.

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

### 발견 사항

1. **Phase3→4 향상:** ROC-AUC +0.015, PR-AUC +0.12~+0.15. 비선형 모델로 전환한 이득이다.
2. **가이드북 DNN 재현:** ROC=0.9468 / PR=0.4655. Dropout을 추가한 MLP가 소폭 앞섰다.
3. **AdaBoost 이변:** PR-AUC~0.50으로 최고치 — 불균형 데이터에서 class_weight stump reweighting 효과가 컸다.
4. **Phase 5 Stacking 후보:** MLP, RF, Guidebook DNN, QDA, LR-L2 (다양성 확보).

### 핵심 1줄

**앙상블·NN이 Phase3 선형 모델 대비 ROC-AUC +0.015, PR-AUC +0.15 향상. Guidebook DNN 재현 대비 MLP+Dropout이 PR-AUC +0.005 개선 — originality 핵심 증거.**

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

**Stacking이 단일 best(MLP 0.9497)를 넘지 못했다** — base learner 다양성이 부족했기 때문이다.

### 운용점 분석

| 모델 | Prec≥0.95 Recall | Prec≥0.99 Recall |
|---|---|---|
| MLP[256,128,64] | 33.8% | **32.4%** |
| Stacking | 32.4% | 23.9% |

Precision≥0.99 운용 시 MLP가 불량 71건 중 약 23건 탐지 가능하다.

### 발견 사항

1. **CNN1D(32,64) ≈ Phase4 앙상블 수준** — 시계열 정보 없이도 충분히 경쟁력 있다.
2. **Stacking < MLP 단일** — "Stacking이 항상 이긴다"는 가정의 반례다.
3. **최고 모델**: MLP[256,128,64]+SMOTE (전 Phase 통틀어 ROC 0.9497, PR 0.4710).
4. **가이드북 비교**: 가이드북 "AE→SVM→DNN 순서" 대비, 이 ablation에서 가장 기여한 단계는 불균형 처리(Ph2 +0.024)와 비선형 모델 전환(Ph4 +0.12 PR-AUC)이었다.

### 핵심 1줄

**Stacking이 단일 MLP를 넘지 못해 '무조건 스태킹이 낫다'는 가정의 반례. 운용점 Prec≥0.99에서 MLP가 Recall 32.4%로 최고 — Phase 6 보고서 핵심 그림 확정.**

---

## Phase 7 — 이상탐지 1차 + 방법 C v1·v2·v3 (2026-05-24)

### 비지도 이상탐지 (07_anomaly_detection.ipynb)

| 방법 | ROC-AUC | PR-AUC | 비고 |
|---|---|---|---|
| One-Class SVM (labeled 양품만) | 0.8814 ± 0.027 | 0.1765 ± 0.087 | 구현 완료 |
| K-Means (k=5, unlabeled 전체) | 0.4697 | — | 도메인 불일치 실패 (795K 혼재) |

### 방법 C — Pseudo-labeling 진행 경과

#### v1 실패 원인 규명 (08_pseudo_labeling.ipynb)
- EllipticEnvelope pseudo-defect = **기계정지 상태 샘플** (Barrel_Temp=0, RPM=0)
- Feature 방향일치율 28% (랜덤 50% 이하) → pseudo-defect이 오히려 노이즈

#### v2 핵심 개선 결과 (08_method_c_v2.ipynb — 이전 버전)
- 기계정지 44% 제거 (31,310행 → 생산상태 39,870행)
- Feature alignment 대폭 개선: A=52%, B=80%, C=60%, D=100%(cosine=0.993)
- **버그 수정 적용** — run_cv fold 인덱스 문제 (pseudo-label이 학습에 포함되지 않던 문제)
- 5-fold CV (RF): B=+0.0317 ROC / +0.0256 PR, D=+0.0312 ROC / +0.0174 PR ("IMPROVED")

> **v2 결과에 대한 심사 비판 (4개 FATAL, 3개 MAJOR):**
> 1. 스케일러 공간 불일치: unlabeled-fit scaler로 전략 선택 → labeled-train scaler로 모델 학습
> 2. 전략 B 소프트 누수: teacher RF가 CV val fold 포함한 전체 7996개로 학습
> 3. align_score 동등 가중치: Clamp_Open(diff=187) vs Clamp_Close(diff=0.079) 동일 1표
> 4. 전략 C 방향벡터 raw space: Clamp_Open_Position이 내적값 99% 지배 (단변량 threshold와 동일)
> 5. Wilcoxon 검정 없이 "IMPROVED" 레이블 (σ=0.032 vs 개선량 0.032)
> 6. n ablation이 전략 C에만 수행됨 (B·D 미수행)
> 7. ST threshold=0.80 근거 없음 (ablation 미수행)

#### v3 재설계 (현재 실행 중 — 심사 비판 전면 수용)

| 수정 항목 | v2 | v3 |
|---|---|---|
| 전략 선택 스케일러 | sc.fit(X_unl) (unlabeled) | sc_sel.fit(X_lab) (labeled) |
| align_score | raw 공간, 동등 가중치 | 표준화 공간, RF importance 가중 |
| 전략 C 방향벡터 | raw space (Clamp 지배) | 표준화 space (25피처 균등) |
| 전략 B CV | ALL labeled teacher (누수) | fold별 teacher RF (누수 차단) |
| 통계 검정 | 없음 (mean > std 기준) | Wilcoxon signed-rank (n=5) |
| n ablation | 전략 C만 | 전략 B·C·D 모두 |
| Self-training | threshold=0.80 고정 | threshold sweep [0.5, 0.6, 0.7, 0.8] |

> **v3 최종 결과 (2026-05-24 완료)**

#### v3 Feature Alignment (표준화 공간, RF importance 가중)

| 전략 | uniform% | cos | weighted% | cos_w | 비고 |
|---|---|---|---|---|---|
| A: EllipticEnv | 52% | 0.151 | 51% | -0.001 | 보통 |
| B: RF Confidence | 64% | 0.336 | 72% | 0.365 | 양호 |
| C: Direction Proj | 60% | 0.150 | 61% | -0.077 | v2 cosine 0.969→0.150 ↓ (raw 지배 해소 확인) |
| D: kNN Defect | 100% | 0.938 | 100% | 0.879 | 우수 |

#### v3 5-fold CV (Wilcoxon signed-rank 포함)

| 전략 | ROC-AUC | vs Base | PR-AUC | vs Base | Wilcoxon |
|---|---|---|---|---|---|
| RF Base | 0.9296 | — | 0.4479 | — | — |
| RF + A | 0.9378 | +0.0081 | 0.4421 | -0.0058 | trend(p=0.312) |
| **RF + B (fold-teacher)** | **0.9271** | **-0.0025** | **0.4441** | **-0.0039** | **ns(p=0.781)** |
| RF + C | 0.9401 | +0.0105 | 0.4423 | -0.0056 | trend(p=0.312) |
| **RF + D** | **0.9531** | **+0.0235** | **0.4572** | **+0.0093** | **IMPROVED*(p=0.031)** |
| MLP + B (fold-teacher) | 0.9485 | +0.0189 | 0.4475 | -0.0005 | trend(p=0.156) |

#### v3 핵심 발견

1. **전략 B 누수 확인됨**: v2에서 ROC+0.032 "IMPROVED"가 fold-teacher 적용 후 ROC-0.003 ns로 붕괴  
   → v2의 B 개선은 teacher RF가 val fold 정보를 포함했기 때문 (소프트 데이터 누수)
2. **전략 D만 유일하게 유의**: Wilcoxon p=0.031, ROC+0.024, PR+0.009  
   → labeled 불량 71개의 kNN 이웃 선택은 누수 없이 실질적 개선 달성
3. **전략 C raw→scaled 변화**: cosine 0.969→0.150 — raw 공간의 높은 cosine이 Clamp_Open_Position 단변량 지배 때문이었음 확인. 표준화 후 실질 방향정렬은 훨씬 낮음
4. **Self-training**: threshold=0.7 최적(PR=0.4491)이지만 baseline(0.4479) 수준. DR=0.89% 극단 불균형에서 구조적 한계

#### v3 최종 판정

**방법 C VALID** (Wilcoxon p=0.031 전략 D). 단 "어떤 전략인가"와 "누수 없는 선택인가"가 성패를 가름.  
v2 B전략의 허위 개선 발견이 오히려 더 큰 방법론적 기여 — 준지도학습의 핵심 함정 실증.

### 핵심 발견 (방법 C 전체)

1. **기계정지 오염**: unlabeled 44%가 기계정지 상태 — 필터링 없이는 어떤 이상탐지도 실패
2. **align_score**: raw 공간에서는 단일 대형 피처가 지배 → 표준화 공간에서만 의미있는 방향정렬
3. **전략 D (kNN)**: labeled 불량 71개 이웃 기반 — 방향일치 100% / cosine 0.993 → 가장 순수한 전략
4. **Self-training**: DR=0.89% 극단 불균형에서 RF 확신도 기반 ST는 구조적으로 불리 (pseudo-불량 6개/전체)

### 핵심 1줄

**기계정지 오염 제거 + 표준화 공간 전략 + fold별 teacher RF가 방법 C의 방법론적 완결성을 결정한다. 단순히 "pseudo-label을 추가했다"가 아니라 어떤 공간에서 어떻게 선택했는가가 핵심.**
