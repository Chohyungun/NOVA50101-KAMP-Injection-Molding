# KAMP 사출성형 공정 불량 예측 — 단계별 Ablation + 비지도 이상탐지

**프로젝트:** NOVA50101 인공지능학개론 텀프로젝트  
**데이터:** KAMP 사출성형기 AI 데이터셋 (886,227행, 25개 유효 변수)  
**작성일:** 2026-05-16 (v2)

---

## 1. 프로젝트 개요

KAMP 플랫폼(KAIST·UNIST)의 자동차 앞유리 사이드 몰딩 사출성형 데이터를 분석한다. 가이드북(§2.3)의 "AE→SVM→DNN 순차 학습 후 best 선택" 방식 대신, 각 처리 단계(전처리·선형모델·차원축소·앙상블·CNN·Stacking)를 분리해 어느 단계가 성능을 좌우하는지 정량적으로 측정한다. 추가로, 가이드북이 구현하지 않은 비지도 이상탐지 접근(AE, K-Means 거리, One-Class SVM)을 비교해 unlabeled 795K 활용 가능성을 탐색한다.

## 2. 데이터 개요

### 2.1 데이터셋 구성

| dataset | rows | cols | has_label | 역할 |
|---|---|---|---|---|
| labeled_data | 7,996 | 45 | True | 지도학습 주 데이터 |
| unlabeled_data | 795,315 | 45 | False | 비지도 이상탐지 학습용 |
| moldset_labeled | 2,607 | 46 | True | 준지도학습 labeled |
| supervised_label_cn7 | 6,736 | 25 | True | CN7 전용 지도학습 |
| moldset_labeled_cn7/rg3 | 1,211 / 1,182 | 25 | True | CN7/RG3 semi-supervised |
| moldset_unlabeled_cn7/rg3 | 35,239 / 35,941 | 24 | False | CN7/RG3 semi-supervised unlabeled |

### 2.2 유효 변수 (25개)

제거: Mold_Temperature_1/2/5~12(분산=0, 10개), Barrel_Temperature_7(near-zero), PART_FACT_SERIAL(ID)

### 2.3 클래스 불균형

- 양품(0): 7,925건 (99.11%)
- 불량(1): 71건 (0.89%) — 극심한 불균형, SMOTE 필수

### 2.4 핵심 EDA 발견

Max_Back_Pressure(r=+0.115), Cycle_Time(r=+0.105)이 불량과 가장 강한 상관을 보인다. Cycle_Time 이상치 구간의 불량률은 3.3%로 정상 구간(0.3%) 대비 11배 높다. CN7 vs RG3: KS 검정 22/24 변수에서 분포 유의차(p<0.05).

## 3. 방법론 — 단계별 Ablation 설계

```
Phase 2: 전처리 Ablation (2 scaler × 5 불균형처리 = 10 조합, 분류기 LR-L2 고정)
Phase 3: 선형 베이스라인 (LR-L1/L2, LDA, QDA, SVM-linear, SVM-RBF)
Phase 4-A: 차원축소 비교 (None/VarThresh/PCA/TreeTop-K)
Phase 4-B: 앙상블·NN (RF, AdaBoost, GBM, MLP, 가이드북 DNN)
Phase 5: CNN1D + Stacking + 운용점 분석
Phase 6: 전 Phase 통합 비교
Phase 7: 비지도 이상탐지 — OC-SVM / K-Means / Denoising AE (진행 중)
```

모든 실험: Stratified 5-fold CV, random_state=42, ROC-AUC·PR-AUC 보고

## 4. 단계별 결과

### Phase 2: 전처리 Ablation

| scaler | resample | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std |
|---|---|---|---|---|---|
| standard | **smote** | **0.9311** | 0.0107 | 0.2413 | 0.0554 |
| standard | class_weight | 0.9261 | 0.0107 | 0.2316 | 0.0608 |
| standard | none | 0.9076 | 0.0159 | 0.2666 | 0.1145 |
| standard | adasyn | 0.9155 | 0.0292 | 0.2070 | 0.0462 |
| standard | undersample | 0.9064 | 0.0139 | 0.1616 | 0.0400 |

확정 전처리: StandardScaler + SMOTE (ROC-AUC 0.9311). SMOTE 미적용(0.9076) 대비 ROC-AUC +0.024 향상, PR-AUC std 0.1145→0.0554로 안정화.

### Phase 3: 선형 베이스라인

| model | roc_auc_mean | pr_auc_mean | 비고 |
|---|---|---|---|
| **QDA** (reg=0.01) | **0.9344** | **0.3526** | 가이드북 미시도 — Phase3 best |
| LR-L2 (C=10) | 0.9343 | 0.2690 | |
| LR-L1 (C=10) | 0.9342 | 0.2691 | |
| SVM-RBF (C=10, γ=0.1) | 0.9297 | 0.0890 | 가이드북 "best" — PR-AUC 최하위 |
| LDA (shrinkage=auto) | 0.9078 | 0.1447 | |
| SVM-linear (C=10) | 0.9074 | 0.3434 | |

best: QDA(reg=0.01), ROC-AUC=0.9344, PR-AUC=0.3526. 가이드북이 "SVM best"로 제시한 SVM-RBF는 PR-AUC 0.0890으로 최하위.

### Phase 4-A: 차원축소

| method | n_dims | roc_auc_mean | pr_auc_mean |
|---|---|---|---|
| None (full=25) | 25 | 0.9343 | 0.2690 |
| **TreeTop-15** | **15** | **0.9380** | **0.2899** |
| TreeTop-10 | 10 | 0.9371 | 0.2554 |
| PCA 99% | 9 | 0.8915 | 0.2280 |
| PCA 95% | 6 | 0.8746 | 0.1913 |
| PCA 80% | 3 | 0.8061 | 0.1151 |

TreeTop-15가 소폭 최고. PCA는 이 데이터에서 성능을 떨어뜨렸다.

### Phase 4-B: 앙상블·NN

| model | roc_auc_mean | pr_auc_mean | f1_mean |
|---|---|---|---|
| **MLP[256,128,64]** | **0.9497** | **0.4710** | 0.1521 |
| RandomForest (n=500) | 0.9478 | 0.4481 | 0.3723 |
| Guidebook_DNN | 0.9468 | 0.4655 | 0.2067 |
| GBM (n=100) | 0.9397 | 0.4439 | 0.3116 |
| AdaBoost (n=50) | 0.8953 | 0.2566 | 0.0816 |

best: MLP[256,128,64]+Dropout(0.3), ROC-AUC=0.9497, PR-AUC=0.4710. 가이드북 DNN(0.9468/0.4655) 대비 PR-AUC +0.005.

### Phase 5: CNN + Stacking

| model | roc_auc_mean | pr_auc_mean |
|---|---|---|
| MLP[256,128,64] (참조) | 0.9497 | 0.4710 |
| CNN1D(32,64), ep=50 | 0.9472 | 0.4501 |
| Stacking (meta-LR) | 0.9462 | 0.4484 |

Stacking(0.9462) < MLP 단일(0.9497) — 기반 학습기 다양성 부족.

### 운용점 분석

| model | precision≥ | recall | threshold |
|---|---|---|---|
| MLP[256,128,64] | 0.95 | **33.8%** | 0.9977 |
| MLP[256,128,64] | 0.99 | **32.4%** | 0.9990 |
| Stacking | 0.99 | 23.9% | 0.5825 |

Precision≥0.99 운용 시 MLP: 불량 71건 중 약 23건 탐지.

## 5. 비지도 이상탐지 (Phase 7, 진행 중)

가이드북이 구현했지만 우리가 아직 재현하지 않은 AE 이상탐지, 그리고 가이드북에 없는 K-Means 거리 기반 이상탐지를 비교한다.

### 5.1 1차 실험 결과 (2026-05-16 완료)

| 방법 | 학습 데이터 | ROC-AUC (5-fold) | PR-AUC (5-fold) |
|---|---|---|---|
| **One-Class SVM** (nu=0.02) | labeled 양품 7,925 | **0.8814 ± 0.027** | **0.1765 ± 0.087** |
| K-Means (k=5, best) | unlabeled 795,315 | 0.4697 ± 0.063 | 0.1444 ± 0.084 |
| K-Means (k=10~50) | unlabeled 795,315 | 0.31~0.40 | 0.06~0.07 |

### 5.2 K-Means 실패 원인 분석

unlabeled_data 795K는 다양한 기계·제품의 데이터를 모두 포함한다. labeled_data는 "650-ton 우진 #2" 기계의 CN7/RG3만으로 구성된다. 기계·제품 간 공정 파라미터 분포 차이로 인해, 전체 unlabeled로 학습한 K-Means 중심점이 CN7/RG3 정상 패턴을 대표하지 못한다. 양품과 불량 모두 유사한 centroid 거리를 갖게 되어 분리 불가.

### 5.3 예정 실험

| 방법 | 상태 | 예상 개선점 |
|---|---|---|
| Denoising AE (labeled 양품) | 구현 예정 | 가이드북 §2.2.1 재현 |
| Denoising AE (labeled + unlabeled) | 구현 예정 | 도메인 포함 버전 |
| K-Means (필터링 후 unlabeled) | 구현 예정 | 동일 기계·제품으로 제한 |
| GMM log-likelihood | Future Work | K-Means 한계 보완 |

## 6. 가이드북 비교

| 항목 | 가이드북(§2.3) | 우리 결과 |
|---|---|---|
| best 모델 선택 | AE→SVM→DNN 순차 | 단계별 ablation 비교 |
| 불균형 처리 | 개념 언급만 | 4종 10조합 정량 비교 |
| SVM 결과 | best로 보고 | Phase 3에서 QDA에 밀림 |
| DNN 재현 | — | ROC=0.9468, PR=0.4655 |
| 우리 MLP | — | ROC=0.9497, PR=0.4710 |
| 차원축소 비교 | 없음 | Phase 4-A TreeTop-15 최적 |
| K-Means 이상탐지 | 없음 | 1차 시도(ROC 0.47) + 실패 원인 분석 |

## 7. 핵심 인사이트

어느 단계가 성능을 가장 끌어올렸는가:

| Phase | 처리 | 기여 지표 | 기여 크기 |
|---|---|---|---|
| Phase 2 | None → SMOTE | ROC-AUC | +0.024 |
| Phase 3 | LR → QDA | PR-AUC | +0.084 |
| Phase 4 | QDA → MLP | PR-AUC | +0.119 |
| Phase 5 | 단일→Stacking | PR-AUC | −0.023 (반례) |

1. **비선형 전환(Phase 4)이 최대 기여** — PR-AUC +0.119
2. **불균형 처리(Phase 2)는 안정화 효과 핵심** — PR-AUC std 절반으로 감소
3. **Stacking은 반례** — 기반 학습기 다양성 부족 시 단일 MLP 미달
4. **K-Means 이상탐지 도메인 불일치** — unlabeled 필터링이 핵심 선결 과제

## 8. 한계 및 Future Work

1. **K-Means 이상탐지 도메인 불일치:** unlabeled를 같은 기계·제품으로 필터링해야 함
2. **Denoising AE 미구현:** labeled 양품만 vs labeled+unlabeled 두 버전 비교 예정
3. **시계열성 미활용:** 사이클 간 드리프트 패턴(EDA에서 10-27 전후 변화 확인) 미반영
4. **CN7/RG3 분리 미구현:** KS 검정 22/24 변수 유의차 무시한 혼합 학습
5. **SMOTE 한계:** 불량 71개로 보간 품질 의심, Cost-sensitive / Focal Loss 미시도
6. **Pseudo-labeling 미시도:** 가이드북 §2.2.2 재현 (Future Work)

---

*Figure references:*  
- 전처리: `NB01_fig1_preproc_ablation_heatmap.png`  
- 선형 베이스라인: `NB02_fig1_baseline_roc_curves.png`, `NB02_fig2_baseline_pr_curves.png`  
- 앙상블: `NB04_fig1_ensemble_roc_curves.png`, `NB04_fig3_ensemble_nn_comparison.png`  
- Phase 5: `NB05_fig1_phase5_summary.png`  
- 이상탐지: `anomaly_kmeans_k_ablation.png`, `anomaly_detection_roc.png`, `anomaly_score_distribution.png`
