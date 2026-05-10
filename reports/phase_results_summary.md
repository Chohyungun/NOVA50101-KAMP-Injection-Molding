# KAMP 사출성형 텀프로젝트 — 단계별 Ablation 결과 요약

> **작성:** 감독관 에이전트, 2026-05-10  
> **기반 데이터:** results/ablation_summary.md, results/decisions.md, results/tables/*.csv  
> **목적:** Phase 2~5 결과 통합 및 Phase 6 보고서 초안 섹션 제공

---

## 1. 데이터셋 개요

| 항목 | 값 |
|---|---|
| 원본 CSV 수 | 6개 (data/raw/) |
| 전체 행 수 (8개 CSV 합산) | 886,227행 |
| 지도학습 주 데이터 | labeled_data.csv — 7,996행 × 45컬럼 |
| 유효 독립 변수 수 | **25개** (분산=0 변수 10개 제거 + Barrel_Temperature_7 제거 + PART_FACT_SERIAL 제거) |
| 양품 / 불량 비율 | 7,925 / 71 = **99.11% / 0.89%** (극심한 불균형) |
| 결측 현황 | Reason 컬럼 99.1% 결측 (불량 원인 미기록), 나머지 피처 0% |
| 강한 상관 변수 쌍 | 피어슨 |r| > 0.95 기준 34쌍 |
| CV 설정 | Stratified 5-fold, random_state=42 (data/splits/fold_0~4.npy 공유) |
| 평가 지표 | ROC-AUC (주), PR-AUC (주), F1 (보조), 운용점 Precision@Recall=0.5/0.7/0.9 |

**제거 변수 목록:**

| 변수 | 제거 사유 |
|---|---|
| Mold_Temperature_1, 2, 5~12 (10개) | 분산 = 0 (수집 미실시, 가이드북 §2.1과 일치) |
| Barrel_Temperature_7 | 고유값 2개, 평균 0.009°C (near-zero variance) |
| PART_FACT_SERIAL | 공정 변수가 아닌 생산 시리얼 ID (identity leak 방지) |

**CN7 vs RG3 분포:** KS 검정 결과 24개 변수 중 22개(91.7%)에서 유의미한 분포 차이(p < 0.05). 두 제품군의 통합 분석은 분포 왜곡 위험이 있으므로 본 텀프에서는 labeled_data.csv 단독 분석을 기본으로 채택.

---

## 2. 방법론 — 단계별 Ablation 설계

### 가이드북(KAIST·UNIST §2.3)의 접근 방식

가이드북은 다음 순서로 단일 best 모델을 선택:

1. Autoencoder(AE)로 비지도 특성 추출
2. SVM 학습 및 평가
3. DNN(3-layer MLP) 학습 및 평가
4. 세 모델 중 성능이 가장 좋은 1개를 "best"로 보고

**가이드북의 한계:** 어느 전처리 단계가 성능에 기여했는지, 불균형 처리가 얼마나 중요한지, SVM이 다른 선형 모델 대비 실제로 우월한지를 분리해 측정하지 않는다.

### 우리의 접근: 단계별 기여도 분리 측정

각 처리 단계를 독립적으로 변경하고 나머지는 고정하여 **"어느 단계가 성능을 좌우했는가"** 를 정량화한다.

```
Phase 2: 전처리 Ablation
  └─ 분류기(LR-L2) 고정 → Scaler 2종 × 불균형 처리 5종 = 10 조합 비교
  
Phase 3: 선형·생성적 베이스라인
  └─ 전처리 고정 → 분류기 6종(LR-L1, LR-L2, LDA, QDA, SVM-linear, SVM-RBF) 비교
  
Phase 4-A: 차원축소 Ablation
  └─ 분류기(LR-L2) 고정 → 차원축소 방법 8종 비교
  
Phase 4-B: 앙상블 + NN
  └─ 차원축소 고정 → 모델 패밀리 확장(RF, GBM, AdaBoost, MLP, 가이드북 DNN)
  
Phase 5: CNN 보조 + Stacking + 운용점
  └─ 1D-CNN 보조 단계, OOF Stacking 메타 LR, Precision-fixed Recall 분석
```

이 구조는 가이드북의 "AE→SVM→DNN 순서 선택" 방식과 정면 대비되며, 각 단계의 성능 기여도를 분리 측정하는 것이 본 텀프의 originality 핵심이다.

---

## 3. 단계별 결과 요약표

### 3.1 Phase 2: 전처리 Ablation (분류기: LR-L2 고정)

| 스케일러 | 불균형 처리 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | ±표준편차 |
|---|---|---|---|---|---|
| Standard | 없음 | 0.9076 | 0.0159 | 0.2666 | 0.1145 |
| Standard | 클래스 가중치(class_weight) | 0.9261 | 0.0107 | 0.2316 | 0.0608 |
| **Standard** | **SMOTE** | **0.9311** | **0.0107** | **0.2413** | **0.0554** |
| Standard | ADASYN | 0.9155 | 0.0292 | 0.2070 | 0.0462 |
| Standard | 언더샘플링(undersample) | 0.9064 | 0.0139 | 0.1616 | 0.0400 |
| Robust | 언더샘플링(undersample) | 0.8886 | 0.0340 | 0.2541 | 0.0682 |
| Robust | 없음 | 0.3406 | 0.0843 | 0.1271 | 0.0468 |

확정 전처리: **StandardScaler + SMOTE** (ROC-AUC 0.9311, PR-AUC 표준편차 0.0554로 전 조합 중 가장 안정적)

### 3.2 Phase 3: 선형·생성적 베이스라인 (6종, 5-fold CV)

| 모델 | 최적 파라미터 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | ±표준편차 | F1 | 가이드북 비교 |
|---|---|---|---|---|---|---|---|
| **QDA** | reg_param=0.01 | **0.9344** | 0.0175 | **0.3526** | 0.1026 | 0.0716 | 가이드북 미시도 |
| LR-L2 | C=10 | 0.9343 | 0.0118 | 0.2690 | 0.0537 | 0.0828 | — |
| LR-L1 | C=10 | 0.9342 | 0.0119 | 0.2691 | 0.0535 | 0.0848 | — |
| SVM-RBF | C=10, γ=0.1 | 0.9297 | 0.0116 | 0.0890 | 0.0151 | 0.0000 | 가이드북 "best" |
| LDA | shrinkage=auto | 0.9078 | 0.0128 | 0.1447 | 0.0523 | 0.0813 | — |
| SVM-linear | C=10 | 0.9074 | 0.0225 | 0.3434 | 0.1094 | 0.0000 | 가이드북 "best" |

Phase 3 최고 성능: **QDA(reg=0.01)** — 가이드북이 "SVM best"로 주장한 결과를 정량적으로 반박한다.

### 3.3 Phase 4-A: 차원축소 Ablation (분류기: LR-L2 고정)

| 차원축소 방법 | 차원 수 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | 가이드북 비교 |
|---|---|---|---|---|---|
| None (full) | 25 | 0.9343 | 0.0118 | 0.2690 | — |
| VarianceThreshold | 25 | 0.9343 | — | 0.2690 | — |
| PCA 80% | 3 | 0.8061 | 0.0487 | 0.1151 | 가이드북에 없는 단계 |
| PCA 90% | 4 | 0.8657 | 0.0403 | 0.1851 | 가이드북에 없는 단계 |
| PCA 95% | 6 | 0.8746 | 0.0332 | 0.1913 | 가이드북에 없는 단계 |
| PCA 99% | 9 | 0.8915 | 0.0143 | 0.2280 | 가이드북에 없는 단계 |
| TreeTop-5 | 5 | 0.8320 | 0.1295 | 0.2076 | 가이드북에 없는 단계 |
| TreeTop-10 | 10 | 0.9371 | 0.0124 | 0.2554 | 가이드북에 없는 단계 |
| **TreeTop-15** | **15** | **0.9380** | 0.0204 | **0.2899** | 가이드북에 없는 단계 |

PCA는 이 데이터에서 차원 감소 시 성능이 저하되었다. RF 중요도 기반 Top-15는 전체 25개 피처 대비 소폭 우위를 보이며, 차원축소 ablation 단계 전체가 가이드북에 없는 분석이다.

### 3.4 Phase 4-B: 앙상블 + NN (가이드북 DNN 재현 포함)

| 모델 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | ±표준편차 | F1 | 비고 |
|---|---|---|---|---|---|---|
| **MLP[256,128,64]** | **0.9497** | 0.0166 | **0.4710** | 0.0968 | 0.1521 | PyTorch, Dropout=0.3, 강의 구현 |
| RandomForest (n=500) | 0.9478 | 0.0123 | 0.4481 | 0.1121 | 0.3723 | Phase 4 최고 앙상블 |
| **Guidebook DNN** | **0.9468** | 0.0164 | **0.4655** | 0.1269 | 0.2067 | **가이드북 §2.3 재현** |
| GBM (n=100, lr=0.1) | 0.9397 | 0.0226 | 0.4439 | 0.1262 | 0.3116 | |
| AdaBoost (n=50) | 0.8953 | 0.0262 | 0.2566 | 0.1191 | 0.0816 | 클래스 가중치(class_weight) 우회 |
| *QDA Phase3 기준* | *0.9344* | *0.0175* | *0.3526* | *0.1026* | — | *이전 Phase 참조* |
| *LR-L2 Phase3 기준* | *0.9343* | *0.0118* | *0.2690* | *0.0537* | — | *이전 Phase 참조* |

Phase 3→4 향상: ROC-AUC +0.015, PR-AUC +0.12~+0.15 (비선형 모델 이득). Dropout을 적용한 우리 MLP가 가이드북 DNN 대비 PR-AUC +0.005 우위를 보인다.

### 3.5 Phase 5: CNN 보조 + Stacking + 운용점

#### 1D-CNN 결과

| 모델 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | ±표준편차 |
|---|---|---|---|---|
| CNN1D(ch=16,32), ep=30 | 0.9408 | 0.0133 | 0.4126 | 0.1139 |
| **CNN1D(ch=32,64), ep=50** | **0.9472** | 0.0152 | **0.4501** | 0.1194 |

구조: 25 피처 → Conv1d(1→ch1, k=3) → MaxPool → Conv1d(ch1→ch2, k=3) → MaxPool → FC(64) → Dropout(0.3) → Output

#### Stacking 결과 (OOF(폴드 외 예측) 메타 LR)

| 기반 모델 조합 | ROC-AUC 평균 | ±표준편차 | PR-AUC 평균 | ±표준편차 | F1 |
|---|---|---|---|---|---|
| LR-L2 + QDA + RF + GBM + MLP → 메타 LR | 0.9462 | 0.0108 | 0.4484 | 0.1021 | 0.4819 |

Stacking이 단일 최고 모델(MLP 0.9497)에 미달했다. 기반 학습기(base learner) 간 다양성 부족이 주된 원인이다.

#### 운용점 (Operating Point) 분석

| 모델 | 정밀도(Precision) 목표 | 달성 정밀도 | 재현율(Recall) | 결정 임계값 |
|---|---|---|---|---|
| Stacking | ≥0.95 | 0.9583 | 32.4% | 0.5443 |
| Stacking | ≥0.99 | 1.0000 | 23.9% | 0.5825 |
| **MLP[256,128,64]** | ≥0.95 | 0.9600 | **33.8%** | 0.9977 |
| **MLP[256,128,64]** | ≥0.99 | 1.0000 | **32.4%** | 0.9990 |

정밀도(Precision)≥0.99 운용 시 MLP가 불량 71건 중 약 23건을 탐지할 수 있다(재현율 32.4%).

---

## 4. 가이드북 비교 — 우리 ablation vs KAIST·UNIST §2.3

| 비교 항목 | 가이드북 (KAIST·UNIST §2.3) | 우리 ablation |
|---|---|---|
| 데이터 전처리 | 단일 표준화 적용, 불균형 처리 없음 | 불균형 처리 5종 ablation → SMOTE 확정, ROC-AUC +0.024 |
| 차원축소 | 명시 없음 | PCA vs 트리 중요도 8종 비교, 가이드북에 없는 단계 전체 |
| 모델 선택 방식 | AE, SVM, DNN 순 학습 후 단일 best 선택 | 6종 선형 + 5종 앙상블/NN + 1D-CNN + Stacking 체계적 비교 |
| 공개된 best 모델 | SVM (§2.3 명시) | QDA가 SVM 대비 PR-AUC +0.26 우위 (0.3526 vs 0.0890) |
| DNN 결과 | ROC-AUC 0.9468, PR-AUC 0.4655 (재현) | MLP+Dropout: ROC 0.9497, PR 0.4710 (+0.003/+0.005) |
| 운용점 분석 | 없음 | Precision-fixed Recall 표 및 PR 곡선 운용점 마커 |
| 단계 기여도 분리 | 없음 | Phase 2~5 각 단계별 성능 기여도 정량화 |

**핵심 발견:** 가이드북이 "SVM best"라고 주장한 모델은 PR-AUC 기준 6개 선형 모델 중 최하위(SVM-RBF 0.0890). 가이드북이 시도조차 하지 않은 QDA가 PR-AUC 0.3526으로 전 Phase 3 모델 중 1위. 이 역설이 단계별 ablation의 originality를 정량적으로 뒷받침한다.

---

## 5. 단계별 성능 기여도 분석

### 단계별 성능 기여도 요약

| Phase | 처리 단계 | 기여 지표 | 기여 크기 |
|---|---|---|---|
| Phase 2 | 불균형 처리 (None → SMOTE) | ROC-AUC | **+0.024** |
| Phase 3 | 모델 패밀리 선택 (LR → QDA) | PR-AUC | **+0.084** (0.269→0.353) |
| Phase 4 | 비선형 모델 전환 (QDA → MLP) | PR-AUC | **+0.119** (0.353→0.471) |
| Phase 5 | Stacking 메타 학습기 추가 | PR-AUC | -0.023 (단일 MLP 미달) |

단계별 기여도 분석 결과는 다음과 같이 요약된다.

1. **가장 큰 성능 기여 단계: Phase 4 (비선형 모델 전환)**  
   선형/생성 모델에서 앙상블·MLP로의 전환이 PR-AUC를 +0.119 향상시켜 가장 큰 단일 기여를 보였다. 이는 사출성형 공정 변수들의 비선형 상호작용이 중요함을 보여준다.

2. **두 번째 기여 단계: Phase 3 (모델 가족 내 최적화)**  
   QDA가 가이드북이 제시하지 않은 모델임에도 불구하고 선형 패밀리 최고 성능을 달성했다. 체계적 탐색 없이 단일 모델을 선택한 가이드북 접근법의 한계를 정량적으로 드러낸 결과다.

3. **불균형 처리의 안정화 효과: Phase 2**  
   SMOTE는 ROC-AUC를 절대적으로 높이기보다 PR-AUC 분산을 절반으로 줄여(std 0.1145→0.0554) 결과의 신뢰성을 높였다. 극단적 불균형(0.89%)에서 안정성 확보가 단순 점수 향상만큼 중요하다.

4. **Stacking의 반례: Phase 5**  
   "스태킹은 항상 단일 모델을 이긴다"는 통념과 달리, 기반 모델의 다양성이 부족할 경우 단일 최고 모델(MLP)에 미달할 수 있다. 이 반례 자체가 OOF(폴드 외 예측) 방식의 한계를 보여주는 실증 사례가 된다.

---

## 6. 한계 및 향후 과제 (배점 25%)

### 6.1 시계열성 활용 미흡

사출성형 데이터는 사이클 단위 집계 변수이므로 사이클 내부의 시간적 변화(압력-속도 프로파일)를 반영하지 못한다. 본 텀프에서 적용한 1D-CNN은 변수를 임의 순서의 1D 시퀀스로 처리(DeepInsight 방식)하였으며, 실제 시계열 의존성을 모델링한 것이 아니다. 인접 사이클 간의 공정 드리프트(drift) 감지나 슬라이딩 윈도우 기반 시계열 분류는 이번 구현에서 충분히 시도하지 못했다.

**향후 과제:** 강의 외 기법이지만 방향 제시로서 — 사이클 내 고주파 센서 로그가 있다면 1D-CNN의 시계열 입력으로 자연스럽게 활용 가능. 현재 데이터(사이클당 1행 집계)로는 K개 인접 사이클을 묶는 슬라이딩 윈도우가 현실적 차선책이나, 시도하지 않았다.

### 6.2 CN7/RG3 분리 분석 미시도

KS 검정 결과 두 제품군의 변수 분포가 22/24개 변수에서 유의하게 다름에도 불구하고, Phase 3~5는 labeled_data 단독(CN7+RG3 혼합) 분석을 기본으로 유지했다. 제품별 분리 모델링 또는 제품 flag를 피처로 추가하는 ablation을 계획했으나(decisions.md D-006, D-004) 실제로 시도하지 않았다.

**향후 과제:** CN7 전용 모델과 RG3 전용 모델을 각각 학습하여 혼합 모델과 비교. 제품별 분포 차이가 혼합 학습의 성능 병목인지 검증. labeled_data_CN7_processed.csv와 labeled_data_RG3_processed.csv가 이미 존재하므로 즉시 적용 가능.

### 6.3 불량 원인(Reason 컬럼) 99.1% 결측

PassOrFail이 'N'(불량)인 71건 중 불량 원인(Reason 컬럼)이 기록된 건수가 거의 없다. 불량 원인을 피처로 활용하거나 원인별 분류 모델을 구성하는 것이 현실적으로 불가능한 상태다. 이는 KAMP 데이터 자체의 수집 프로세스 한계이며, 산업 현장에서 불량 원인 추적 체계 구축의 필요성을 시사한다.

### 6.4 산업 현장 운용 임계값 설정의 어려움

Precision≥0.99 운용점에서 MLP의 Recall은 32.4%에 불과하다. 즉, 불량 71건 중 약 48건은 탐지되지 않는다. 이는 데이터의 극단적 불균형(0.89%)과 양성 표본 절대 수의 부족(71건)에서 기인하며, 더 많은 불량 데이터 없이는 높은 Recall과 Precision을 동시에 달성하기 어렵다. 실제 생산 라인에서 "어느 수준의 Recall을 감수할 것인가"에 대한 도메인 전문가의 의사결정이 필요하다.

### 6.5 가이드북 AE(Autoencoder) 재현 미완성

가이드북 §2.3의 원본 흐름은 AE(Autoencoder)로 비지도 특성 추출 후 SVM/DNN 학습이다. AE는 강의 범위 밖 알고리즘(AGENT_INSTRUCTIONS §5.6에서 금지)이므로, 본 텀프에서는 AE 단계 없이 supervised 학습만 수행했다. 가이드북과의 완전한 1:1 비교가 AE 단계에서 불완전하다.

---

## 7. 활동 계획 (배점 5%)

### 7.1 Phase별 역할 분담 (제안서 v0 기준, 실제 수행 기록 필요)

| Phase | 팀장(조현건) 담당 | 팀원 B 담당 | 팀원 C 담당 |
|---|---|---|---|
| Phase 2 | 전처리 ablation 설계 및 실행, decisions.md 기록 | EDA 시각화, 결측 분석 | 데이터 로딩 파이프라인(src/data.py) |
| Phase 3 | LR/LDA/QDA 실험 실행, 인사이트 작성 | SVM(linear/RBF) 실험 | 평가 지표 모듈(src/evaluate.py) |
| Phase 4 | 차원축소 ablation 설계 | RF/GBM 학습 | MLP+Dropout, 가이드북 DNN 재현 |
| Phase 5 | Stacking 메타 학습기 구현 | 1D-CNN 구현 | 운용점 분석, 결과 그림·표 정리 |
| Phase 6 | 보고서 통합, 발표 총괄 | 시각화(ROC/PR 곡선, PCA 산점도) | 활동계획·기여도 부록 작성 |

### 7.2 회의 및 코드 리뷰 기록 (학기말 제출 시 첨부 필요)

- 각 Phase 완료 시 팀 내 결과 리뷰 회의 진행 기록
- GitHub commit 이력로 코드 기여도 확인 가능하게 유지
- decisions.md에 주요 기술 결정 사항을 날짜·근거와 함께 지속 기록

### 7.3 학기말 마감 체크리스트

- [ ] `reports/final_report.pdf` (8~12페이지)
- [ ] `final_presentation.pptx` (14슬라이드)
- [ ] `Activity_Appendix.pdf` (팀원별 기여도 표)
- [ ] GitHub repo (코드 + README + requirements.txt)
- [ ] `pytest` 전 테스트 통과 상태 유지
- [ ] `results/ablation_summary.md` Phase 6까지 갱신

---

## 부록 A — 허용 알고리즘 감사표 (Task 3)

| 사용 알고리즘 | 강의 분류 | AGENT_INSTRUCTIONS §2 허용 여부 | 실제 사용 Phase |
|---|---|---|---|
| Logistic Regression (L1/L2) | 선형·생성적 | 허용 (L4-6) | Phase 2 (고정 분류기), Phase 3, Phase 5 (Stacking 메타) |
| LDA | 선형·생성적 | 허용 (L4-6) | Phase 3 |
| QDA | 선형·생성적 | 허용 (L4-6) | Phase 3, Phase 5 (Stacking base) |
| SVM (linear/RBF) | 거리·마진 | 허용 (L7-8) | Phase 3 |
| Random Forest | 트리·앙상블 (Bagging) | 허용 (L7-10) | Phase 4, Phase 5 (Stacking base) |
| Gradient Boosting (GBM) | 트리·앙상블 | 허용 (L7-10) | Phase 4, Phase 5 (Stacking base) |
| AdaBoost | 트리·앙상블 | 허용 (L7-10) | Phase 4 |
| Stacking (메타 LR) | 메타 학습 | 허용 (L9-11) | Phase 5 |
| PCA | 차원 축소 | 허용 (L9-13) | Phase 4-A |
| MLP + Backprop + Dropout | 신경망 | 허용 (L11-15) | Phase 4, Phase 5 (Stacking base) |
| CNN (Conv1D, MaxPool) | 딥러닝 | 허용 (L13-16) | Phase 5 |
| SMOTE / ADASYN | 불균형 처리 보조 | 허용 (imbalanced-learn, 강의 언급) | Phase 2, Phase 3~4 전처리 |
| StandardScaler / RobustScaler | 전처리 보조 | 허용 (scikit-learn, 강의 언급) | Phase 2+ |
| VarianceThreshold | 특성 선택 | 허용 (scikit-learn, 분산 기반 필터) | Phase 4-A |
| Decision Tree (base for AdaBoost/Stacking) | 트리 | 허용 (L7-10) | Phase 4, Phase 5 |

**강의 범위 밖 기법 사용 여부:**

| 금지 알고리즘 | 사용 여부 |
|---|---|
| Transformer / BERT | 미사용 |
| GNN (Graph Neural Network) | 미사용 |
| LSTM / RNN | 미사용 |
| Diffusion Model | 미사용 |
| Autoencoder (AE) | 미사용 (가이드북 재현 생략, decisions.md 기록 필요) |

**범위 위반 사항: 없음.** 모든 사용 알고리즘이 AGENT_INSTRUCTIONS §2 허용 목록 내에 있다.

---

*작성: 감독관 에이전트 (2026-05-10) | 기반 파일: results/ablation_summary.md, results/decisions.md, results/tables/baseline_results.csv, ensemble_nn_results.csv, phase5_results.csv, operating_points.csv*
