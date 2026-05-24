# 방법 C 분석 보고서 — Semi-supervised Pseudo-labeling

**작성일:** 2026-05-24  
**대상 노트북:** `08_pseudo_labeling.ipynb`, `08_method_c_v2.ipynb`, `08a~08i_eda.ipynb`  
**담당:** 조현건

---

## 목차

1. [배경 및 연구 동기](#1-배경-및-연구-동기)
2. [데이터 개요 및 핵심 제약](#2-데이터-개요-및-핵심-제약)
3. [EDA — 데이터 구조 파악](#3-eda--데이터-구조-파악)
4. [이상탐지 기법 9종 비교 (NB08 Step 2)](#4-이상탐지-기법-9종-비교)
5. [첫 번째 시도가 실패한 이유 — 기계정지 오염](#5-첫-번째-시도가-실패한-이유--기계정지-오염)
6. [개발 과정 — 세 번의 시도](#6-개발-과정--세-번의-시도)
7. [최종 구현 설계 세부 사항](#7-최종-구현-설계-세부-사항)
8. [실험 결과 — 5-fold CV](#8-실험-결과--5-fold-cv)
9. [n_pseudo Ablation](#9-n_pseudo-ablation)
10. [Self-training Threshold Ablation](#10-self-training-threshold-ablation)
11. [종합 결론 및 방법론적 기여](#11-종합-결론-및-방법론적-기여)
12. [노트북 흐름 안내](#12-노트북-흐름-안내)

---

## 1. 배경 및 연구 동기

### 1.1 왜 준지도 학습인가

KAMP 사출성형 데이터셋에서 레이블이 붙은 데이터는 7,996행뿐이다. 그 중 실제 불량은 71개(불량률 0.89%)에 불과하다. 분류기를 지도학습으로 훈련하면 이 71개로 "불량이란 무엇인가"를 학습해야 하는데, 이는 구조적으로 불안정하다. 단일 fold에서 불량 샘플이 약 56개(train)와 15개(val)로 나뉘는 상황에서 분류기가 불량의 일반적 패턴을 포착하기 어렵다.

반면 레이블이 없는 데이터는 795,315행이 존재한다. 이 중 동일 장비·부품으로 필터링해도 71,180행이 남는다 — 레이블 데이터의 약 9배다. unlabeled에서 불량 후보를 신뢰성 있게 선별할 수 있다면, 불량 학습 샘플을 크게 늘려 분류기 성능을 보완할 수 있다.

### 1.2 방법 C의 전체 아이디어

#### 1.2.1 Unlabeled에서 불량 후보 선별

**pseudo-label이란** unlabeled 데이터에 사람이 직접 달지 않고 알고리즘이 추정한 레이블이다. 방법 C는 이상탐지 기법으로 unlabeled에서 "불량처럼 보이는 샘플"을 골라내 pseudo-label을 붙이는 방식을 쓴다.

먼저 unlabeled 795K 중 동일 장비·부품으로 필터링한 71,180행을 대상으로 이상탐지 기법 9종(Isolation Forest, LOF, HBOS, COPOD, GMM, K-Means, OC-SVM, Elliptic Envelope, KNN Distance)을 실행한다. 각 기법은 labeled 7,996행을 기준으로 ROC-AUC를 평가해 순위를 매긴다. 최고 성능 기법으로 unlabeled 각 샘플에 이상 점수를 부여하고, 상위 **contamination%**(이상으로 의심할 비율)에 해당하는 샘플을 불량 pseudo-label로 지정한다. contamination은 labeled 불량률 0.89%와 맞추는 것이 기본값이다 — 같은 공정이면 비슷한 비율로 불량이 발생할 것이라는 전제.

그런데 이상탐지 점수만으로 선별하면 "실제 불량과 비슷한 샘플"이 아니라 "단순히 분포에서 벗어난 샘플"이 뽑힌다. 이것이 첫 번째 시도가 실패한 이유다(Section 5 참조). 최종 구현에서는 선별 기준을 4가지 전략으로 다변화했다.

| 전략 | 선별 기준 | 직관 |
|---|---|---|
| A: EllipticEnvelope | labeled 양품 분포의 공분산 기준 Mahalanobis 거리가 가장 먼 샘플 | 양품과 가장 다른 샘플 |
| B: RF Confidence | RF를 먼저 labeled로 학습한 뒤, unlabeled에서 불량 확률이 가장 높은 샘플 | 분류기가 불량이라고 확신하는 샘플 |
| C: Direction Projection | 표준화 공간에서 "양품 평균→불량 평균" 방향으로 가장 멀리 투영된 샘플 | 불량과 같은 방향으로 이탈한 샘플 |
| D: kNN Defect | labeled 불량 71개와 가장 가까운 샘플(kNN 거리 기준) | labeled 불량과 가장 닮은 샘플 |

각 전략이 뽑은 pseudo-defect가 실제로 labeled 불량과 피처 분포가 비슷한지는 **align_score**로 확인한다. 25개 피처 중 불량과 같은 방향으로 이탈한 피처 비율(방향 일치율, 50%=무작위 수준)과 코사인 유사도를 동시에 계산한다.

#### 1.2.2 Labeled + Pseudo-defect 혼합 학습

선별한 pseudo-defect N개를 labeled 불량 71개에 더해 분류기(RF, MLP)를 학습한다. N은 고정하지 않고 [50, 100, 200, 354, 500, 750, 1000, 2000]을 따로 실험해 최적값을 찾는다.

학습할 때 두 가지를 신경 써야 한다.

**SMOTE 순서**: 불량 클래스가 labeled 불량 + pseudo-defect로 늘어났기 때문에, SMOTE(합성 소수 클래스 오버샘플링)를 적용하면 불량이 많아진 만큼 k_neighbors를 `min(5, 불량수-1)`로 자동 조정해야 에러가 나지 않는다. SMOTE는 fold 안에서만 fit하고, train 데이터에만 적용한다. val fold에는 절대 손대지 않는다.

**스케일러**: 전략 선택에 쓴 스케일러(`sc_sel`, labeled 전체로 fit한 StandardScaler)와 CV 학습에 쓰는 스케일러(`sc_f`, fold train set으로 fit)는 별개다. 전략 선택 때는 labeled 전체 분포 기준으로 unlabeled를 변환해야 일관성이 있고, CV 학습 때는 val 정보 누수를 막으려면 train fold로만 fit해야 한다. 두 스케일러를 혼용하면 안 된다.

#### 1.2.3 5-fold CV로 효과 검증

Phase 1~6 전체와 동일한 fold_*.npy 인덱스를 사용해 baseline(labeled only)과 각 전략을 비교한다. fold별로 ROC 차이를 5개 구하고, Wilcoxon signed-rank test(단측)로 그 차이가 0보다 유의하게 큰지 검정한다. p < 0.05면 **IMPROVED\***, p가 높아도 평균 차이가 양수면 **trend**, 개선이 없으면 **ns**로 표기한다.

fold가 5개뿐이라 검정력이 낮다. 검정력이 낮으면 실제로 개선이 있어도 p < 0.05 달성이 어렵다는 뜻이다. 그런 조건에서도 전략 D(RF 기준)가 p=0.031을 달성했다.

결국 성패를 가르는 건 pseudo-label의 품질이다. labeled 불량과 비슷하지 않은 샘플을 아무리 많이 추가해도 분류기는 나빠지거나 제자리다 — 이 점은 Section 5에서 수치로 확인할 수 있다.

---

## 2. 데이터 개요 및 핵심 제약

### 2.1 데이터 구성

| 구분 | 행 수 | 비고 |
|---|---|---|
| labeled_data.csv | 7,996 | 레이블 있음, 불량 71개 |
| unlabeled_data.csv | 795,315 | 레이블 없음 |
| CN7/RG3+동일장비 필터 후 | 71,180 | 방법 C의 pseudo-label 후보 풀 |
| 생산상태 필터 후 | 39,870 | 기계정지(44%) 제거, 두 번째 시도부터 적용 |

**필터 기준 (CN7/RG3):**
- `EQUIP_NAME` ∈ {'650톤-우진', '1800TON-우진', '650톤-우진2호기'}
- `PART_NAME` ∈ {CN7 W/S SIDE MLD'G RH/LH, RG3 MOLD'G W/SHLD LH/RH}

이 필터를 적용하는 이유는 **도메인 일관성** 때문이다. 사출성형 공정은 금형·장비 조합에 따라 공정 파라미터 분포가 완전히 다르다. Labeled 데이터와 다른 장비·부품에서 수집된 unlabeled 데이터로 pseudo-label을 만들면, 그 pseudo-label은 labeled 불량과 아무 관련이 없는 "다른 공정의 정상 범위 이탈"이 된다.

### 2.2 유효 피처 25개

총 컬럼에서 다음을 제외하고 25개 수치 피처를 사용한다.

| 제거 이유 | 해당 컬럼 |
|---|---|
| 분산=0 (상수) | Mold_Temperature_1,2,5,6,7,8,9,10,11,12 (10개) |
| Near-zero variance | Barrel_Temperature_7 (1개) |
| 식별자/메타 | PART_FACT_SERIAL 등 |

25개 피처는 사출 시간, 위치, 속도, 압력, 배럴 온도, 금형 온도 등 사출성형 공정 전반을 포괄한다.

### 2.3 극심한 클래스 불균형

불량률 0.89%는 단순히 "불균형"이 아니라 **구조적 제약**이다.

- 5-fold CV에서 train fold당 불량 ~56개, val fold당 ~15개
- PR-AUC(Precision-Recall AUC)는 Accuracy나 ROC-AUC보다 불균형 데이터에서 더 엄격한 지표다. Random baseline의 PR-AUC = 불량률 = 0.89%이므로, PR-AUC 0.4 이상이면 실질적으로 의미 있는 탐지 성능이다.
- **평가지표로 Accuracy는 절대 쓰지 않는다.** 모든 샘플을 양품으로 예측해도 Accuracy 99.1%가 나온다.

---

## 3. EDA — 데이터 구조 파악

### 3.1 PCA 분석 결과

```
PC1: 58.8%  PC2: 19.7%  PC3: 10.2%  누적(PC1~3): 88.8%
```

상위 3개 주성분으로 전체 분산의 88.8%를 설명한다. PC1이 압도적으로 높은 이유는 Clamp_Open_Position, Average_Screw_RPM 등 스케일이 큰 피처들이 raw 공간에서 분산을 지배하기 때문이다(이것이 최종 구현에서 표준화 공간을 쓰는 이유 — Section 7 참조).

- 불량 71개 모두 PC1·PC2 기준으로 양품의 범위 안에 혼재 (100% 겹침) — **불량은 선형적으로 분리되지 않는다.**
- Unlabeled 71,180행 중 48.4%가 labeled 양품의 PC1 범위 안에 분포 → 필터링 후 도메인 일치 확인

### 3.2 UMAP 분석 결과

```
양품 퍼짐: 12.192  불량 퍼짐: 9.143  (비율 0.75x)
Unlabeled 100%가 labeled 양품 UMAP bounding box 내 존재
불량 71개 중 62개 주변 1.0 이내에 unlabeled 이웃 존재
```

퍼짐 비율 0.75x — 불량이 양품보다 좁게 군집하는 게 아니라 여러 패턴으로 분산돼 있다. 단일 이상 패턴이 없으니 클러스터링 기반 탐지는 구조적으로 불리하다. Unlabeled 100%가 labeled 양품 bounding box 내에 들어온다는 것은 CN7/RG3 필터가 도메인을 제대로 맞췄다는 뜻이다. 불량 62개(87%) 주변 1.0 이내에 unlabeled 이웃이 존재한다는 수치가 전략 D(kNN from Labeled Defects) 설계의 출발점이다.

### 3.3 EDA 노트북(08a~08i)의 역할

각 EDA 노트북은 이상탐지 기법 하나를 동일한 구조로 심층 분석한다.
- PCA/UMAP에 이상 점수 overlay → 어느 공간의 샘플이 높은 점수를 받는지
- 피처별 이상 점수 상관도 → 기법의 "신호원" 피처 파악
- 병렬 좌표계 → 상위 이상 샘플과 실제 불량의 피처 패턴 비교
- KS 검정 → 양품/불량 이상 점수 분포 차이의 통계적 유의성

특히 **08h(Elliptic Envelope)**와 **08a(COPOD)** 노트북이 핵심이다. 전자는 최고 성능(ROC=0.8938)을 내지만 labeled 양품이 필요하다는 반지도 성격을 가지고, 후자는 완전 비지도 기법 중 최고(ROC=0.8793)다.

---

## 4. 이상탐지 기법 9종 비교

`08_pseudo_labeling.ipynb` Section 5 결과다. 9개 기법 전체를 동일한 labeled 7,996행으로 평가했다.

### 4.1 전체 결과 표

| 순위 | 기법 | ROC-AUC | PR-AUC | 학습 데이터 | 성격 |
|---|---|---|---|---|---|
| 1 | **Elliptic Envelope** | **0.8938** | **0.1397** | labeled 양품 7,925개 | 반지도 |
| 2 | [ref] OC-SVM (Phase 7) | 0.8814 | 0.1765 | labeled 양품 | 반지도 |
| 3 | **COPOD** | **0.8793** | **0.1042** | unlabeled 71K | 완전 비지도 |
| 4 | Isolation Forest | 0.5293 | 0.0357 | unlabeled 71K | 완전 비지도 |
| 5 | HBOS | 0.4776 | 0.0362 | unlabeled 71K | 완전 비지도 |
| 6 | K-Means (k=20) | 0.4757 | 0.0383 | unlabeled 71K | 완전 비지도 |
| 7 | OC-SVM (unlabeled) | 0.4593 | 0.1327 | unlabeled 5K | 완전 비지도 |
| 8 | LOF | 0.3827 | 0.0206 | unlabeled 10K | 완전 비지도 |
| 9 | KNN distance | 0.3685 | 0.0305 | unlabeled 10K | 완전 비지도 |
| - | GMM (k=5) | 0.3081 | 0.0068 | unlabeled 71K | 완전 비지도 |

### 4.2 해석

**패턴 1 — "labeled 양품을 아는가"가 성능을 가른다**

Elliptic Envelope와 Phase 7 OC-SVM은 "양품이 어디에 있는지"를 직접 학습한다. 나머지 완전 비지도 기법은 unlabeled 분포에서 이상을 찾는다. 그런데 unlabeled 데이터에는 기계정지 레코드가 44%나 섞여 있어서, 기법들이 실제 공정 불량보다 기계정지 상태를 이상으로 인식한다. 결과적으로 완전 비지도 기법들의 성능이 낮다.

**패턴 2 — COPOD는 완전 비지도 기법 중 예외적**

ROC=0.8793은 완전 비지도 기법 중 압도적이다. COPOD는 각 피처의 empirical CDF와 copula 결합 구조를 모델링해 이상 점수를 계산한다. KS 검정 결과 양품/불량 이상 점수 분리도가 KS=0.6736으로 가장 높다. 분포 가정 없이 다변량 의존 구조를 포착하는 특성이 사출성형 데이터의 복잡한 피처 간 상관에 잘 맞은 것으로 보인다.

**패턴 3 — K-Means의 ROC=0.47은 Phase 7 결과 재현**

Phase 7 보고서에서 K-Means ROC=0.47로 "성능 없음"으로 기록된 결과가 k=20에서 정확히 재현됐다. 이는 CN7/RG3 필터링 후에도 기계정지 오염이 남아있음을 뜻한다. 기계정지 레코드(온도=0, RPM=0)가 하나의 클러스터를 형성하고, centroid 거리 기준으로는 그것이 "이상"이 된다. 실제 공정 불량과 무관하다.

### 4.3 최적 기법 선택 및 Pseudo-label 생성

**최적 기법: Elliptic Envelope** (ROC-AUC 기준, labeled normal 학습)

```
Pseudo-label 생성 결과:
  불량 후보: 633개 (0.89% threshold)
  정상 후보: 70,547개
  Labeled 불량 71개 대비 8.9배
```

threshold는 labeled 불량률 0.89%와 동일하게 설정했다. 이론적으로 가장 자연스러운 선택이다 — 동일한 공정에서 같은 비율의 불량이 발생한다고 가정.

---

## 5. 첫 번째 시도가 실패한 이유 — 기계정지 오염

### 5.1 pseudo-label 검증 결과

`08_pseudo_labeling.ipynb`에서 생성한 pseudo-defect 633개와 labeled 불량 71개를 피처별로 비교했다.

```
방향성 일치도: 7/25 = 28%  (기준: 50%=무작위, 100%=완전일치)
```

**28% < 50%** — 이것은 pseudo-불량이 실제 불량과 반대 방향으로 이탈한다는 뜻이다.

### 5.2 불일치 피처 상세

| 피처 | Pseudo 평균 | Labeled 불량 평균 | Labeled 양품 평균 | 진단 |
|---|---|---|---|---|
| Average_Screw_RPM | **0.00** | 258.39 | 148.04 | **기계정지** |
| Average_Back_Pressure | **0.00** | 62.42 | 59.55 | **기계정지** |
| Max_Screw_RPM | **0.00** | 30.79 | 30.70 | **기계정지** |
| Barrel_Temperature_1 | **0.90** | 280.50 | 277.43 | **기계정지** |
| Max_Back_Pressure | 4.05 | 49.08 | 40.63 | **기계정지** |
| Plasticizing_Time | 2.97 | 15.06 | 16.22 | 기계정지 영향 |

RPM=0, 배럴온도≈0°C — **기계가 멈춰 있는 상태**다. Elliptic Envelope가 이상으로 탐지한 것들이 공정 불량이 아니라 기계가 꺼진 레코드였다.

### 5.3 왜 이런 일이 벌어졌나

1. CN7/RG3 필터 후 unlabeled 71,180행 중 기계정지 레코드가 **44% = 31,310개** 섞여 있다
2. Elliptic Envelope는 labeled 양품(기계 가동 중)의 공분산으로 Mahalanobis 거리를 계산한다
3. 기계정지 레코드는 Barrel_Temperature=0, RPM=0 — 가동 중 양품 분포에서 완전히 벗어난다
4. 결과적으로 Elliptic Envelope가 기계정지 레코드를 최고 이상으로 탐지

### 5.4 Step 4 결과 — pseudo-label 오염의 영향

```
Config A (labeled only):   QDA ROC=0.9344  RF ROC=0.9271  MLP ROC=0.9494
Config B (labeled+pseudo): QDA ROC=0.8226  RF ROC=0.9328  MLP ROC=0.9325

변화:
  QDA: -0.1118  ← 분류기 붕괴 수준
  RF:  +0.0057  ← 통계적 비유의 (std=0.0287 >> 개선폭)
  MLP: -0.0169  ← 소폭 하락
```

**QDA가 가장 심각하게 붕괴하는 이유:**

QDA는 클래스별 공분산 행렬로 결정 경계를 정의한다. 진짜 불량(온도≈280°C, RPM≈258)과 pseudo-불량(온도≈0°C, RPM≈0)을 같은 "불량" 클래스로 묶으면, 불량 클래스의 공분산 행렬이 두 개의 완전히 다른 분포의 혼합이 된다. 결정 경계가 왜곡되어 ROC -0.11이라는 붕괴 수준 하락이 발생했다.

---

## 6. 개발 과정 — 세 번의 시도

방법 C는 세 번에 걸쳐 구현됐다. 팀 내 논의나 코드 주석에서 "v1/v2/v3"로 부르는 경우를 대비해 각 단계의 의미를 정리한다.

- **첫 번째 시도 (v1)**: `08_pseudo_labeling.ipynb`. 이상탐지 9종을 비교한 뒤 최고 성능 기법으로 pseudo-label을 생성했다. Section 5에서 설명한 기계정지 오염 문제로 실패했다.
- **두 번째 시도 (v2)**: 첫 번째 실패 이후 기계정지 필터를 추가하고 4가지 선택 전략(A/B/C/D)을 설계한 버전. 별도 노트북이 아닌 같은 코드베이스에서 이어진 중간 단계다. 치명적 설계 결함이 발견돼 결과를 그대로 쓸 수 없었다.
- **최종 구현 (v3)**: `08_method_c_v2.ipynb`. 두 번째 시도의 결함을 전부 수정한 버전이다. 파일명이 "v2"인 것은 방법 C 전용 두 번째 노트북이라는 의미이고, 방법론상으로는 세 번째 시도다.

### 6.1 첫 번째 → 두 번째 시도: 기계정지 필터 추가

첫 번째 시도는 CN7/RG3 필터(동일 장비·부품 조건)만 적용하고 이상탐지로 바로 pseudo-label을 생성했다. Section 5에서 확인한 대로 결과물 대부분이 기계가 꺼진 레코드(RPM=0, 온도≈0°C)였다.

두 번째 시도에서 추가한 것이 생산상태 필터다.

```python
# 기계가 실제로 가동 중인 레코드만 남긴다
mask_prod = (X_unl_df['Barrel_Temperature_1'] >= 200) & (X_unl_df['Average_Screw_RPM'] > 0)
```

- 기계정지 제거: 31,310개 (44.0%)
- 생산상태 유지: 39,870개 (56.0%)

이상탐지 점수 하나에만 의존하던 방식도 바꿔 4가지 선택 전략(A~D)을 새로 설계했다. 그러나 코드에 치명적 결함이 남아 있었다.

### 6.2 두 번째 → 최종 구현: 설계 결함 8개 수정

두 번째 시도는 "전략 B에서 ROC +0.032 IMPROVED"를 주장했으나, 검토 과정에서 8개 문제가 발견됐다.

| 번호 | 내용 | 심각도 |
|---|---|---|
| 1 | **스케일러 불일치**: 전략 선택 단계는 unlabeled 전체로 fit한 스케일러, CV 학습 단계는 fold별 스케일러 — 두 단계가 다른 공간에서 계산됨 | 치명적 |
| 2 | **전략 B 정보 누수**: teacher RF가 val fold를 포함한 labeled 전체(7,996행)로 학습 → val fold 정보가 pseudo-label 선택에 반영됨 | 치명적 |
| 3 | **align_score 스케일 무시**: Clamp_Open_Position(범위≈200)과 작은 피처(범위≈0.1)를 동등하게 취급 | 치명적 |
| 4 | **방향벡터 raw space**: 표준화 없이 raw 값으로 내적 → Clamp_Open_Position 하나가 내적의 99% 지배 | 치명적 |
| 5 | 통계 검정 없이 IMPROVED 주장 | 중요 |
| 6 | 전략 C만 n ablation, 나머지 전략은 고정값 | 중요 |
| 7 | Self-training threshold=0.80 근거 없음 | 중요 |
| 8 | fillna(0) 검증 없음 | 경미 |

전략 B의 ROC +0.032가 **2번(정보 누수)** 때문이었음은 최종 구현에서 수치로 확인됐다. fold별 teacher로 차단하자 -0.003 ns(p=0.781)로 붕괴했다.

### 6.3 최종 구현에서 바꾼 것

1. `sc_sel = StandardScaler().fit(X_lab)` — labeled 전체로 학습한 스케일러를 전략 선택·CV 모두에 통일
2. `run_cv_B_clean()` — fold별 teacher RF (val fold 정보 완전 차단)
3. 표준화 공간에서 align_score 계산 + RF importance 가중치 옵션
4. `defect_dir_sc` — 표준화 공간의 방향벡터
5. Wilcoxon signed-rank test (단측, α=0.05)
6. 전략 B·C·D 모두 n ablation
7. Self-training threshold sweep [0.5, 0.6, 0.7, 0.8]
8. 결측 패턴 분석 후 fillna(0) 안전 확인

---

## 7. 최종 구현 설계 세부 사항

### 7.1 두 개의 스케일러

```python
sc_vis = StandardScaler().fit(X_unl)   # 시각화 전용 (PCA 등)
sc_sel = StandardScaler().fit(X_lab)   # 전략 선택 + CV 학습 전용
```

`sc_vis`는 unlabeled 전체 분포 기준이라 PCA 시각화에 적합하고, `sc_sel`은 labeled 분포 기준이라 전략 선택과 CV 학습 양쪽에서 공간 기준을 통일시킨다.

`sc_vis`를 전략 선택에 쓰면 기준이 어긋난다. unlabeled 71K에 기계정지가 44% 섞여 있으면 RPM 평균이 낮고 분산이 크다. 이 스케일러로 labeled를 변환하면 labeled 양품이 "높은 RPM" 방향으로 치우쳐 보여 전략 선택이 왜곡된다.

### 7.2 방향벡터 표준화 공간 (전략 C)

```python
# Raw 방향벡터: 피처 raw 차이
defect_dir_raw = X_lab[mask_d].mean(0) - X_lab[mask_n].mean(0)

# 표준화 방향벡터 (sc_sel 공간)
Xl_sc = sc_sel.transform(X_lab)
defect_dir_sc = Xl_sc[mask_d].mean(0) - Xl_sc[mask_n].mean(0)
```

Raw 방향벡터를 계산하면:

```
Clamp_Open_Position: 불량평균(358) - 양품평균(546) = -187.9  ← 압도적으로 크다
Average_Screw_RPM:   불량평균(258) - 양품평균(148) = +110.3
Max_Back_Pressure:   불량평균(49)  - 양품평균(41)  = +8.4
```

내적(projection) 계산 시 Clamp_Open_Position의 값이 187.9라서 다른 24개 피처의 신호가 묻힌다. 단변량 기준으로는 유의한 차이지만, 그것만으로 "불량 방향"을 정의하면 나머지 24개 피처의 정보가 무시된다.

표준화 후:

```
Max_Back_Pressure: 1.228  ← 이제 최고
Cycle_Time:        1.118
Average_Screw_RPM: 0.843
Clamp_Open_Position: -0.795  ← raw에서 지배하던 피처가 정상 범위로 내려옴
```

표준화 공간에서는 각 피처의 불량/양품 간 차이가 표준편차 단위로 표현되므로, 25개 피처가 공평하게 방향벡터에 기여한다. 두 번째 시도의 cosine 0.969 → 최종 구현의 cosine 0.150 — raw 공간에서 Clamp_Open 하나에 의해 방향이 거의 결정되던 문제가 해소됐다.

### 7.3 전략 B의 누수 차단 (run_cv_B_clean)

두 번째 시도의 전략 B는 labeled 7,996개 전체로 teacher RF를 학습했다. 5-fold CV에서 val fold가 teacher RF 학습에 포함되어 정보 누수가 발생했다.

최종 구현의 `run_cv_B_clean()`:
```python
for fold in folds:
    tr_idx, val_idx = fold["train"], fold["val"]
    # teacher RF는 train fold만으로 학습
    rf_teacher.fit(X_tr_l_sc, y_tr_l)
    # unlabeled에서 pseudo-label 선택
    prob_teacher = rf_teacher.predict_proba(Xu_f_sc)[:, 1]
    idx_pseudo = np.argsort(prob_teacher)[-n_pseudo:]
    # student는 train + pseudo로 학습, val로 평가
    student.fit(...)
```

fold별 teacher는 val fold 56개 이상을 보지 않는다. 두 번째 시도의 "IMPROVED +0.032"가 이 누수 때문이었음이 최종 구현에서 확인됐다. 누수 차단 후 -0.003 ns(p=0.781).

### 7.4 align_score (표준화 공간, 중요도 가중)

```python
def align_score(pseudo_X_raw, uniform=True):
    Xp_sc = sc_sel.transform(pseudo_X_raw)
    dv = defect_dir_sc  # 표준화 공간 방향벡터
    pv = Xp_sc.mean(0) - Xl_sc[mask_n].mean(0)  # pseudo-defect의 이탈 방향

    if uniform:
        # 각 피처가 같은 방향으로 이탈하는지 (0/1 투표)
        dir_match = float(np.mean((pv * dv) > 0))
        cos = float((dv @ pv) / (||dv||·||pv|| + 1e-9))
    else:
        # RF importance로 중요한 피처에 더 큰 가중치
        w = feat_imp / feat_imp.sum()
        dir_match = float(np.sum(w * ((pv * dv) > 0)))
        cos = 가중 cosine
```

pseudo-defect들의 집단적 이탈 방향과 labeled 불량 방향의 일치도를 정량화한다.

- **dir_match**: 25개 피처 중 불량과 같은 방향으로 이탈한 피처 비율 (50%=무작위)
- **cosine similarity**: 두 벡터의 각도 (1=완전 일치, 0=무관, -1=반대)

### 7.5 4가지 전략 요약

| 전략 | 방법 | align_score (uniform/weighted) | 특징 |
|---|---|---|---|
| **A: EllipticEnvelope** | sc_sel 공간에서 Mahalanobis 거리 | 52% / 51%, cos=-0.001 | 보통 |
| **B: RF Confidence** | fold별 teacher RF 확신도 | 64% / 72%, cos=0.365 | 양호, 누수 차단 필수 |
| **C: Direction Projection** | 표준화 방향벡터에 내적 투영 | 60% / 61%, cos=-0.077 | 양호 |
| **D: kNN Defect** | labeled 불량과 kNN 거리 기준 선택 | **100% / 100%, cos=0.879** | **우수** |

전략 D의 align_score가 압도적인 이유는 **선택 기준 자체가 labeled 불량과의 거리**이기 때문이다. labeled 불량과 가장 가까운 unlabeled 샘플을 선택하니, 그 샘플들의 피처 분포가 labeled 불량과 유사한 것은 수학적으로 자명하다. 누수도 없다 — 이 전략은 labeled val fold 정보를 일절 사용하지 않는다.

### 7.6 통계 검정 — Wilcoxon signed-rank test

5-fold CV에서 각 fold별로 base ROC와 전략 ROC의 차이를 계산하면 5개의 차이값이 나온다. 이 5개 차이값의 분포가 0보다 유의하게 크면 "IMPROVED"다.

```python
dr = roc_f - base_roc_f  # shape (5,)
_, p_roc = wilcoxon(dr, alternative="greater")
```

Wilcoxon signed-rank test는 t-검정과 달리 정규성 가정이 필요 없다. fold가 5개뿐이므로 검정력이 낮아 p<0.05를 달성하기 어렵다. 그럼에도 전략 D(RF)는 p=0.031로 유의 임계를 넘었다.

---

## 8. 실험 결과 — 5-fold CV

`08_method_c_v2.ipynb` Section 7 결과다. RF와 MLP 두 분류기로 실험했다.

### 8.1 RF 결과

| 전략 | ROC-AUC | PR-AUC | Δ ROC | Δ PR | 유의성 |
|---|---|---|---|---|---|
| Labeled only (base) | 0.9296 ± 0.0316 | 0.4479 ± 0.1113 | — | — | — |
| +A: EllipticEnv | 0.9378 | 0.4421 | +0.0081 | -0.0058 | trend (p=0.312) |
| +B: RF Confidence | 0.9271 | 0.4441 | -0.0025 | -0.0039 | ns (p=0.781) |
| +C: Direction | 0.9401 | 0.4423 | +0.0105 | -0.0056 | trend (p=0.312) |
| **+D: kNN Defect** | **0.9531** | **0.4572** | **+0.0235** | **+0.0093** | **IMPROVED* (p=0.031)** |

### 8.2 MLP 결과

| 전략 | ROC-AUC | PR-AUC | Δ ROC | Δ PR | 유의성 |
|---|---|---|---|---|---|
| Labeled only (base) | 0.9494 ± 0.0149 | 0.4349 ± 0.1191 | — | — | — |
| +A: EllipticEnv | 0.9510 | 0.4527 | +0.0016 | +0.0178 | trend (p=0.219) |
| +B: RF Confidence | 0.9485 | 0.4475 | -0.0008 | +0.0126 | trend (p=0.094) |
| +C: Direction | 0.9504 | 0.4552 | +0.0010 | +0.0203 | trend (p=0.062) |
| **+D: kNN Defect** | **0.9636** | **0.4660** | **+0.0142** | **+0.0311** | trend (p=0.062) |

### 8.3 결과 해석

**전략 D만 RF에서 Wilcoxon p=0.031 유의**

전략 D는 labeled 불량과 가장 가까운 unlabeled 샘플을 선택한다. align_score가 100%인 만큼, pseudo-defect의 피처 분포가 labeled 불량과 유사해 분류기 학습에 실질적으로 도움이 된다. ROC +0.0235, PR +0.0093 — fold수가 5개로 적음에도 유의성이 나온 것은 효과가 일관적이었음을 의미한다.

**전략 B의 허위 개선 확인**

두 번째 시도에서 전략 B가 IMPROVED(+0.032)를 기록했던 것은 teacher RF가 val fold를 포함한 7,996개 전체로 학습하면서 발생한 소프트 누수 때문이었다. 최종 구현에서 fold별 teacher로 누수를 차단하자 ROC -0.003 ns(p=0.781)가 됐다. 준지도 학습에서 teacher 모델의 누수가 얼마나 심각한 허위 양성을 만드는지 수치로 확인한 결과다.

**MLP에서 전략 D가 trend에 그친 이유**

MLP base 자체가 ROC=0.9494로 높아 개선 여지가 좁다. PR +0.0311은 실질적으로 의미있는 수준이지만, fold 5개로 p=0.062까지만 내려갔다. fold 수가 많아지면 유의에 도달할 가능성이 있다.

---

## 9. n_pseudo Ablation

`08_method_c_v2.ipynb` Section 9 결과. 전략 B, C, D에 대해 n=[50, 100, 200, 354, 500, 750, 1000, 2000]으로 ablation했다. 기준선: RF ROC=0.9296, PR=0.4479.

### 9.1 전략별 최적 n (PR-AUC 기준)

| 전략 | 최적 n | 최적 ROC | 최적 PR | vs baseline |
|---|---|---|---|---|
| B: RF Confidence | 2000 | 0.9433 | 0.4474 | ROC+0.0136, PR-0.0006 |
| C: Direction | 50 | 0.9446 | 0.4461 | ROC+0.0149, PR-0.0018 |
| **D: kNN Defect** | **50** | **0.9619** | **0.4826** | **ROC+0.0323, PR+0.0346** |

### 9.2 해석

**전략 D는 n=50에서 최고 성능**

n=50은 작은 수다. labeled 불량 71개에서 가장 가까운 unlabeled 샘플 50개만 선택한다는 뜻이다. 소수지만 품질이 높아 효과적이다. n이 커질수록 불량 영역에서 멀어진 샘플들이 포함되어 효과가 희석된다.

n=354(DR×unlabeled 수)에서도 ROC 개선이 있지만 n=50보다 낮다. 숫자보다 **품질(labeled 불량과의 유사성)**이 학습에 더 큰 영향을 준다.

**전략 B는 n=2000에서 최적**

많은 샘플을 추가해야 효과가 나타난다는 것은 개별 pseudo-defect의 품질이 낮음을 의미한다. fold별 teacher의 확신도 기준 선택이 불량을 정확히 포착하기보다 노이즈가 많다.

**전략 C는 n=50에서 최적**

n이 커지면 방향벡터와 유사한 샘플이 아닌 다른 샘플들이 섞인다. 소수 고품질 선택이 효과적.

---

## 10. Self-training Threshold Ablation

`08_method_c_v2.ipynb` Section 10. 가이드북 §2.2.2의 자기 지도 학습 방식을 검증했다.

### 10.1 self-training 결과

| threshold | avg pseudo_불량수 | ROC-AUC | PR-AUC |
|---|---|---|---|
| 0.50 | 50개 | 0.8156 ± 0.0474 | 0.4294 ± 0.1076 |
| 0.60 | 43개 | 0.8329 ± 0.0261 | 0.4267 ± 0.1124 |
| **0.70** | **7개** | **0.8709 ± 0.0505** | **0.4491 ± 0.1184** |
| 0.80 | 4개 | 0.8640 ± 0.0668 | 0.4474 ± 0.1224 |

### 10.2 해석

**모든 threshold에서 baseline 수준**

전략 D를 포함한 최고 전략과 비교하면 ROC 0.87 수준이 baseline(0.9296)보다 낮다. Self-training은 분류기가 자신의 예측으로 pseudo-label을 생성하는데, **불량률 0.89% 극단 불균형에서 분류기는 대부분의 unlabeled를 양품으로 예측**한다. threshold=0.70에서 겨우 7개의 불량 pseudo-label이 생성된다. 7개로는 의미 있는 학습 강화가 어렵다.

**최적 threshold=0.70**

PR=0.4491은 baseline(0.4479)보다 미세하게 높다. 하지만 개선폭이 std에 비해 작아 통계적으로 유의하지 않다.

**구조적 한계**

DR=0.89% 극단 불균형에서 분류기는 대부분의 unlabeled 샘플을 양품으로 예측한다. threshold=0.70에서 겨우 7개의 불량 pseudo-label이 생성된다는 것 자체가 이를 보여준다. 불균형이 심할수록 분류기가 불량에 높은 확신도를 부여하는 경우가 드물어 self-training의 전제 자체가 성립하지 않는다.

---

## 11. 종합 결론 및 방법론적 기여

### 11.1 방법 C의 최종 판정

**방법 C VALID** — 단, 전략 D(kNN from Labeled Defects)만 Wilcoxon p=0.031 유의.

| 항목 | 결과 |
|---|---|
| 유의한 개선 전략 | D (kNN Defect), RF: ROC+0.024, PR+0.009, p=0.031 |
| 최고 n_pseudo 성능 | D, n=50: ROC=0.9619, PR=0.4826 |
| 두 번째 시도 허위 개선 확인 | 전략 B +0.032는 두 번째 시도의 누수 (fold-teacher 차단 후 ns) |
| Self-training | 모든 threshold에서 baseline 수준 (구조적 한계) |

### 11.2 방법론적 기여

**기여 1 — 기계정지 오염 확인 및 제거**

CN7/RG3 필터 후에도 44%의 기계정지 레코드가 잔존함을 확인하고, 생산상태 필터(`Barrel_Temperature_1 >= 200 & Average_Screw_RPM > 0`)로 제거했다.

**기여 2 — 전략 B 허위 개선 발견**

Teacher RF가 val fold 포함 시 +0.032 IMPROVED처럼 보였지만, fold별 teacher로 차단하자 ns(-0.003)로 붕괴했다. 준지도 학습 평가에서 teacher 모델의 정보 누수가 false positive를 만드는 구체적 메커니즘을 실증했다.

**기여 3 — 표준화 공간의 필요성 실증**

Raw 공간에서 방향벡터는 Clamp_Open_Position 하나가 내적의 99%를 지배했다 (cosine 0.969). 표준화 후 cosine 0.150으로 내려가며 25개 피처가 균등하게 기여하게 됐다. 스케일 불일치 해소가 전략 설계에 필수적임을 수치로 보였다.

**기여 4 — pseudo-label 품질이 양보다 중요**

n ablation에서 전략 D는 n=50(소수 고품질)이 n=2000보다 PR 기준으로 우수했다. pseudo-defect 수를 늘리는 것보다 labeled 불량과의 피처 유사성이 준지도 학습 효과의 핵심 결정인자임을 실증했다.

### 11.3 방법 C의 한계

1. **유의성 달성 전략이 하나뿐** — 전략 D만 p=0.031. fold가 5개라 검정력이 낮다는 점을 감안해도, 방법론적 신뢰성은 제한적이다.
2. **PR-AUC 개선이 ROC 개선보다 작다** — 실제 불량 탐지(Recall)보다 순위 구분(ROC) 개선이 주효과다. 운용 임계값에서의 실질 성능 개선은 더 작을 수 있다.
3. **Self-training은 DR=0.89%에서 작동하지 않는다** — 구조적 한계이므로, 불량률이 더 높은 다른 데이터셋에서만 유효할 수 있다.

---

## 12. 노트북 흐름 안내

### 12.1 권장 읽기 순서

```
1. 08_pseudo_labeling.ipynb  ← 전체 흐름 파악 (첫 번째 시도 포함)
2. 08h_8h_eda.ipynb          ← Elliptic Envelope 심층 분석 (최고 이상탐지)
3. 08a_8a_eda.ipynb          ← COPOD 심층 분석 (완전비지도 최고)
4. 08_method_c_v2.ipynb      ← 최종 구현 (핵심)
5. 나머지 08b~08i            ← 필요 시 참조
```

### 12.2 `08_pseudo_labeling.ipynb` 읽기 포인트

| 섹션 | 주의할 내용 |
|---|---|
| 섹션 3 (PCA) | 불량이 양품 범위에 100% 혼재 — 선형 분리 불가 확인 |
| 섹션 5 (기법 비교) | 학습 데이터 차이(labeled vs unlabeled)가 ROC 차이의 원인 |
| **섹션 8 (pseudo-label 검증)** | **28% 방향 일치 = 실패. 기계정지 오염이 원인임을 확인** |
| **섹션 4 Step 4** | **QDA ROC -0.11 붕괴 = 오염된 pseudo-label의 결과** |
| 섹션 5 Step 5 | threshold를 올릴수록 성능 단조 감소 = 근본 원인은 pseudo-label 종류 |

### 12.3 `08_method_c_v2.ipynb` 읽기 포인트

| 섹션 | 주의할 내용 |
|---|---|
| 제목 표 | 두 번째 시도 → 최종 구현 수정 8개 항목 전체 파악 필수 |
| 섹션 2 (스케일러 분리) | sc_vis vs sc_sel 각각의 용도 이해 |
| 섹션 2 (방향벡터) | raw 상위 5 vs sc 상위 5 출력 비교 — Clamp_Open 지배 해소 확인 |
| **섹션 4 (전략 align_score)** | **전략 D가 100%/100%인 이유 이해 (선택 기준 자체가 거리)** |
| **섹션 7 (CV 결과)** | **전략 B fold-teacher 결과 = ns. 두 번째 시도 IMPROVED가 누수였음 확인** |
| 섹션 9 (n ablation) | D 최적 n=50 — 품질 > 양 |
| 섹션 10 (self-training) | threshold=0.70 최적이지만 baseline 수준 — 구조적 한계 |

### 12.4 EDA 노트북(08a~08i) 읽기 포인트

각 노트북은 동일한 구조로 되어 있다. 기법마다 다른 것은 섹션 4(하이퍼파라미터 탐색)와 섹션 10(방법 특화 분석)이다.

- **08h (Elliptic Envelope)**: Step 4 실패 원인 분석 섹션이 있다. pseudo-label로 생성된 불량이 왜 기계정지인지 피처별로 확인 가능.
- **08d (K-Means)**: Phase 7 ROC=0.47 재현 확인. 왜 k를 바꿔도 성능이 낮은지.
- **08g (OC-SVM)**: labeled 양품 학습 vs unlabeled 학습의 성능 차이 정량화.

### 12.5 각 노트북의 출력 파일

| 노트북 | 그림 파일 | 표 파일 |
|---|---|---|
| 08_pseudo_labeling | NB08_fig1~7.png | method_c_anomaly_comparison.csv, method_c_step4_results.csv, method_c_ablation.csv |
| 08_method_c_v2 | 각 섹션별 fig 저장 | — |
| 08a~08i | NB08a~i_fig_*.png | — |

모든 그림은 `results/figures/`, 표는 `results/tables/`에 저장된다.

---

수치는 모두 실제 실행 출력값이다. (SEED=42, fold_*.npy 고정)
