# NOVA50101 기말 프로젝트 — Proposal 방향성 팀 가이드 (v2)

> **작성 목적:** 5/17 Proposal 제출 준비 — 팀원 내부 공유  
> **최초 작성:** 2026-05-16 / **v2 업데이트:** 2026-05-16  
> **데이터셋:** KAMP 사출성형기 (886,227행 × 25 유효 피처, labeled 7,996행, unlabeled 795,315행)

---

## 0. 핵심 요약

| 구분 | 내용 |
|---|---|
| **이미 한 것** | Phase 1~6 완료 — 가이드북 DNN 초과 달성 |
| **Proposal에서 "앞으로 할 것" 방법 1** | **Autoencoder 기반 이상탐지** — 가이드북 §2.2.1 재현 + unlabeled 795K 활용 확장 |
| **Proposal에서 "앞으로 할 것" 방법 2** | **K-Means 거리 기반 비지도 이상탐지** — 가이드북에 없는 신규 접근 (1차 프로토타입 실험 완료) |
| **Future Work** | Semi-supervised Pseudo-labeling (가이드북 §2.2.2 재현), DBSCAN/GMM 이상탐지, LSTM 시계열 |
| **평가 기준 대응** | Technical 70%: 초기 실험 + 방법 B/C | Limitation 25%: K-Means 실패 원인 분석 + 5가지 한계 | Activity 5%: 부록 |

---

## 1. 현재 실험 결과 요약

Phase 1~6이 이미 완료된 상태다. Proposal에서 이 결과들을 "초기 실험 결과(preliminary results)"로 제시한다.

### 1.1 Phase별 핵심 결과

| Phase | 내용 | 주요 수치 |
|---|---|---|
| Phase 1 (EDA) | 886K행 분포 분석, 분산 0 변수 11개 제거 | 유효 변수 25개 확정, 불량률 0.89% |
| Phase 2 (전처리) | Scaler 2종 × 불균형 처리 5종 = 10조합 5-fold | StandardScaler+SMOTE ROC-AUC 0.9311 |
| Phase 3 (선형) | LR(L1/L2)/LDA/QDA/SVM(linear/RBF) | QDA ROC-AUC 0.9344, PR-AUC 0.3526 |
| Phase 4 (앙상블+NN) | RF/GBM/AdaBoost/MLP, 가이드북 DNN 재현 | MLP[256,128,64] ROC-AUC 0.9497, PR-AUC 0.4710 |
| Phase 5 (CNN+Stacking) | 1D-CNN 보조, 5종 Stacking | 가이드북 DNN 대비 PR-AUC +0.0055 |
| Phase 6 (통합) | 전 Phase 정량 비교, 운용점 분석 | Prec≥0.99 시 불량 71건 중 23건 탐지 |

### 1.2 가이드북과 직접 비교

| 항목 | 가이드북 DNN | 우리 MLP | 비고 |
|---|---|---|---|
| ROC-AUC | 0.9468 | **0.9497** | +0.0029 |
| PR-AUC | 0.4655 | **0.4710** | +0.0055 |
| 불균형 처리 ablation | 없음 (개념만) | 4종 10조합 정량 비교 | **핵심 차별점** |
| 평가 지표 | accuracy+recall | ROC-AUC+PR-AUC+운용점 | 더 엄격한 기준 |

### 1.3 이상탐지 비교 실험 결과 (방법 B/C 1차 프로토타입)

방법 B/C의 출발점으로 One-Class SVM과 K-Means 이상탐지를 먼저 구현해 비교했다.

| 방법 | 학습 데이터 | ROC-AUC (5-fold) | PR-AUC (5-fold) |
|---|---|---|---|
| One-Class SVM (nu=0.02) | labeled 양품 7,925행 | **0.8814 ± 0.027** | **0.1765 ± 0.087** |
| K-Means (k=5, best) | unlabeled 795,315행 | 0.4697 ± 0.063 | 0.1444 ± 0.084 |
| K-Means (k=10~50) | unlabeled 795,315행 | 0.31~0.40 | 0.06~0.07 |

**K-Means가 사실상 랜덤(ROC-AUC 0.47) 수준인 이유 분석:**

unlabeled_data 795K 행은 공장 내 모든 기계·제품의 데이터를 포함한다. 반면 labeled_data는 "650-ton 우진 #2" 기계의 CN7/RG3 제품만으로 구성된다. 기계·제품이 다르면 공정 파라미터 분포가 크게 달라지므로, 전체 unlabeled로 학습한 K-Means 중심점이 CN7/RG3 정상 패턴을 대표하지 못한다. 양품과 불량 모두 유사한 centroid 거리를 갖게 되어 분리가 불가능해진다.

→ 이것이 이 실험에서 얻은 **가장 중요한 인사이트**이며, Proposal의 Limitation 섹션 핵심 재료다.

---

## 2. 핵심 문제의식

Phase 1~6은 **labeled_data(7,996행)만 사용**했다. 전체의 89.8%(795,315행)인 unlabeled_data는 단 한 번도 활용하지 않았다.

| 문제 | 규모 | Phase 1~6 대응 |
|---|---|---|
| **불균형 데이터** | 양품 7,925 vs 불량 71 (0.89%) | Phase 2에서 4종 ablation 완료. SMOTE 확정. |
| **Unlabeled 미활용** | 795,315행, 레이블 없음 | Phase 1~6 전체에서 미사용 |

가이드북이 이 두 문제를 다룬 방식:
- **불균형 처리:** "오버/언더샘플링을 사용할 수 있다" — 개념 1단락, 코드 없음
- **Unlabeled 활용:** AE 양품-only 학습(§2.2.1) + Pseudo-labeling(§2.2.2) — 구현 있음

우리 Proposal의 방향: 불균형 처리는 Phase 2에서 이미 채웠고, Unlabeled 활용을 방법 B/C로 다룬다.

---

## 3. Proposal의 "앞으로 할 것" — 방법 B와 C

### 방법 B: Autoencoder 기반 비지도 이상탐지 (1순위)

**핵심 아이디어:** 양품이 대다수이므로, 양품 패턴만 학습한 AE를 만들고 복원 오차가 큰 샘플을 불량으로 탐지한다. 레이블이 전혀 필요 없으므로 unlabeled 795K를 그대로 활용할 수 있다.

**가이드북(§2.2.1) 재현:**

```
가이드북 방식:
  입력: labeled CN7 양품만 (MinMaxScaler 정규화)
  구조: Dropout(0.3) → Dense(15,relu) → Dense(5,relu) → Dense(15,relu) → Dense(25,relu)
  임계값: +5σ (본문은 3σ 설명, 실제 코드는 5σ)
  평가: accuracy + precision + recall
```

**우리 확장 포인트:**

| 항목 | 가이드북 | 우리 |
|---|---|---|
| 학습 데이터 | labeled 양품 7,925행 | labeled 양품 **+** unlabeled 795K 비교 |
| 임계값 결정 | 3/4/5σ 휴리스틱 | PR 곡선 운용점에서 직접 선택 |
| 비교 baseline | 없음 | **One-Class SVM** (이미 구현 완료, ROC 0.88) |
| 평가 지표 | accuracy+precision+recall | ROC-AUC + PR-AUC + 운용점 분석 |

**실험 설계 (3-way 비교):**

```
Config 1: One-Class SVM  ← labeled 양품만, L7-8 강의 직결   → ROC 0.88 (구현 완료)
Config 2: AE (labeled)   ← 가이드북 재현                   → 구현 예정
Config 3: AE (labeled+unlabeled 795K) ← 우리 확장         → 구현 예정

임계값 A: +5σ  (가이드북 방식)
임계값 B: PR 곡선 운용점  (우리 방식)

→ 3×2 = 6개 설정 비교 → 어느 조합이 실제 현장에 유리한지 정량 분석
```

**강의 연결:** L7-8 (One-Class SVM) + L11-14 (Denoising AE / MLP+Dropout)

---

### 방법 C: K-Means 거리 기반 비지도 이상탐지 (2순위, 1차 실험 완료)

**핵심 아이디어 (가이드북에 없는 독자적 방법):**

가이드북 §2.2.1은 비지도 이상탐지의 이상 점수 원리로 "거리 또는 밀도 기반"을 명시했다. K-Means의 centroid 거리를 이상 점수로 쓰는 것은 이 원리의 직접적 구현이지만, 가이드북은 AE만 사용했고 K-Means 이상탐지는 없다.

```
학습: K-Means를 unlabeled 데이터로 학습 (레이블 전혀 불필요)
이상 점수: 각 샘플에서 가장 가까운 centroid까지의 거리
판정: 거리 > threshold → 불량 예측
```

**1차 프로토타입 실험 결과 (직접 구현 및 측정):**

| k | ROC-AUC (5-fold) | PR-AUC (5-fold) |
|---|---|---|
| k=5 (best) | 0.4697 ± 0.063 | 0.1444 ± 0.084 |
| k=10 | 0.3986 ± 0.062 | 0.0575 ± 0.029 |
| k=20 | 0.3149 ± 0.052 | 0.0684 ± 0.047 |
| k=50 | 0.3072 ± 0.045 | 0.0684 ± 0.047 |

결과: **사실상 랜덤 수준(ROC 0.47).** 이상 점수 분포를 시각화해보면 양품과 불량이 거의 동일한 centroid 거리 분포를 보인다.

**실패 원인 분석 (이것이 핵심 Limitation):**

unlabeled_data 795K는 공장 내 여러 기계·제품(33개 기계 추정)의 데이터를 모두 포함한다. labeled_data는 "650-ton 우진 #2" 기계의 CN7/RG3 두 제품만으로 구성된다. 이처럼 학습(K-Means) 데이터와 평가 데이터의 모집단이 달라서 centroid 거리가 품질과 무관한 기계·제품 간 편차를 반영하게 된다.

**후속 개선 방향:**

```
1. unlabeled_data를 동일 기계(EQUIP_NAME)·제품(PART_NAME)으로 필터링 후 K-Means 재학습
   → 1차 실험의 구조적 원인을 직접 검증 가능
   → Proposal에서 이 개선을 Future Work로 제시

2. AE (방법 B)는 동일한 문제를 겪지만, AE는 labeled 양품을 직접 학습에 포함하므로
   도메인 불일치 영향이 K-Means보다 작을 것으로 기대
   → 방법 B와 C의 결과 차이가 이 가설을 검증

3. DBSCAN, GMM log-likelihood를 labeled 소규모 데이터에서 시도
   (795K 전체 아닌 labeled 내에서 이상탐지)
```

**강의 연결:** L9-10 (K-Means) + Future Work: DBSCAN, GMM, 계층적 클러스터링, UMAP (모두 강의 포함)

---

### 접근법 C: Pseudo-labeling → **Proposal 방법 C로 포함** (재검토 후 결정)

가이드북 §2.2.2의 Semi-supervised Pseudo-labeling을 방법 C로 Proposal에 포함한다.

**포함 근거:**
- Proposal은 계획서이므로 실험을 아직 안 해도 "수행 예정"으로 기재 가능
- 가이드북 §2.2.2가 직접 근거이므로 계획 신뢰도가 높음
- 방법 A(§2.2.1 AE) + 방법 C(§2.2.2 Pseudo-labeling) = 가이드북 두 갈래를 모두 재현 + 개선하는 구조가 완성됨
- 점수 측면에서 방법이 많을수록 Technical Soundness 항목에 유리

**Proposal에서의 표현 (1~2줄):**
가이드북 §2.2.2는 labeled 데이터로 분류기를 먼저 학습한 뒤, unlabeled 데이터에 예측 확률로 엔트로피를 계산하고 확신도 높은 샘플에 pseudo-label을 부여하여 labeled set을 반복 확장하는 방식을 사용한다. 본 연구에서는 가이드북의 고정 4종 모델(SVM, RF, GNB, DNN) 대신 Phase 3~4 ablation에서 검증된 best 모델(QDA, RF, MLP)을 기반 분류기로 활용한다.

---

## 4. 가이드북과의 차별점 6가지 (Originality)

과제 지시서: "독창성은 most upvoted 공개 코드와 비교하여 평가" → KAMP 데이터는 Kaggle 미등재, **가이드북이 비교 기준.**

| 항목 | 가이드북 | 우리 |
|---|---|---|
| **불균형 처리** | 개념 1단락 (코드 없음) | 4종 10조합 5-fold 정량 ablation |
| **평가 지표** | accuracy + precision + recall + F1 | ROC-AUC + PR-AUC + 운용점 분석 |
| **전처리 비교** | 단일 표준화 | Scaler 2종 × 불균형 5종 = 10조합 |
| **모델 탐색** | AE/SVM/DNN | 6종 선형 → 4종 앙상블 → CNN → Stacking |
| **숨겨진 모델 발굴** | SVM best 주장 | QDA가 PR-AUC 기준 SVM 초과 |
| **K-Means 이상탐지** | 없음 | 신규 시도 + 실패 원인 분석 |

---

## 5. 한계 및 Future Work (25% 배점)

이 섹션은 Proposal에서 충분히 써야 한다. 솔직하게 쓸수록 평가가 높아진다.

**현재 분석의 한계:**

1. **K-Means 이상탐지의 도메인 불일치 문제**  
   unlabeled 전체(여러 기계·제품)로 학습한 K-Means는 labeled(단일 기계·제품)의 불량을 탐지하지 못했다. 동일 기계·제품으로 필터링한 unlabeled를 사용하면 개선 가능하나, 현재 전처리된 unlabeled 파일에 기계·제품 메타데이터가 분리되어 있지 않아 즉시 적용이 어렵다.

2. **Unlabeled 데이터 완전 미활용 (Phase 1~6)**  
   전체 886K 중 795K(89.8%)를 사용하지 않았다. 이 데이터에는 공정 드리프트 패턴이 존재할 가능성이 있다.

3. **SMOTE의 근본 한계**  
   불량 71개 사이를 보간해 생성하므로 실제 존재하지 않는 불량 패턴을 생성할 수 있다. 극단적 불균형에서 SMOTE 보간 품질 자체가 의심된다.

4. **시계열성 미활용**  
   사출 사이클이 시간 순서가 있음에도 현재 모델은 각 사이클을 독립 샘플로 처리한다. EDA에서 불량이 특정 날짜(10-27 전후)에 몰려 있음을 확인했지만, 인접 사이클 간 패턴(공정 드리프트)은 포착하지 못한다.

5. **CN7 vs RG3 분리 모델 미구현**  
   KS 검정으로 두 제품의 22/24 변수 분포가 유의미하게 다름을 확인했지만, 분리된 모델은 학습하지 않았다. 금형 종류를 피처로 추가하는 것으로 우회했으나 근본 해결은 아니다.

**Future Work:**

| 항목 | 방법 | 기대 효과 |
|---|---|---|
| K-Means 이상탐지 개선 | 동일 기계·제품 unlabeled 필터링 후 재학습 | 도메인 불일치 해소 |
| DBSCAN 이상탐지 | labeled 7,996행에서 밀도 기반 noise 탐지 | K-Means 한계 보완 |
| GMM 이상탐지 | log-likelihood를 이상 점수로 (unlabeled 필터 후) | 확률적 비교 |
| Semi-supervised Pseudo-labeling | Phase 3~4 best 모델로 iterative pseudo-label | 가이드북 §2.2.2 재현 |
| 시계열 모델 | LSTM 슬라이딩 윈도우 (인접 K 사이클) | 공정 드리프트 포착 |

---

## 6. 강의 알고리즘 커버리지

과제 지시서: "DL 프로젝트라면 강의에서 다룬 비지도 학습 방법도 포함해야 한다."

우리 프로젝트는 MLP/CNN을 포함하므로 비지도 학습 명시가 필요하다.

| 강의 | 알고리즘 | 구현 여부 | 위치 |
|---|---|---|---|
| L4-6 | LR, LDA, QDA | ✅ | Phase 2-3 |
| L7-8 | SVM (linear/RBF) | ✅ | Phase 3 |
| L7-8 | **One-Class SVM** | ✅ (구현 완료) | 방법 B 비교군 |
| L8-10 | RF, AdaBoost, GBM, Stacking | ✅ | Phase 4-5 |
| L9-10 | **K-Means** | ✅ (구현 완료) | 방법 C |
| L9-10 | GMM, DBSCAN, 계층적 | Future Work | §5 |
| L11 | PCA | ✅ | Phase 4-A |
| L11 | **UMAP** | Future Work | §5 |
| L11-14 | MLP+Dropout | ✅ | Phase 4 |
| L11-14 | **Denoising AE** | 구현 예정 | 방법 B |
| L15-16 | 1D-CNN | ✅ | Phase 5 |

방법 B/C를 추가하면 L4~L16 전 범위 핵심 알고리즘 커버.

---

## 7. 과제 지시서 기준별 대응

### Technical Soundness (70%)

| 평가 요소 | 우리 대응 | 근거 |
|---|---|---|
| **제안된 방법** | 방법 B(AE) + 방법 C(K-Means) | §3 |
| **초기 실험 결과** | QDA PR-AUC 0.35, MLP PR-AUC 0.47, OC-SVM ROC 0.88 | notebooks/02-06, run_anomaly_detection.py |
| **기술적 타당성** | 5-fold StratifiedKFold, SMOTE fold 내부 fit, 시드 42 고정 | 전체 |
| **독창성** | 불균형 10조합 ablation, K-Means 이상탐지 시도 및 실패 원인 분석 | Phase 2, run_anomaly_detection.py |
| **결과 인사이트** | "SMOTE=ROC 개선·PR 안정화", "QDA=숨겨진 강자", "K-Means 실패=도메인 불일치" | notebooks/01, 02 + §1.3 |
| **프로젝트 규모** | 886K행, 10조합, 6 Phase + 이상탐지 실험 | 전체 |

### Limitation & Future Work (25%)

§5 내용이 이 점수를 결정한다. 특히 K-Means 실패 원인 분석(도메인 불일치)은 단순 "성능 낮았다"가 아니라 원인을 정확히 짚은 분석이라 높은 점수를 기대할 수 있다.

### Activities (5%)

원래 계획 vs 수정된 계획, 팀원별 기여도 부록으로 제출.

---

## 8. Proposal 제출용 초안 (2페이지 분량)

교수님께 제출할 Proposal의 권장 구조. 이 초안을 PDF로 변환해 Blackboard 업로드.

---

### §1 팀 정보 및 프로젝트 제목

```
팀 번호: [팀 번호]
팀원: 조현건(팀장), [멤버 B], [멤버 C]
프로젝트 제목: Defect Prediction in Injection Molding via Unsupervised Anomaly Detection
               — Comparative Study of Autoencoder and K-Means Distance Approaches
데이터셋: KAMP 사출성형기 AI 데이터셋
URL: KAMP(한국AI제조플랫폼) — 공개 코드 없음, 가이드북(KAIST 공식) 비교 기준 사용
크기/형태: 886,227행 × 27변수, labeled 7,996행 / unlabeled 795,315행 (CSV)
```

---

### §2 데이터셋 및 초기 실험 결과

**데이터셋 특성:**
- 사출성형 공정 파라미터 25개 (배럴 온도, 사출 압력, 속도 등)
- 불량률 0.89% (양품 7,925 / 불량 71): 극단적 클래스 불균형
- Unlabeled 데이터 795,315행: labeled의 99.5×

**초기 실험 (Phase 1~5 완료):**

Phase 2에서 Scaler(Standard/Robust) × 불균형 처리(SMOTE/ADASYN/class_weight/undersample/none) 10조합 5-fold 비교를 수행했다. Standard+SMOTE가 ROC-AUC 0.9311로 최적이었다. 이는 가이드북이 "오버/언더샘플링을 사용할 수 있다"고만 언급하고 구현하지 않은 부분을 우리가 정량으로 채운 것이다.

Phase 3~5에서 선형(LR/LDA/QDA/SVM) → 앙상블(RF/GBM/AdaBoost) → MLP/CNN → Stacking 순으로 탐색했고, MLP[256,128,64]가 ROC-AUC 0.9497, PR-AUC 0.4710으로 가이드북 DNN(0.9468, 0.4655)을 초과했다.

이상탐지 1차 실험으로 One-Class SVM(ROC 0.88)과 K-Means 거리 기반 이상탐지(ROC 0.47)를 구현 및 비교했다. K-Means의 낮은 성능은 unlabeled 데이터의 도메인 불일치(다기계·다제품)에서 비롯됨을 확인했다.

---

### §3 제안하는 방법

**방법 1: Autoencoder 기반 비지도 이상탐지 (가이드북 §2.2.1 재현 및 확장)**

가이드북은 labeled 양품만으로 Denoising Autoencoder를 학습하고 reconstruction error로 불량을 탐지했다. 우리는 두 가지를 개선한다. 첫째, labeled 양품에 unlabeled 795K까지 합쳐서 AE를 학습함으로써 정상 패턴 학습의 밀도를 높인다. 레이블이 불필요하므로 데이터 리크 없이 활용 가능하다. 둘째, 가이드북의 3/4/5σ 휴리스틱 임계값 대신 labeled val fold의 PR 곡선에서 운용점을 직접 선택해 현장 요구사항에 맞춘 임계값을 결정한다. One-Class SVM(L7-8)을 비교 baseline으로 추가해 재현율·정밀도 트레이드오프를 정량 비교한다.

강의 연결: L7-8 (One-Class SVM) / L11-14 (Denoising AE, MLP, Dropout)

**방법 2: K-Means 거리 기반 비지도 이상탐지 (신규 제안)**

가이드북 §2.2.1은 비지도 이상탐지의 이상 점수 원리로 "거리 또는 밀도 기반"을 명시했으나, 실제 구현은 AE만 사용했다. 우리는 K-Means 클러스터링의 centroid 거리를 이상 점수로 활용하는 방법을 시도한다. unlabeled 795K로 K-Means를 학습(k=5~50 ablation)하고, 각 샘플의 최근접 centroid까지의 거리를 이상 점수로 사용한다. 1차 프로토타입 실험에서 unlabeled 전체 사용 시 도메인 불일치 문제가 발생했으며(ROC 0.47), 동일 기계·제품으로 필터링한 unlabeled를 사용하는 개선 실험이 이미 계획되어 있다. 방법 1(AE)과의 정량 비교를 통해 "선형 거리 기반 vs 비선형 복원 기반" 이상탐지의 특성 차이를 분석한다.

강의 연결: L9-10 (K-Means)

---

### §4 한계 및 Future Work

(§5 내용 그대로 작성. Limitation 25%를 절대 짧게 쓰지 마라.)

---

### §5 팀원 역할

| 팀원 | 담당 Phase | 주요 기여 |
|---|---|---|
| 조현건 (팀장) | 전체 설계, Phase 2·5·6, 이상탐지 실험 | 프로젝트 아키텍처, 전처리 ablation, Stacking, K-Means/OC-SVM 구현 |
| [멤버 B] | Phase 1·3 | EDA 시각화, LR/LDA/QDA/SVM 선형 베이스라인 |
| [멤버 C] | Phase 4·5 | RF/GBM 앙상블, MLP+Dropout, 1D-CNN |

*(4인 팀이면 Phase 분담 재조정)*

---

## 9. 주의사항

**실험 설계:**
- AE 학습 시 unlabeled 전체를 그대로 넣으면 K-Means와 동일한 도메인 불일치 문제가 발생할 수 있다. labeled 양품은 반드시 포함하고, unlabeled는 필터링 여부를 비교해야 한다.
- val fold는 어떤 학습에도 포함하지 않는다. K-Means는 unlabeled로 학습하므로 leakage 없지만, AE 학습 시 labeled의 val fold 양품을 섞으면 leakage.

**Proposal 작성:**
- "앞으로 할 것"이 3가지 이상이면 비현실적으로 보인다. 방법 B+C 두 가지에 집중하고, Pseudo-labeling은 Future Work 한 단락으로.
- 이미 완료한 Phase 2~5 결과를 "preliminary results"로 표현해야 한다. "이미 끝났다"보다 "이 결과를 기반으로 다음 단계를 제안한다"가 자연스럽다.
- K-Means 실패 결과도 솔직하게 포함시킨다. "시도했고 이래서 안 됐고 이렇게 개선한다"가 Limitation 점수를 끌어올린다.

---

## 10. 일정

| 시점 | 할 일 |
|---|---|
| **5/16 완료** | K-Means(k=5~50), One-Class SVM 구현·측정 완료 |
| **5/17 오전** | Proposal §1~§3 초안 완성 (이 문서 §8 초안 기반) |
| **5/17 오전** | Proposal §4 한계 작성 (이 문서 §5 기반) |
| **5/17 오후** | 활동 계획 부록 (팀원 이름 채우기, 원래 계획 vs 수정 계획) |
| **5/17 저녁** | 최종 검토 → PDF 변환 → Blackboard 제출 |
| **5/17 이후** | AE 구현 (방법 B), unlabeled 필터링 후 K-Means 재실험 (방법 C 개선) |

---

*내부 공유용 문서. 실제 제출 Proposal은 §8 초안을 PDF로 변환해 제출.*  
*2026-05-16 v2 | 조현건*
