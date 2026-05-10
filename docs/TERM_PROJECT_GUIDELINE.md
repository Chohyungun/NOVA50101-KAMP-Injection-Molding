# NOVA50101 텀프로젝트 진행 가이드라인 — KAMP 사출성형 (1순위 확정안)

> **작성자:** 조현건  
> **작성일:** 2026-05-10  
> **데이터셋:** KAMP 사출성형기 AI 데이터셋 (886,227행 × 27변수, 자동차 앞유리 사이드 몰딩)  
> **가이드북 출처:** `04.-Guidebook_사출성형기.pdf` (KAIST·UNIST·㈜이피엠솔루션즈, 2020)  
> **목표:** 가이드북의 "AE/SVM/DNN 차례로 학습 → best 선택" 흐름을 우리의 **단계별 ablation**으로 정면 비판하고 originality를 확보. 강의 14개 알고리즘(LR/LDA/QDA/SVM/Tree/Bagging/AdaBoost/GBM/Stacking/PCA/MLP/Dropout/CNN/PyTorch)을 모두 자연스럽게 시연.

---

## 0. 전체 일정 한눈에 (오늘 → 학기말)

| Phase | 기간 | 산출물 | 시간 추정 |
|---|---|---|---|
| **0. 제안서 마감** | **오늘 23:59** | `제안서_사출성형_v3.docx` (2페이지) | 2~3시간 |
| 1. 데이터 확보 + 환경 셋업 | 5/11–5/12 | KAMP 데이터셋 7개 csv, Colab/conda 환경, GitHub repo | 0.5일 |
| 2. EDA + 전처리 비교 ablation | 5/13–5/19 | EDA 노트북, 결측·표준화·불균형 처리 비교표 | 1주 |
| 3. 선형·생성적 베이스라인 | 5/20–5/26 | LR/LDA/QDA/SVM 5-fold CV 결과표 | 1주 |
| 4. 차원축소 + 앙상블 + NN | 5/27–6/2 | PCA·트리중요도, RF/GBM, MLP+Dropout 결과 | 1주 |
| 5. CNN 보조 + 스태킹 | 6/3–6/9 | 1D-CNN 보조 단계, Stacking 메타 학습기 | 1주 |
| 6. 보고서 + 발표 자료 | 6/10–학기말 | 최종 보고서 PDF + PPT | 1주 |

(학기말 정확한 마감은 NOVA50101_Default_Final_Project.pdf에 명시되어 있지 않음. 강의 진행 일정상 6월 둘째~셋째 주로 추정. Blackboard 공지 확인 필수.)

---

## Phase 0 — 오늘 23:59 마감 (제안서 v3)

### 0.1 제안서 작성 체크리스트

기존 `제안서_사출성형.docx`를 베이스로:

- [ ] **§0 헤더:** 팀번호(5/4 Blackboard 공지본 확인), 팀원 3~4명, 제목 = "KAMP 사출성형 데이터로 본 공정 불량 예측 — AE/SVM/DNN 가이드북 vs 단계별 ablation"
- [ ] **§1 배경**: 자동차 앞유리 사이드 몰딩 사출 공정. 26개 공정 변수(온도/압력/시간/속도/위치). 라벨 PassOrFail. 약 88만 건.
- [ ] **§2 데이터**: 출처 KAMP·KAIST·UNIST·㈜이피엠솔루션즈. 크기 886,227행 × 27변수 (35.2MB). PassOrFail 양품/불량 이진. **Kaggle 미등재 → 강의 구두 공지로 Kaggle 의무 해제 적용.**
- [ ] **§3 베이스라인 (가이드북 비교 대상)**: 
  - 가이드북의 "AE/SVM/DNN 차례 학습 → best 선택" 흐름 명시
  - **차별점: 우리는 "단계별 ablation으로 어느 처리가 점수를 좌우했는지 분리해 보여준다"** (이게 originality의 1차 차별점)
  - Top 3 Kaggle 노트북 항목은 "If no public code is available" 조항 인용으로 면제 — 가이드북을 베이스라인 1로 사용
- [ ] **§4 접근 방법** (4단계 Ablation):
  1. EDA + 결측/표준화/불균형 처리 비교 (가중치 vs SMOTE vs ADASYN)
  2. 선형·생성적 베이스라인: LR(L1/L2), LDA, QDA, SVM(linear/RBF) 5-fold CV
  3. 차원축소 비교: 분산임계 / PCA(80/90/95/99% 보존) / 트리 중요도
  4. 앙상블·NN·CNN·스태킹: RF, GBM, **가이드북 DNN 재현**, **+ 1D-CNN 보조**, Stacking 메타 학습기
  - 평가: ROC-AUC, PR-AUC, F1, 운용점 분석 (Precision-fixed Recall)
- [ ] **§5 일정·역할**: 위 §0의 6 phase 일정 요약 + 팀원별 1~3단계 분담
- [ ] **2페이지 제한**: 다듬기, 활동계획 부록은 제안서 본문 외 별도 첨부

### 0.2 제출 직전 최종 점검

- [ ] PDF 변환 후 글자 깨짐 없는지 확인
- [ ] 팀번호 정확한지 Blackboard 재확인
- [ ] 팀장이 Blackboard 업로드 (조현건이 팀장이면 본인이)
- [ ] (선택) TA 이메일 1줄: "KAMP 데이터셋 사용. 강의 Kaggle 의무 해제 공지 적용. 가이드북 베이스라인."

---

## Phase 1 — 데이터 확보 + 환경 셋업 (5/11–5/12)

### 1.1 KAMP 데이터셋 다운로드

- KAMP 플랫폼(https://www.kamp-ai.kr) 접속 → 회원가입 → "사출성형기 AI 데이터셋" 검색
- 다운로드 받을 파일 (가이드북 §2.1에 명시):
  - **1차 가공 데이터:**
    - `labeled_data.csv` (지도 학습용, PassOrFail 포함)
    - `moldset_labeled.csv` (준지도 학습용, label 포함)
    - `unlabeled_data.csv` (준지도 학습용, label 없음, 일별 불량률만)
  - **2차 가공 데이터 (제품별 분리):**
    - `supervised_label_cn7.csv` (지도 학습, CN7 제품)
    - `moldset_labeled_cn7.csv`, `moldset_unlabeled_cn7.csv` (준지도 학습, CN7)
    - `moldset_labeled_rg3.csv`, `moldset_unlabeled_rg3.csv` (준지도 학습, RG3)
- 저장 경로: `E:\IAAI_Term_project\data\raw\` (새 폴더)

### 1.2 변수 명세 (27개 → 26개 독립 + 1개 종속)

| 그룹 | 변수 | 단위 |
|---|---|---|
| 시간 | Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time, Clamp_Close_Time | 초 |
| 위치 | Cushion_Position, Switch_Over_Position, Plasticizing_Position, Clamp_Open_Position | mm |
| 속도 | Max_Injection_Speed, Max_Screw_RPM, Average_Screw_RPM | mm/s |
| 압력 | Max_Injection_Pressure, Max_Switch_Over_Pressure, Max_Back_Pressure, Average_Back_Pressure | MPa |
| 온도 | Barrel_Temperature_1~7 (7개), Hopper_Temperature, Mold_Temperature_1~12 (12개) | °C |
| **종속변수** | **PassOrFail** | 1=양품 / 0=불량 |

> **EDA 시 주의:** 가이드북 통계표를 보면 `Mold_Temperature_1, 2, 5~12`는 **모든 통계량이 0** (수집되지 않음). `Barrel_Temperature_7`도 통계량이 0. → **분산이 0인 변수는 즉시 제거**, 의미 있는 변수는 약 12~14개로 축소될 가능성 큼.

### 1.3 환경 셋업 (Colab 권장)

- **로컬 환경:** Anaconda + Jupyter Notebook (가이드북 명세) — Python 3.10+
- **Colab 권장:** GPU 무료 사용 (NN/CNN 단계용). 88만 행이면 메모리 13GB 충분.
- **필요 패키지:**
  ```
  pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
  torch torchvision  # for CNN, MLP
  jupyter
  ```
- **GitHub repo:** `team_<번호>_kamp_molding` 으로 생성 → 팀원 모두 collaborator 추가

### 1.4 디렉토리 구조 권장

```
E:\IAAI_Term_project\
├── docs/                                    (이미 존재)
│   ├── NOVA50101_Default_Final_Project.pdf
│   ├── 04.-Guidebook_사출성형기.pdf       (방금 추가)
│   ├── lecture7_8~15_16.pdf
│   ├── 제안서_사출성형.docx → v3로 보강
│   └── TERM_PROJECT_GUIDELINE.md           (이 파일)
├── data/
│   ├── raw/                                 (KAMP 원본 csv 7개)
│   ├── processed/                           (전처리 완료 npy/parquet)
│   └── splits/                              (CV fold 인덱스)
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing_ablation.ipynb     (Phase 2)
│   ├── 02_linear_baselines.ipynb           (Phase 3)
│   ├── 03_dimensionality_reduction.ipynb   (Phase 4)
│   ├── 04_ensemble_nn.ipynb                (Phase 4)
│   ├── 05_cnn_aux_stacking.ipynb           (Phase 5)
│   └── 06_results_summary.ipynb            (Phase 6)
├── src/
│   ├── data.py                              (데이터 로딩, fold 분리)
│   ├── preprocess.py                        (결측·표준화·SMOTE 등)
│   ├── models/                              (LR/SVM/RF/MLP/CNN 클래스)
│   ├── evaluate.py                          (ROC-AUC/PR-AUC/운용점)
│   └── stacking.py                          (메타 학습기)
├── results/
│   ├── tables/                              (단계별 ablation 결과 csv)
│   ├── figures/                             (PCA, ROC, PR 곡선 png)
│   └── ablation_summary.md                  (단계별 결과 요약)
└── reports/                                 (이미 존재)
    ├── 데이터셋_최종_결정_보고서_v3.md
    └── final_report.pdf                     (Phase 6 산출물)
```

---

## Phase 2 — EDA + 전처리 비교 Ablation (5/13–5/19)

### 2.1 EDA 핵심 항목 (`00_EDA.ipynb`)

1. **데이터 로딩 + 기본 정보:** shape, dtype, 메모리, 결측 비율
2. **변수별 분포:** 히스토그램 (특히 0이 많은 변수 식별 → 분산 0인 변수 제거)
3. **PassOrFail 클래스 분포:** 양/불량 비율 확인 (가이드북에 명시는 없으나 일반적으로 매우 불균형)
4. **변수 간 상관행렬:** 히트맵 → 강한 상관(>0.95) 변수 짝 식별
5. **시간 의존성:** TimeStamp 또는 PART_FACT_PLAN_DATE 기준 시계열 플롯 — 공정 안정성 변동 확인
6. **제품별(CN7/RG3) 분포 차이:** 두 제품 합쳐 분석할지, 분리할지 결정

**산출물:** `results/figures/eda_*.png`, `results/tables/feature_summary.csv`

### 2.2 전처리 Ablation (`01_preprocessing_ablation.ipynb`)

같은 분류기(예: 로지스틱 회귀 L2)를 고정하고 **전처리만 바꿔가며** 5-fold stratified CV로 비교:

| 결측 처리 | 표준화 | 클래스 불균형 처리 | 차원 축소 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| (가이드북: 사전 표준화 적용됨) | StandardScaler | None (baseline) | None | TBD | TBD |
| 결측 변수 제거 (분산 0) | StandardScaler | Class weight | None | TBD | TBD |
| 결측 변수 제거 | StandardScaler | SMOTE | None | TBD | TBD |
| 결측 변수 제거 | StandardScaler | ADASYN | None | TBD | TBD |
| 결측 변수 제거 | RobustScaler | Class weight | None | TBD | TBD |

→ **Insight 정리:** "어느 처리가 PR-AUC를 가장 끌어올렸는가?" 한 줄로 요약 가능해야 함.

> **사출성형 데이터의 약점 우회 전략:** 가이드북이 사전 표준화를 적용했으므로 결측 처리 ablation 효과는 약하다. 대신 **클래스 불균형 처리 ablation에 시간 배분**한다 (Class weight vs SMOTE vs ADASYN vs Random Undersample 4종 비교).

---

## Phase 3 — 선형·생성적 베이스라인 (5/20–5/26)

### 3.1 모델 5종 학습 (`02_linear_baselines.ipynb`)

Phase 2에서 결정한 best 전처리 위에서, 다음 5개 분류기를 5-fold stratified CV:

| 모델 | 하이퍼파라미터 | 강의 출처 |
|---|---|---|
| Logistic Regression (L1) | C ∈ {0.01, 0.1, 1, 10} | L4-5 |
| Logistic Regression (L2) | C ∈ {0.01, 0.1, 1, 10} | L4-5 |
| LDA | shrinkage='auto' | L5-6 |
| QDA | reg_param ∈ {0, 0.01, 0.1} | L5-6 |
| SVM (linear) | C ∈ {0.1, 1, 10}, class_weight='balanced' | L7-8 |
| SVM (RBF) | C, gamma ∈ {0.01, 0.1, 1} | L7-8 |

**산출물:** `results/tables/baseline_results.csv` (모델별 ROC-AUC, PR-AUC, F1, Precision@0.99 평균±std)

### 3.2 1차 인사이트 작성

- "선형 결정경계로 어디까지 갈 수 있는가?"
- "SVM RBF가 LR L2 대비 의미 있는 차이를 만드는가?"
- "QDA의 정규성 가정이 데이터에 맞는가?"

이 질문에 답하는 1~2 페이지 분량의 ablation 인사이트를 `results/ablation_summary.md`에 누적 기록.

---

## Phase 4 — 차원축소 + 앙상블 + NN (5/27–6/2)

### 4.1 차원축소 비교 (`03_dimensionality_reduction.ipynb`)

분류기를 LR-L2로 고정하고 차원축소 방법만 바꾸며 5-fold CV:

| 차원축소 방법 | 차원 수 | ROC-AUC | PR-AUC | 시각화 |
|---|---|---|---|---|
| 없음 (full features) | ~14 | TBD | TBD | — |
| 분산 임계값 필터 | k | TBD | TBD | — |
| PCA (분산 보존 80%) | k | TBD | TBD | scatter plot 2D |
| PCA (분산 보존 90%) | k | TBD | TBD | scatter plot 2D |
| PCA (분산 보존 95%) | k | TBD | TBD | scatter plot 2D |
| PCA (분산 보존 99%) | k | TBD | TBD | scatter plot 2D |
| 트리 중요도 Top-K | k ∈ {5, 10, 15} | TBD | TBD | feature importance |

→ **PCA 시각화는 발표 자료의 핵심 그림이 됨.** 2D 또는 3D scatter로 양품/불량 분포 명확히 시각화.

### 4.2 앙상블 + NN (`04_ensemble_nn.ipynb`)

| 모델 | 강의 출처 | 주요 하이퍼 |
|---|---|---|
| Random Forest | L8 (Bagging) | n_estimators ∈ {100, 300, 500}, max_depth |
| Gradient Boosting | L9-10 | n_estimators, learning_rate, max_depth |
| AdaBoost | L7-8 | n_estimators, learning_rate |
| MLP (Dropout) | L11-15 | hidden ∈ {[64], [128, 64], [256, 128, 64]}, Dropout 0.3 |
| **가이드북 DNN 재현** | L11-15 | 가이드북 §2.3 그대로 (참고용) |
| **가이드북 SVM 재현** | (이미 §3.1) | (Phase 3 결과 인용) |

→ **가이드북 모델 재현 + 우리의 ablation 결과를 한 표에 두고 비교** = originality 핵심 그림.

---

## Phase 5 — CNN 보조 + 스태킹 (6/3–6/9)

### 5.1 1D-CNN 보조 단계 (`05_cnn_aux_stacking.ipynb`)

**목적:** "강의 후반 토픽(CNN)도 따라잡았다"는 시그널 + 사출성형의 시계열 색채 활용.

**구현 전략 (3가지 옵션, 1개 이상 시도):**

1. **사이클 슬라이딩 윈도우 1D-CNN (권장):** 인접한 K개 사이클(예: K=10)을 묶어 26 변수 × K 길이 시계열로 만들어 1D-CNN 입력. PyTorch + Conv1D(kernel=3) → MaxPool → Conv1D → Flatten → FC.
2. **DeepInsight식 변수 → 2D 이미지 변환:** 26 변수의 상관행렬 위에서 t-SNE로 2D 좌표 잡고 변수값을 픽셀 강도로 매핑 → 2D-CNN.
3. **변수 자체를 1D 시퀀스로 처리 (강제):** 26 변수를 임의 순서 1D 시퀀스로 보고 Conv1D 적용 (가장 간단하지만 자연성 낮음).

**산출물:** Conv1D 모델의 ROC-AUC, PR-AUC를 §3·§4 결과표에 추가.

### 5.2 Stacking 메타 학습기

- 1단계 (Base): Phase 3·4의 최고 성능 모델 5개 (예: LR-L2, SVM-RBF, RF, GBM, MLP)의 OOF 예측 확률
- 2단계 (Meta): Logistic Regression on Base predictions
- 5-fold CV에서 OOF 예측을 메타 학습기 입력으로 사용 (data leakage 방지)

→ **단일 best 모델 vs 스태킹 결과 비교**. 스태킹이 best 단일을 넘어서면 originality 강화.

### 5.3 운용점(Operating Point) 분석

- **Precision = 0.95, 0.99 고정시 Recall** 측정
- **PR 곡선 위에 운용점 마커** 표시 → 산업 현장 응용 시 유의미한 그림

---

## Phase 6 — 보고서 + 발표 자료 (6/10–학기말)

### 6.1 최종 보고서 구조 (PDF 8~12페이지 권장)

1. **Abstract & Introduction** — 가이드북 단일 best vs 우리 ablation 차별점 1줄 요약
2. **Dataset & Preprocessing** — KAMP 사출성형, 변수 27→14 축소 근거, EDA 핵심 그림 2~3개
3. **Methodology** — 4-Phase ablation 설계 도표 (Phase 2~5)
4. **Results & Insights** — 단계별 ablation 결과표 + ROC/PR 곡선 + PCA 시각화 + 운용점 그림
5. **Discussion** — "어느 단계가 점수를 좌우했는가?" 한 페이지 인사이트
6. **Limitation & Future Work (25%)** — 시계열성 충분히 활용 못함, 멀티태스크 (CN7+RG3 분리), 사출 사이클 내 더 정밀한 시계열 모델링 (LSTM/Transformer), 도메인 지식 결합 등
7. **Conclusion**
8. **Appendix** — 활동계획 (5%): 원안 + 수정안 + 멤버별 기여 표

### 6.2 발표 PPT 구조 (10~15분 추정)

| 슬라이드 | 내용 |
|---|---|
| 1 | 표지 + 팀번호·구성·제목 |
| 2 | 문제 정의: 사출성형 불량 예측 + 가이드북 비판 |
| 3 | 데이터셋 개요 (886K × 27, PassOrFail) |
| 4 | 우리의 차별점: 단계별 ablation |
| 5 | EDA 핵심 (분산 0 변수 제거, 클래스 분포) |
| 6 | Phase 2: 전처리 ablation 결과표 |
| 7 | Phase 3: 선형·생성적 베이스라인 결과 |
| 8 | Phase 4: 차원축소 비교 + PCA 시각화 |
| 9 | Phase 4: 앙상블·NN 결과 (가이드북 DNN 비교) |
| 10 | Phase 5: 1D-CNN + Stacking 결과 |
| 11 | 운용점 분석 (Precision-fixed Recall) |
| 12 | 단계별 인사이트 한 페이지 요약 |
| 13 | Limitation & Future Work |
| 14 | Q&A 대비 백업 슬라이드 |

---

## 산출물 마스터 체크리스트

### 필수 (제출용)
- [ ] `제안서_사출성형_v3.docx` (오늘 23:59 마감)
- [ ] `final_report.pdf` (학기말 마감)
- [ ] `final_presentation.pptx` (발표일)
- [ ] `Activity_Appendix.pdf` (활동계획 + 멤버 기여, 5%)

### 권장 (재현성)
- [ ] GitHub repo (코드 + README + 노트북)
- [ ] `results/ablation_summary.md` (단계별 결과 누적 기록)
- [ ] `requirements.txt` (패키지 버전)
- [ ] 5-fold CV split index 파일 (재현성)

---

## 역할 분담 템플릿 (3인 가정, 4인이면 2명이 단계 분담)

| 멤버 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 |
|---|---|---|---|---|---|
| **팀장 (조현건)** | 전처리 ablation 설계, 기록 | LR/LDA/QDA 실험 | 차원축소 비교 | 스태킹 메타 학습기 | 보고서 통합, 발표 |
| 멤버 B | EDA 시각화 | SVM 실험 | RF/GBM | 1D-CNN 구현 | 결과 그림·표 정리 |
| 멤버 C | 데이터 로딩 파이프라인 | 평가 지표 모듈 | MLP+Dropout, 가이드북 DNN 재현 | 운용점 분석 | 활동계획·기여도 정리 |

---

## 자주 빠지기 쉬운 함정 (예방용)

1. **데이터 누수:** SMOTE/스케일링은 반드시 fold **내부**에서 fit. fold 밖에서 fit하면 ROC-AUC 0.05+ 부풀려짐.
2. **클래스 불균형:** Accuracy 보고하면 안 됨. ROC-AUC + PR-AUC + Precision-fixed Recall이 표준.
3. **CV 일관성:** 모든 단계가 **같은 fold split**을 써야 비교 가능. `random_state` 고정 + fold index를 파일로 저장.
4. **88만 행 메모리:** SVM RBF는 90% 이상 시간 걸릴 수 있음 → SVM은 stratified subsample (예: 10만 행) 또는 LinearSVC 우회. 가이드북도 SVM 사용 시 일부 데이터로 학습.
5. **DL 시드 고정:** PyTorch `torch.manual_seed`, NumPy seed, Python `random.seed` 모두 고정해야 재현성 확보.
6. **가이드북 단순 재현 금지:** 텀프 명세 §1 "should not be a simple replication". 가이드북 DNN은 베이스라인으로만, 우리 ablation은 그 위에 얹어야 함.

---

## 빠른 시작 (오늘 23:59 후, 5/11 부터)

```bash
# 1. 환경
conda create -n kamp python=3.11 -y
conda activate kamp
pip install pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn torch jupyter

# 2. 데이터 (KAMP 다운로드 후)
mkdir -p E:\IAAI_Term_project\data\raw
# csv 7개를 raw/ 아래로 이동

# 3. EDA 시작
jupyter notebook E:\IAAI_Term_project\notebooks\00_EDA.ipynb
```

---

## 참고자료

- **이 가이드라인의 1순위 결정 근거:** `E:\IAAI_Term_project\reports\데이터셋_최종_결정_보고서_v3.md`
- **데이터셋 가이드북:** `E:\IAAI_Term_project\docs\04.-Guidebook_사출성형기.pdf`
- **텀프 명세:** `E:\IAAI_Term_project\docs\NOVA50101_Default_Final_Project.pdf`
- **강의 슬라이드:** `lecture7_8 ~ lecture15_16` (강의 14개 알고리즘 카탈로그)
- **KAMP 플랫폼:** https://www.kamp-ai.kr (회원가입 필요)
- **UNIST 산업지능 연구실:** https://faculty.unist.ac.kr/sunghoonlim (가이드북 원본 호스팅)

---

*본 가이드라인은 v3 데이터셋 결정 보고서를 토대로 작성된 실행 계획서다. Phase 진행 중 결정사항이 변경되면 본 문서를 갱신할 것.*
