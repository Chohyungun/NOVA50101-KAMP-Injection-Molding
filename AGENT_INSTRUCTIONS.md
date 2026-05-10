# 개발 에이전트 지침서 — KAMP 사출성형 텀프로젝트

> **읽는 사람:** 이 프로젝트의 코드를 작성·수정·실행하는 개발 에이전트(Claude Code 등).
> **목표:** 이 한 파일만 읽으면 무엇을 / 왜 / 어떻게 / 어디까지 해야 하는지 알 수 있게 한다.
> **작성:** 2026-05-10, 조현건

---

## 0. 1줄 미션

> **KAMP 사출성형 데이터(7,996행 × 45컬럼, PassOrFail 양성비 0.89%)에 강의 14개 알고리즘을 단계별 ablation으로 적용해, KAIST·UNIST 가이드북의 "AE/SVM/DNN 단일 best 선택" 흐름을 우리의 "어느 단계가 점수를 좌우했는지 분리해 보여주기"로 정면 비판하고 originality를 확보한다.**

이 한 문장이 모든 결정의 기준이다. 헷갈리면 여기로 돌아온다.

> **데이터 크기 정정(2026-05-10):** 가이드북이 언급하는 "886,227행"은 unlabeled_data.csv (raw 센서 스트림, AE 학습용)의 수치다. 라벨이 있는 supervised 데이터(`labeled_data.csv`)는 **7,996행 × 45컬럼**. 우리의 단계별 ablation은 supervised 위주이므로 이 7,996행이 주력 데이터다. unlabeled_data는 KAMP 회원가입 후만 받을 수 있어 이번 텀프에서는 사용하지 않는다.

---

## 1. 필수 컨텍스트 (반드시 먼저 읽기)

| # | 파일 | 무엇 |
|---|---|---|
| 1 | `docs/NOVA50101_Default_Final_Project.pdf` | 텀프 명세 (평가 70% 기술 / 25% 한계 / 5% 활동, 마감일) |
| 2 | `docs/04.-Guidebook_사출성형기.pdf` | KAMP 사출성형 공식 가이드북 (KAIST·UNIST·㈜이피엠솔루션즈). 변수 정의·통계·AE/SVM/DNN 코드 흐름 |
| 3 | `docs/TERM_PROJECT_GUIDELINE.md` | 6 Phase 일정·디렉토리·역할분담·자주 빠지는 함정 |
| 4 | `reports/데이터셋_최종_결정_보고서_v3.md` | 왜 KAMP 사출성형이 1순위인지(가중평균 4.45/5)·왜 NTIS·SECOM이 아닌지 |
| 5 | `docs/제안서_사출성형.docx` | 우리 팀 제안서 v0 (v3 보강 가이드는 v3 보고서 §5 참조) |
| 6 | `docs/lecture7_8 ~ lecture15_16` | 강의 슬라이드. 다룬 알고리즘만 사용한다 |

이 6개를 다 읽기 전에는 코드 한 줄도 쓰지 않는다.

---

## 2. 강의에서 다룬 알고리즘 풀 (이게 우리 도구상자다)

| 그룹 | 알고리즘 | 강의 회차 |
|---|---|---|
| 선형·생성적 | Logistic Regression (L1/L2), LDA, QDA | L4–6 |
| 거리·마진 | K-NN, SVM (linear / polynomial / RBF kernel) | L4, L7–8 |
| 트리·앙상블 | Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting | L7–10 |
| 메타 학습 | Stacking | L9–11 |
| 차원 축소 | PCA | L9–13 |
| 신경망 | MLP, Backprop, SGD/Momentum/Adam, Dropout, ReLU/Sigmoid/Tanh | L11–15 |
| 딥러닝 | CNN (Convolution), PyTorch 구현 (Model/Loss/Optimizer/DataLoader) | L13–16 |
| 평가 | ROC-AUC, PR-AUC, F1, 운용점(Precision-fixed Recall), Stratified k-fold CV | L5–6 |

**이 14가지 안에서만 작업한다.** 강의 밖 알고리즘(XGBoost는 GBM의 변형이라 OK / Transformer·GNN은 강의 밖이라 NG)을 쓸 때는 `results/decisions.md`에 한 줄 근거 남기고 PR 리뷰 후 진행.

---

## 3. 디렉토리 구조 (바꾸지 말 것)

```
E:\IAAI_Term_project\
├── AGENT_INSTRUCTIONS.md        ← 이 파일
├── docs/                         (읽기 전용 입력)
│   ├── NOVA50101_Default_Final_Project.pdf
│   ├── 04.-Guidebook_사출성형기.pdf
│   ├── TERM_PROJECT_GUIDELINE.md
│   ├── lecture7_8~15_16.pdf
│   ├── 제안서_사출성형.docx          (v0, v3는 docs에 별도 저장 예정)
│   └── On_predicting_research_grants_productivity_via_machine_learning.pdf
├── data/
│   ├── raw/                      (KAMP 6개 csv. 절대 수정 금지. §3.1 참조)
│   ├── processed/                (전처리 완료 파일. parquet 또는 npy)
│   └── splits/                   (5-fold split 인덱스. fold_0.npy ... fold_4.npy)
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing_ablation.ipynb
│   ├── 02_linear_baselines.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_ensemble_nn.ipynb
│   ├── 05_cnn_aux_stacking.ipynb
│   └── 06_results_summary.ipynb
├── src/                          (재사용 모듈. 노트북은 src를 import)
│   ├── __init__.py
│   ├── data.py                   (load_raw, load_processed, get_fold)
│   ├── preprocess.py             (impute, scale, resample)
│   ├── models/
│   │   ├── linear.py             (LR, LDA, QDA)
│   │   ├── svm.py
│   │   ├── tree.py               (DT, RF, AdaBoost, GBM)
│   │   ├── nn.py                 (MLP, CNN — PyTorch)
│   │   └── stacking.py
│   ├── evaluate.py               (roc_auc, pr_auc, f1, precision_at_recall)
│   └── utils.py                  (set_seed, timer, logger)
├── results/
│   ├── tables/                   (모든 ablation 결과 csv)
│   ├── figures/                  (PCA·ROC·PR·feature importance png)
│   ├── ablation_summary.md       (Phase별 결과 누적 — Phase 끝마다 갱신)
│   └── decisions.md              (강의 밖 라이브러리·기법 사용 결정 로그)
├── reports/
│   ├── NTIS_조사_및_데이터셋_결정_보고서.md  (v1)
│   ├── 데이터셋_최종_결정_보고서_v3.md          (확정안)
│   ├── NTIS_가치_및_비채택_사유_상세보고서.md
│   └── final_report.pdf          (Phase 6 산출물)
└── tests/                        (pytest 단위 테스트)
    ├── test_data.py
    ├── test_preprocess.py
    └── test_evaluate.py
```

**규칙:**
- `data/raw/`는 절대 수정 금지.
- `notebooks/`는 탐색·시각화용, 재사용 로직은 `src/`로 옮긴다.
- 새 디렉토리는 만들지 않는다 (필요하면 `decisions.md`에 근거 적고 사용자에게 물음).

### 3.1 실제 가용 데이터 인벤토리 (`data/raw/` 6개 파일)

출처: `johnwslee/injection_molding_analysis` GitHub repo (KAMP 공식 labeled_data.csv를 그대로 호스트). 모두 1차/2차 가공 단계의 supervised 학습용 데이터.

| 파일명 | 크기 | 행수 | 역할 |
|---|---|---|---|
| `labeled_data.csv` | 4.58 MB | 7,996 | **메인 데이터 — 1차 가공, 6개 제품 합본, 45컬럼(메타 9 + 변수 36)** |
| `labeled_data_CN7_interim.csv` | 1.92 MB | ~3,300 | CN7 제품 분리 (1차) |
| `labeled_data_RG3_interim.csv` | 0.61 MB | ~1,000 | RG3 제품 분리 (1차) |
| `labeled_data_preprocessed_interim.csv` | 2.73 MB | ~7,996 | 결측치 처리 후 통합 |
| `labeled_data_CN7_processed.csv` | 1.70 MB | ~3,300 | CN7 ML-ready (2차) |
| `labeled_data_RG3_processed.csv` | 0.53 MB | ~1,000 | RG3 ML-ready (2차) |

**라벨 분포(`labeled_data.csv`):** PassOrFail = `Y` 7,925 / `N` 71 → **양성(불량) 비율 0.89%**. 매우 강한 클래스 불균형 → SMOTE/ADASYN/class_weight 비교 ablation이 핵심.

**제품 분포:** `CN7 W/S SIDE MLD'G LH·RH`, `RG3 MOLD'G W/SHLD LH·RH`, `SP2 CVR ROOF RACK CTR RH`, `JX1 W/S SIDE MLD'G RH` (총 6종). 가이드북·johnwslee 모두 CN7과 RG3에 집중하므로 **SP2/JX1 표본은 EDA 단계에서 통계적 유의성 검토 후 분석 대상 결정**.

**라벨 매핑 주의:** PassOrFail은 'Y'/'N' 문자열 → 학습 시 `Y→1, N→0`으로 변환 필요(가이드북은 1=양품/0=불량 표기, 텀프 ablation도 동일 규약).

**가용하지 않은 KAMP 파일(가이드북 §2.3에서 언급되지만 미러 없음):**
- `unlabeled_data.csv` (~795K행, raw 센서 스트림, 가이드북 AE 학습용 — 회원가입 후 KAMP에서만 다운로드 가능)
- `moldset_labeled.csv`, `moldset_*_cn7.csv`, `moldset_*_rg3.csv` (준지도 학습 변형)
- `supervised_label_cn7.csv` (2차 가공 alternate)

→ 이번 텀프는 supervised 학습 위주이므로 위 파일 없어도 모든 단계 진행 가능. 가이드북 AE 재현은 `processed/labeled_data_*.csv` 위에서 unsupervised 부분 모방으로 대체.

---

## 4. 6 Phase 워크플로우

각 Phase는 **DoD(Definition of Done)**가 충족되어야 다음으로 넘어간다.

### Phase 1 — 데이터 + 환경 셋업

- `data/raw/` 6개 csv는 이미 배치되어 있음 (§3.1 참조). 에이전트는 무결성 검증만.
- `src/data.py` 작성: `load_raw(name)`, `load_processed(name)`, `get_fold(i, X, y)` (Stratified 5-fold), `binarize_label(y)` (Y→1, N→0)
- `src/utils.py`: `set_seed(42)` 가 numpy / random / torch 시드 모두 고정
- `tests/test_data.py`: 6개 csv 로딩·shape·`PassOrFail` Y/N 분포·라벨 매핑 검증
- DoD: `pytest tests/test_data.py` 통과 + `notebooks/00_EDA.ipynb`에서 6개 csv 모두 로드 확인

### Phase 2 — EDA + 전처리 Ablation

- `00_EDA.ipynb`: shape, dtype, 결측, 분산 0 변수, 상관행렬, 라벨 분포, 시계열 흐름
- `src/preprocess.py`: 결측 처리 (drop / median / KNN), 스케일러 (Standard / Robust), 리샘플러 (none / class_weight / SMOTE / ADASYN / undersample)
- `01_preprocessing_ablation.ipynb`: **분류기 = LR-L2 고정**, 위 전처리 옵션을 5-fold CV로 그리드. 결과 `results/tables/preproc_ablation.csv`
- 가이드북이 사전 표준화 → 결측 ablation 효과 약함 → **클래스 불균형 ablation에 시간 배분**
- DoD: `preproc_ablation.csv`에 모든 조합 결과(평균±표준편차) + 1줄 인사이트가 `ablation_summary.md`에 추가

### Phase 3 — 선형·생성적 베이스라인

- Phase 2의 best 전처리 위에서 LR(L1)·LR(L2)·LDA·QDA·SVM(linear)·SVM(RBF) 5-fold CV
- 하이퍼파라미터: 그리드는 `src/models/linear.py`·`svm.py` 안에 정의, 결과 `results/tables/baseline_results.csv`
- 데이터 7,996행이라 SVM RBF도 무난하게 돌아감(메모리 6.9MB). subsample 불필요.
- 양성 71개 → fold당 양성 ~14개 → **5-fold stratified CV가 필수**, fold당 양성 0개 안 되게 검증
- DoD: 6개 모델 결과표 + ROC/PR 곡선 png + "선형 결정경계로 어디까지 갔나" 인사이트 1단락

### Phase 4 — 차원축소 + 앙상블 + NN

- 차원축소 비교: 분산임계 / PCA(80·90·95·99% 분산보존) / 트리 중요도 Top-K — **분류기 = LR-L2 고정**
- 앙상블: Random Forest, AdaBoost, Gradient Boosting + **가이드북 DNN·SVM 재현**
- NN: MLP (구조 [128, 64], Dropout 0.3, ReLU, Adam) — PyTorch
- DoD: PCA 2D 시각화 + 차원축소 비교표 + 앙상블·NN 결과표가 `tables/`에 + 가이드북 모델 결과와 한 표에 비교

### Phase 5 — CNN 보조 + 스태킹 + 운용점

- 1D-CNN: 데이터가 사이클당 집계 변수라 자연스러운 시계열 입력은 어려움. 두 옵션 중 선택:
  - (a) 시간순 정렬 후 슬라이딩 윈도우(K=10) — 시간 의존성을 가정하는 경우
  - (b) 36 변수를 1D 시퀀스로 보고 Conv1D 적용 (DeepInsight 식, 강제적이지만 강의 CNN 시연 가능)
- 결정사항은 `decisions.md`에 기록.
- Stacking: Phase 3·4 best 5개 모델의 OOF 예측 → 메타 LR (data leakage 방지: OOF만 사용)
- 운용점: 양성비 0.89%이므로 **Precision=0.5/0.7/0.9 고정 시 Recall**이 의미. (Precision=0.99는 양성 절대수 부족으로 의미 약함)
- DoD: CNN 결과 + Stacking이 단일 best 모델을 넘는지 확인 + 운용점 표 + ROC/PR 곡선에 모든 모델 동시 표시

### Phase 6 — 보고서 + 발표

- `notebooks/06_results_summary.ipynb`: 모든 Phase 결과를 한 표·한 그림으로 통합
- `reports/final_report.pdf`: 8~12쪽 (구조는 `TERM_PROJECT_GUIDELINE.md` §6.1 참조)
- 발표 PPT: 14쪽 (구조는 §6.2 참조)
- 활동계획 부록 5%
- DoD: PDF·PPT 빌드되고 `pytest` 전부 통과

---

## 5. 행동 규칙 (절대 위반 금지)

### 5.1 데이터 누수 방지

- **SMOTE / 스케일링 / 결측 처리는 반드시 fold 안에서 fit.** 외부에서 fit하면 ROC-AUC 0.05 이상 부풀려진다.
- 5-fold split은 **모든 Phase가 같은 인덱스를 공유**한다. `data/splits/fold_0.npy ~ fold_4.npy` 한 번 만들고 재사용.
- `random_state=42` 고정. PyTorch는 `torch.manual_seed(42) + torch.cuda.manual_seed_all(42)`.

### 5.2 평가 지표

- **Accuracy 단독 보고 금지.** PassOrFail이 불균형이면 의미 없다.
- 주: ROC-AUC, PR-AUC. 보조: F1, Precision@Recall=0.9, Precision@Recall=0.99.
- 모든 결과는 5-fold CV 평균±표준편차로 보고.

### 5.3 결과 누적

- 각 Phase 끝나면 **`results/ablation_summary.md`에 1쪽 분량 인사이트 추가**.
- 표 한 줄, 그림 한 장, 인사이트 한 단락 형식.
- 누적되어야 Phase 6에서 보고서 쓸 때 그대로 쓴다.

### 5.4 가이드북과 항상 비교

- 모든 결과표는 **가이드북 모델 (AE / SVM / DNN) 결과를 같이 두 번째 열로 둔다**.
- 가이드북이 없는 단계(예: PCA ablation)는 "가이드북에 빠진 단계" 라고 명시.
- 이게 originality의 1차 차별점이다.

### 5.5 코드 품질

- `src/` 모듈은 type hint + docstring + 단위 테스트 1개씩.
- 노트북은 결과 보여주기 + 시각화. 로직은 `src/`로 옮긴다.
- `pytest` 전부 통과 상태로 유지. 실패 테스트 방치 금지.
- import 정렬·black 포맷.

### 5.6 강의 범위 외 사용 금지

- 강의 14개 알고리즘 안에서만 (§2 표 참조).
- XGBoost는 GBM 변형이라 OK. LightGBM도 OK.
- Transformer / BERT / GNN / LSTM 사용 시 `decisions.md`에 근거 + 사용자 승인 필요.

### 5.7 비상시 행동

- 막히면 **사용자에게 한 번에 1개 질문**, 그 질문에 multiple choice 옵션 2~4개 제시.
- 가이드북·강의 범위에서 답이 나오는 질문은 묻지 말고 직접 결정 + `decisions.md`에 기록.
- 88만 행 메모리 문제: 우선 stratified subsample → 사용자에게 보고.
- 마감 압박: TERM_PROJECT_GUIDELINE.md §6 즉시 실행 체크리스트 참조.

---

## 6. 첫 작업 (Phase 1)

**바로 시작할 일:**

1. `docs/04.-Guidebook_사출성형기.pdf` 읽기 (특히 §2.1 데이터 정의표, §2.3 코드 단계 9개)
2. `data/raw/` 의 6개 csv 무결성 검증 (§3.1 표와 일치, shape·라벨 분포)
3. `src/__init__.py`, `src/utils.py`, `src/data.py` 골격 작성
   - `set_seed(42)` 함수 (numpy/random/torch)
   - `RAW_DIR = Path("data/raw")`, `PROCESSED_DIR = Path("data/processed")`, `SPLITS_DIR = Path("data/splits")`
   - `load_raw(name)` — 6개 파일명을 Literal로 (§3.1 표 그대로)
   - `binarize_label(y)` — Y→1, N→0 (가이드북 규약과 일치)
4. `tests/test_data.py`: 6개 csv 모두 로드, `labeled_data.csv` 7,996행, PassOrFail Y=7925/N=71 (양성비 0.89%) 검증
5. `notebooks/00_EDA.ipynb` 시작: 가이드북 §2.1의 통계표를 우리 `labeled_data.csv`로 재계산해 일치 여부 검증

**Phase 1 DoD 충족 시 사용자에게 5줄 보고 후 Phase 2 진입.**

---

## 7. 보고 형식 (각 Phase 끝마다)

```markdown
## Phase N 완료

- 추가된 파일: src/foo.py, tests/test_foo.py, notebooks/0N_xxx.ipynb, results/tables/xxx.csv
- 핵심 결과 1줄: "결측 KNN > 중앙값 > drop 순으로 PR-AUC +0.012 개선"
- 다음 Phase 진입 가능 여부: ✅ / ⚠ (블로커 있음 → 사유)
- 결정 로그: `results/decisions.md`에 N개 항목 추가
- 사용자 확인 필요 사항: (있으면 1~3줄)
```

위 5줄 형식 그대로. 길게 풀어쓰지 않는다.

---

## 8. Anti-pattern (이러면 안 된다)

- ❌ 5-fold CV 안 돌리고 train/test split 한 번만 쓰기 → 결과 신뢰성 ↓ (특히 양성 71개 환경에서 치명적)
- ❌ Accuracy만 보고 → 양성비 0.89%면 모두 음성으로 찍어도 99.1% Acc. 의미 없음
- ❌ Stratified 안 쓴 fold split → fold당 양성 0개 발생 가능, 학습·평가 깨짐
- ❌ 가이드북 모델 재현 없이 우리 모델만 보고 → originality 비교 무의미
- ❌ 노트북에 모든 로직 작성 → 재현·디버깅·테스트 불가
- ❌ 시드 미고정 → 매번 결과 다르게 나옴, 보고서 작성 불가
- ❌ 단계별 ablation 없이 단일 best 모델만 고름 → 가이드북 따라하기, originality 0
- ❌ PassOrFail Y/N 문자열 그대로 모델 입력 → 학습 깨짐. 반드시 Y→1/N→0
- ❌ "이렇게 해도 될까요" 질문 5개 한 번에 → 한 번에 1개

---

## 9. 빠른 참조

- 데이터 크기: **`labeled_data.csv` 7,996행 × 45컬럼** (메타 9 + 변수 36). PassOrFail Y/N 문자열 → 학습 시 Y=1/N=0
- 라벨 분포: **Y 7,925 / N 71 (양성비 0.89%)** — SMOTE/ADASYN/class_weight 비교가 ablation 핵심
- 제품: CN7 LH·RH, RG3 LH·RH, SP2, JX1 (6종). 가이드북·선행연구는 CN7·RG3에 집중
- 시드: **42** (모든 곳: numpy, random, torch, torch.cuda)
- CV: **Stratified 5-fold**, fold split 한 번 만들고 재사용 (`data/splits/fold_*.npy`). fold당 양성 ~14개
- 주 평가지표: **ROC-AUC + PR-AUC + Precision@Recall=0.5/0.7/0.9**
- 마감: **2026-05-10 23:59 (오늘)** 제안서 / 이후 5주 (학기말 정확한 마감은 Blackboard 공지)
- 분석 환경: **Anaconda + Jupyter** (가이드북 명세) 또는 **Colab** (GPU 무료, NN/CNN용)
- 패키지: pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, imbalanced-learn, torch, jupyter, pytest

---

## 10. 한 줄 요약

> **읽기 6개 → src 골격 → 테스트 → EDA → 단계별 ablation × 5-fold CV × 가이드북 비교 × 결과 누적 보고. 시드 고정·데이터 누수 방지·강의 범위 안에서.**

질문은 한 번에 하나, 결정은 `decisions.md`에 기록. 시작.
