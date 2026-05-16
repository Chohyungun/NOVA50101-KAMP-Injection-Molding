# 개발 에이전트 지침서 — KAMP 사출성형 텀프로젝트 (v2, 2026-05-16)

> **읽는 사람:** 이 프로젝트를 이어받아 작업하는 에이전트(Claude Code 등).  
> **목표:** 이 파일 하나만 읽으면 현재 상태·남은 작업·규칙을 모두 파악할 수 있게 한다.

---

## 0. 현재 상태 (2026-05-16 기준)

| 항목 | 상태 |
|---|---|
| **Phase 1~6** | ✅ 완료 (notebooks/00~06 전부 실행 완료, 출력 있음) |
| **가이드북 DNN 재현** | ✅ ROC 0.9468/PR 0.4655 재현, MLP[256,128,64] ROC 0.9497/PR 0.4710 초과 |
| **Proposal** | ⏳ 5/17 마감 — `docs/Proposal_방향성_팀가이드.md` 초안 완료 |
| **방법 B (AE 이상탐지)** | ⏳ One-Class SVM ROC 0.8814 구현 완료, Denoising AE 구현 예정 |
| **방법 C (K-Means 이상탐지)** | ⚠️ 1차 구현 완료 (ROC 0.47, 도메인 불일치 원인 확인), 개선 실험 예정 |

**1줄 미션:**  
> KAMP 사출성형 데이터에 강의 알고리즘을 단계별 ablation으로 적용해 가이드북의 "단일 best 선택" 방식을 정량적으로 비판하고, 추가로 Unlabeled 795K를 활용한 비지도 이상탐지로 가이드북에 없는 originality를 확보한다.

---

## 1. 필수 컨텍스트 파일

| # | 파일 | 용도 |
|---|---|---|
| 1 | `docs/NOVA50101_Default_Final_Project_EN.md` | 텀프 명세 (평가 70/25/5%, Proposal 2페이지 요건) |
| 2 | `docs/04.-Guidebook_InjectionMolding_EN.md` | 가이드북 전문 번역 (§2.2~2.3 알고리즘, 데이터 정의) |
| 3 | `docs/Proposal_방향성_팀가이드.md` | Proposal 방향성 + 방법 B/C 설계 + 실험 결과 (핵심 문서) |
| 4 | `docs/TERM_PROJECT_GUIDELINE.md` | 디렉토리 규칙, 코드 품질 기준 |
| 5 | `results/ablation_summary.md` | Phase 1~6 누적 인사이트 |
| 6 | `results/decisions.md` | 기술 결정 로그 (알고리즘 선택 근거 등) |

---

## 2. 강의 알고리즘 풀 (확정 목록)

| 그룹 | 알고리즘 | 강의 | 구현 상태 |
|---|---|---|---|
| 선형·생성적 | LR(L1/L2), LDA, QDA | L4-6 | ✅ Phase 2-3 |
| 거리·마진 | SVM(linear/RBF), **One-Class SVM** | L7-8 | ✅ Phase 3 / 이상탐지 B |
| 트리·앙상블 | DT, Bagging, RF, AdaBoost, GBM | L7-10 | ✅ Phase 4 |
| 메타학습 | Stacking | L9-10 | ✅ Phase 5 |
| 비지도·클러스터링 | **K-Means**, **GMM**, **DBSCAN**, 계층적 | L9-10 | ⚠️ K-Means 1차 완료 |
| 차원축소 | PCA, **UMAP** | L11 | ✅ Phase 4-A |
| 신경망 | MLP, Dropout, ReLU/Adam | L11-14 | ✅ Phase 4 |
| 딥러닝 · AE | CNN(Conv1D), **Denoising Autoencoder** | L11-16 | ✅ CNN / ⏳ AE |
| 평가 | ROC-AUC, PR-AUC, 운용점, Stratified k-fold | L5-6 | ✅ 전체 |

> **이전 버전에서 AE를 "강의 밖"으로 잘못 분류했으나, L11-14 강의 포함임이 확인됐다.** AE 구현은 방법 B의 핵심이다.

---

## 3. 디렉토리 구조 (현재 상태)

```
E:\IAAI_Term_project\
├── AGENT_INSTRUCTIONS.md           ← 이 파일
├── fix_and_run_notebooks.py        ← 노트북 일괄 재실행 유틸 (Phase 1~6 완료 후 생성)
├── run_dim_reduction.py            ← Phase 4-A 차원축소 실험 스크립트
├── run_baseline_figures.py         ← Phase 3 figure 생성
├── run_phase4_figures.py           ← Phase 4 figure 생성
├── run_phase5.py                   ← Phase 5 실험
├── run_eda.py / run_eda_enhanced.py ← Phase 1 EDA figure 생성
├── run_anomaly_detection.py        ← 비지도 이상탐지 (OC-SVM, K-Means) ← NEW
├── docs/
│   ├── NOVA50101_Default_Final_Project_EN.md   ← 과제 명세 (영문 번역)
│   ├── 04.-Guidebook_InjectionMolding_EN.md    ← 가이드북 전문 번역 (핵심)
│   ├── Proposal_방향성_팀가이드.md              ← Proposal 설계 문서 (핵심)
│   ├── TERM_PROJECT_GUIDELINE.md
│   └── lecture7_8 ~ lecture15_16.pdf
├── data/
│   ├── raw/         (8개 CSV, 절대 수정 금지)
│   ├── processed/
│   └── splits/      (fold_0~4.npy)
├── notebooks/       (00~06, 모두 실행 완료)
├── src/             (data.py, preprocess.py, evaluate.py, utils.py)
├── results/
│   ├── tables/      (ablation CSV + anomaly_detection_results.csv)
│   ├── figures/     (NB{nn}_fig{n}_ 명명 + anomaly_*.png)
│   ├── ablation_summary.md
│   └── decisions.md
└── reports/
    ├── project_report_full.md      ← Phase 1~6 + 이상탐지 통합 보고서
    ├── presentation_outline.md     ← 발표 PPT 아웃라인
    ├── notion_draft.md             ← Notion 업로드용 초안
    └── 데이터셋_최종_결정_보고서_v3.md ← 데이터셋 선택 근거 (참고용)
```

### 3.1 데이터 인벤토리 (`data/raw/` 8개 파일)

| 파일명 | 행수 | 역할 |
|---|---|---|
| `labeled_data.csv` | 7,996 | 메인 supervised 학습 데이터 (PassOrFail 있음) |
| `unlabeled_data.csv` | 795,315 | **이상탐지 비지도 학습용** — 현재 접근 가능 |
| `moldset_labeled.csv` | 2,607 | Semi-supervised 학습용 labeled |
| `supervised_label_cn7.csv` | 6,736 | CN7 제품 전용 supervised |
| `moldset_labeled_cn7.csv` | 1,211 | CN7 semi-supervised labeled |
| `moldset_unlabeled_cn7.csv` | 35,239 | CN7 semi-supervised unlabeled |
| `moldset_labeled_rg3.csv` | 1,182 | RG3 semi-supervised labeled |
| `moldset_unlabeled_rg3.csv` | 35,941 | RG3 semi-supervised unlabeled |

> **v1에서 "unlabeled_data.csv 미접근"이라고 적혔지만 현재는 접근 가능하다.** 방법 B(AE)/C(K-Means) 학습에 적극 활용한다.

### 3.2 Figure 명명 규칙

- **노트북 생성 figure:** `NB{nn}_fig{n}_{설명}.png` (예: `NB00_fig1_eda_class_distribution.png`)
- **스크립트 생성 figure:** `{주제}_{설명}.png` (예: `anomaly_kmeans_k_ablation.png`)
- **스크립트별 prefix:** `dim_reduction_*`, `anomaly_*` 등

---

## 4. 실험 단계별 상태 및 남은 작업

### ✅ 완료: Phase 1~6

| Phase | 핵심 결과 | 관련 파일 |
|---|---|---|
| Phase 1 (EDA) | 유효 피처 25개 확정, 불량률 0.89%, CN7≠RG3 | notebooks/00_EDA.ipynb |
| Phase 2 (전처리) | StandardScaler+SMOTE 확정, ROC 0.9311 | notebooks/01 |
| Phase 3 (선형) | QDA ROC 0.9344/PR 0.3526 — SVM best 반박 | notebooks/02 |
| Phase 4-A (차원축소) | TreeTop-15 최적, PCA는 성능 저하 | notebooks/03 |
| Phase 4-B (앙상블) | MLP[256,128,64] ROC 0.9497/PR 0.4710 — 가이드북 초과 | notebooks/04 |
| Phase 5 (CNN/Stacking) | CNN1D(32,64) ROC 0.9472, Stacking MLP 미초과 | notebooks/05 |
| Phase 6 (통합) | 전 Phase 비교, 운용점 Prec≥0.99→Recall 32.4% | notebooks/06 |

### ⏳ 진행 중: 비지도 이상탐지 (Phase 7 격)

**방법 B: Autoencoder 기반 이상탐지**
- 가이드북 §2.2.1 재현 + unlabeled 795K 포함 버전 비교
- 임계값: 3/4/5σ(가이드북) vs PR 곡선 운용점(우리)
- 비교 baseline: One-Class SVM (ROC 0.8814 — 완료)
- **다음 할 일:** Denoising AE 구현 (`run_autoencoder.py` 또는 `notebooks/07_anomaly_detection.ipynb`)

**방법 C: K-Means 거리 기반 이상탐지**
- 1차 실험 완료: unlabeled 전체 학습 시 ROC 0.47 (도메인 불일치)
- **실패 원인:** unlabeled 795K는 다기계·다제품, labeled는 CN7/RG3 단일 기계 → centroid가 도메인 불일치
- **다음 할 일:** unlabeled를 동일 기계·제품(EQUIP_NAME, PART_NAME)으로 필터링 후 재학습
  - 단, 현재 processed unlabeled 파일에 메타컬럼이 없을 수 있어 원본 unlabeled_data.csv 확인 필요

### 📋 Proposal 마감 (5/17)

- `docs/Proposal_방향성_팀가이드.md` §8의 초안을 PDF로 변환해 Blackboard 제출
- 팀원 이름, 역할 표 실명 채우기

---

## 5. 행동 규칙

### 5.1 데이터 누수 방지

- SMOTE·스케일링은 반드시 fold 안에서 fit
- K-Means를 unlabeled로 학습할 때는 leakage 없음 (레이블 불필요)
- AE 학습에 labeled val fold 양품 포함하지 않음
- 5-fold split은 `data/splits/fold_*.npy` 재사용

### 5.2 평가 지표

- **Accuracy 단독 보고 금지.** 주 지표: ROC-AUC, PR-AUC. 보조: F1, 운용점(Prec-fixed Recall)
- 이상탐지 평가: labeled val fold 기준

### 5.3 결과 누적

- 새 실험 끝나면 `results/ablation_summary.md`에 결과 한 단락 추가
- 기술 결정은 `results/decisions.md`에 날짜·근거 기록

### 5.4 Figure 저장

- 모든 figure는 `results/figures/`에 저장
- 스크립트 실행 전 `sns.set_theme()` 후 반드시 `setup_korean_font()` 호출 (순서 바꾸면 한글 깨짐)

### 5.5 코드 품질

- `src/` 모듈: type hint + docstring + 단위 테스트
- 새 실험은 `run_{주제}.py` 스크립트로 분리. 노트북은 탐색용

### 5.6 강의 범위

- §2 표에 있는 알고리즘 전부 사용 가능 (AE 포함 — 강의 확인 완료)
- LSTM/Transformer/GNN은 `results/decisions.md`에 근거 + 사용자 승인 필요

---

## 6. 현재 시점에서 에이전트가 해야 할 일

우선순위 순:

1. **Denoising AE 구현** (`run_autoencoder.py`):
   - labeled 양품만 / labeled 양품+unlabeled 두 가지 버전
   - reconstruction error를 이상 점수로 사용
   - 임계값: 3/5σ vs PR 곡선 운용점 비교
   - `anomaly_ae_*.png` figure 3종 생성

2. **unlabeled 필터링 후 K-Means 재실험**:
   - `unlabeled_data.csv`에 EQUIP_NAME/PART_NAME 메타컬럼 존재 여부 확인
   - 있으면 650-ton 우진 #2 기계 + CN7/RG3 필터링 후 K-Means 재학습
   - 1차(ROC 0.47)와 2차 비교

3. **Proposal 최종 검토 및 제출** (5/17):
   - `docs/Proposal_방향성_팀가이드.md` §8 초안으로 PDF 작성
   - 팀원 이름·역할 채우기

---

## 7. 빠른 참조

| 항목 | 값 |
|---|---|
| 주 데이터 | `labeled_data.csv`, 7,996행, 25 유효 피처 |
| 불량 비율 | 0.89% (양품 7,925 / 불량 71) |
| 시드 | 42 (전체 고정) |
| CV | Stratified 5-fold, `data/splits/fold_*.npy` |
| 주 평가 지표 | ROC-AUC, PR-AUC |
| Proposal 마감 | 2026-05-17 |
| 최고 성능 (Phase 1~6) | MLP ROC 0.9497, PR 0.4710 |
| 이상탐지 현황 | OC-SVM ROC 0.8814 / K-Means ROC 0.47(도메인 불일치) / AE 구현 예정 |

---

*v1 작성: 2026-05-10 / v2 업데이트: 2026-05-16 | Phase 1~6 완료, unlabeled 접근 가능, AE 강의 포함 확인*
