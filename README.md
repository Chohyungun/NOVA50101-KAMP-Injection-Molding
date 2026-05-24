# KAMP 사출성형 공정 불량 예측

**NOVA50101 인공지능학개론 텀프로젝트**  
팀원: 전정은(20268528) · 박상은(20268521) · 조현건(20268532)

KAMP 사출성형기 AI 데이터셋(7,996행 레이블 / 795,315행 무레이블)으로 불량 예측 모델을 단계별로 구축하고, 무레이블 데이터를 활용한 준지도 학습(방법 C)까지 확장한다.

---

## 결과 요약

### Phase 1-6 지도학습

| Phase | 모델 | ROC-AUC | PR-AUC |
|---|---|---|---|
| 2 (전처리 ablation) | LR-L2 + SMOTE | 0.9311 | 0.2413 |
| 3 (선형 베이스라인) | **QDA** (reg=0.01) | 0.9344 | 0.3526 |
| 4-A (차원축소) | TreeTop-15 + LR | 0.9380 | 0.2899 |
| **4-B (앙상블·NN)** | **MLP [256,128,64]** | **0.9497** | **0.4710** |
| 5 (CNN) | CNN1D (32->64) | 0.9472 | 0.4501 |
| 5 (Stacking) | meta-LR | 0.9462 | 0.4484 |
| 가이드북 DNN 재현 | DNN [128,64,32] | 0.9468 | 0.4655 |

운용점 (Precision >= 0.99): MLP Recall = 32.4% (71개 불량 중 23개 탐지)

### Phase 7 이상탐지 + 방법 C 준지도학습

| 방법 | 학습 데이터 | ROC-AUC | PR-AUC | 비고 |
|---|---|---|---|---|
| OC-SVM | labeled 양품 7,925 | 0.8814 +- 0.027 | 0.1765 | Phase 7 |
| K-Means (k=5) | unlabeled 795K | 0.4697 | 0.1444 | Phase 7, 도메인 불일치 |
| **방법 C - 전략 D (kNN)** | labeled 불량 seed | **0.9531** | **0.4572** | **Wilcoxon p=0.031** |

방법 C 핵심 발견: 전략 B v2의 ROC+0.032 "개선"이 teacher RF 데이터 누수였음을 확인. fold별 teacher로 차단하자 ns(-0.003)로 붕괴. n_pseudo ablation에서 전략 D n=50이 n=2000보다 PR 기준으로 우수 (품질 > 양).

---

## 데이터셋

| 항목 | 값 |
|---|---|
| 출처 | [KAMP 플랫폼](https://www.kamp-ai.kr) - 사출성형기 AI 데이터셋 |
| 전체 행 | 886,227 (CSV 8개) |
| 레이블 데이터 | `labeled_data.csv` 7,996행 (불량 71개, 0.89%) |
| 무레이블 데이터 | `unlabeled_data.csv` 795,315행 |
| 유효 피처 | 25개 (분산=0인 10개 + near-zero 1개 + ID 제거) |
| 방법 C 필터 후 | 71,180행 (CN7/RG3 + 동일 장비) -> 생산상태 39,870행 |

`data/raw/` CSV는 KAMP 저작권으로 미포함. [KAMP 플랫폼](https://www.kamp-ai.kr)에서 다운로드 후 `data/raw/`에 배치.

---

## 디렉토리 구조

```
IAAI_Term_project/
├── data/
│   ├── raw/               <- .gitignore (다운로드 필요)
│   └── splits/            <- 5-fold 인덱스 (fold_0~4.npy)
│
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing_ablation.ipynb
│   ├── 02_linear_baselines.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_ensemble_nn.ipynb
│   ├── 05_cnn_aux_stacking.ipynb
│   ├── 06_results_summary.ipynb
│   ├── 07_anomaly_detection.ipynb
│   ├── 08_pseudo_labeling.ipynb   <- 방법 C v1 (이상탐지 9종 비교 + 실패 분석)
│   ├── 08_method_c_v2.ipynb       <- 방법 C v3 최종 (4전략 + Wilcoxon)
│   └── 08a~08i_eda.ipynb          <- 이상탐지 기법별 상세 EDA
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   └── evaluate.py
│
├── results/
│   ├── figures/           <- 모든 그래프 (NB0x_fig*.png)
│   ├── tables/            <- ablation CSV
│   ├── ablation_summary.md
│   └── decisions.md
│
├── docs/
│   ├── method_c_report.md          <- 방법 C 분석 보고서 (팀 공유용)
│   ├── Proposal_NOVA50101.md
│   ├── Proposal_방향성_팀가이드.md
│   ├── TERM_PROJECT_GUIDELINE.md
│   ├── NOVA50101_Default_Final_Project_EN.md
│   ├── 04.-Guidebook_InjectionMolding_EN.md
│   └── 04.-Guidebook_사출성형기.pdf
│
├── _gen_all.py            <- 08a~08i 노트북 생성기
├── _gen_mc2.py            <- 08_method_c_v2 노트북 생성기
├── pyproject.toml
└── requirements.txt
```

---

## 실행

```bash
pip install -r requirements.txt
```

노트북은 `00` -> `08_method_c_v2` 순서로 실행한다. 모든 Phase가 `data/splits/fold_*.npy`를 공유한다.

방법 C 노트북만 재생성할 경우:
```bash
python _gen_mc2.py
jupyter nbconvert --to notebook --execute --inplace notebooks/08_method_c_v2.ipynb
```

---

## 가이드북 대비 차별점

| 항목 | 가이드북 | 본 프로젝트 |
|---|---|---|
| 불균형 처리 | 개념 소개만 | 4종 x 10조합 ablation, 5-fold |
| 평가지표 | accuracy + recall | ROC-AUC + PR-AUC (Accuracy 미사용) |
| 모델 선택 | AE -> SVM -> DNN 순차 | 선형 6종 / 앙상블 4종 / CNN / Stacking 체계적 비교 |
| SVM 주장 | SVM이 최적 | QDA가 PR-AUC에서 SVM 상회 |
| 무레이블 795K | AE(양품) + pseudo-label 언급 | 이상탐지 9종 비교 + 준지도학습 4전략 + 누수 검증 |
