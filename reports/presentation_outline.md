# 발표 PPT 슬라이드 아웃라인 (16슬라이드)

> **업데이트:** 2026-05-16 (v2) — 비지도 이상탐지 슬라이드 추가  
> **발표 예상 시간:** 12~15분

---

## 슬라이드 구성 개요

| # | 제목 | 핵심 메시지 | Figure |
|---|---|---|---|
| 1 | 표지 | 팀 정보 + 제목 | — |
| 2 | 문제 정의 | 불량 예측 + 가이드북 한계 | — |
| 3 | 데이터셋 | 불량 0.89%, 25 변수, unlabeled 795K | 클래스 분포 |
| 4 | 우리 차별점 | 단계별 Ablation + 비지도 이상탐지 추가 | 설계 도표 |
| 5 | EDA | 분산 0 제거, CN7≠RG3, 드리프트 | 상관 히트맵 |
| 6 | Phase 2 — 전처리 | SMOTE: ROC +0.024, std 절반 | Ablation 히트맵 |
| 7 | Phase 3 — 선형 | QDA가 SVM "best" 뒤집음 | PR 곡선 |
| 8 | Phase 4-A — 차원축소 | PCA 손해, TreeTop-15 소폭 우위 | 차원축소 비교 |
| 9 | Phase 4-B — 앙상블·NN | MLP가 가이드북 DNN 초과 | ROC/PR 앙상블 |
| 10 | Phase 5 — CNN + Stacking | Stacking이 MLP 못 넘은 반례 | Phase5 종합 |
| 11 | 운용점 분석 | Prec≥0.99 → MLP Recall 32.4% | PR 운용점 |
| 12 | 단계별 기여도 요약 | Phase 4 비선형 전환이 최대 기여 | 기여도 표 |
| 13 | 비지도 이상탐지 (신규) | OC-SVM ROC 0.88 / K-Means ROC 0.47 + 실패 원인 | ROC/분포 |
| 14 | Limitation & Future Work | 6가지 한계 + K-Means 개선 방향 | — |
| 15 | 결론 | 단계별 ablation의 가치 | — |
| 16 | Q&A 백업 | 상세 결과표 + 이상탐지 설계 | — |

---

## 슬라이드별 상세 내용

### Slide 1 — 표지

"KAMP 사출성형 불량 예측: 가이드북 단일 Best vs 단계별 Ablation + 비지도 이상탐지"
- 팀번호 / 팀원 이름 / NOVA50101 / 발표일

---

### Slide 2 — 문제 정의

"가이드북은 AE→SVM→DNN 순서로 하나를 골랐지만, 어느 단계가 성능을 좌우했는지, unlabeled 795K를 어떻게 활용할지는 다루지 않았다."

- 사출성형 불량 예측의 산업 중요성
- 가이드북 흐름: AE → SVM → DNN → best 1개
- 가이드북 한계: 전처리 ablation 없음, 불균형 처리 개념만, unlabeled 활용 미흡
- 우리 접근: 6-Phase Ablation + 비지도 이상탐지 비교

---

### Slide 3 — 데이터셋

"불량 71건/7,996건(0.89%)의 극단적 불균형 + unlabeled 795K 활용 가능"

- 유효 변수 25개 (분산=0 11개 제거)
- 양품/불량 분포 수치
- unlabeled 795K: 비지도 이상탐지 학습에 활용 예정
- 평가: ROC-AUC, PR-AUC, 운용점

Figure: `NB00_fig1_eda_class_distribution.png`

---

### Slide 4 — 우리 차별점

"Phase별 변수 1개씩만 바꾸고 나머지 고정 → 단계 기여도 분리. 추가로 가이드북에 없는 비지도 이상탐지 비교."

- 4-Phase Ablation 구조 (LR 고정 → 분류기 고정 → 앙상블 비교)
- 비지도 이상탐지 추가 (방법 B: AE / 방법 C: K-Means)
- 가이드북과 달리 PR-AUC + 운용점 분석

Figure: 4-Phase 설계 도표 (PPT 직접 작성)

---

### Slide 5 — EDA 핵심

"분산=0 변수 11개 제거, 상관 34쌍, CN7≠RG3(22/24 변수 유의차), 10-27 전후 공정 드리프트"

Figure: `NB00_fig3_eda_correlation_heatmap.png`

---

### Slide 6 — Phase 2: 전처리 Ablation

"Standard+SMOTE ROC 0.9311 최고. SMOTE는 점수보다 분산(std) 절반으로 안정화."

- 10조합(Scaler 2 × Resample 5) 비교표
- SMOTE: ROC +0.024, PR-AUC std 0.1145→0.0554
- 가이드북: 단일 표준화만 적용, 불균형 처리 없음

Figure: `NB01_fig1_preproc_ablation_heatmap.png`

---

### Slide 7 — Phase 3: 선형 베이스라인

"가이드북 'SVM best'는 PR-AUC 최하위(0.0890). QDA(0.3526)가 전 지표 1위."

- 6종 모델 결과표
- QDA vs SVM-RBF 대비
- ablation 없이 모델을 고르면 wrong answer

Figure: `NB02_fig2_baseline_pr_curves.png`

---

### Slide 8 — Phase 4-A: 차원축소

"PCA는 이 데이터에서 성능 저하. TreeTop-15가 원본 25보다 소폭 우위."

- PCA 80%(0.8061) vs PCA 99%(0.8915) vs None(0.9343) vs TreeTop-15(0.9380)
- 가이드북에 없는 단계 전체

Figure: `NB03_fig1_dim_reduction_comparison.png`

---

### Slide 9 — Phase 4-B: 앙상블·NN

"MLP[256,128,64]+Dropout ROC 0.9497, PR 0.4710. 가이드북 DNN 대비 Dropout이 PR-AUC +0.005."

- 5종 앙상블·NN 결과: MLP > RF > Guidebook DNN > GBM > AdaBoost
- 가이드북 DNN 재현(0.9468/0.4655)을 같은 표에 병렬 제시

Figure: `NB04_fig3_ensemble_nn_comparison.png`

---

### Slide 10 — Phase 5: CNN + Stacking

"Stacking(0.9462)이 MLP(0.9497) 미초과 — base learner 다양성 부족이 원인."

- CNN1D(32,64): ROC 0.9472
- Stacking < MLP: 통념의 반례

Figure: `NB05_fig1_phase5_summary.png`

---

### Slide 11 — 운용점 분석

"Precision≥0.99 운용 시 MLP: 불량 71건 중 23건 탐지(Recall 32.4%)"

- MLP: Prec≥0.95→Recall 33.8%, Prec≥0.99→Recall 32.4%
- "71건 중 48건 탐지 안 됨" — 도메인 전문가 판단 필요

Figure: `NB06_fig4_phase6_final_operating_point.png`

---

### Slide 12 — 단계별 기여도 종합

"Phase 4 비선형 전환 PR-AUC +0.119이 최대 기여. Stacking은 음의 기여(반례)."

| Phase | 처리 | 기여 |
|---|---|---|
| Phase 2 | None → SMOTE | ROC +0.024 |
| Phase 3 | LR → QDA | PR +0.084 |
| Phase 4 | QDA → MLP | PR +0.119 |
| Phase 5 | 단일→Stacking | PR −0.023 |

---

### Slide 13 — 비지도 이상탐지 (신규)

"OC-SVM ROC 0.88 — 레이블 없이도 양품 패턴으로 불량 탐지 가능. K-Means는 도메인 불일치로 실패(ROC 0.47) — 원인 분석 완료."

세 방법 비교:

| 방법 | 학습 데이터 | ROC-AUC | 설명 |
|---|---|---|---|
| One-Class SVM | labeled 양품 | **0.8814** | L7-8 강의 |
| K-Means (k=5) | unlabeled 795K | 0.4697 | 도메인 불일치 실패 |
| Denoising AE | 구현 예정 | — | 가이드북 §2.2.1 |

K-Means 실패 원인: unlabeled 795K는 다기계·다제품, labeled는 CN7/RG3 단일 기계. 동일 기계·제품 필터링 후 재실험 예정.

Figure: `anomaly_detection_roc.png`, `anomaly_score_distribution.png`

---

### Slide 14 — Limitation & Future Work

"6가지 한계, 각각에 데이터 기반 근거와 개선 방향."

1. K-Means 이상탐지 도메인 불일치 → 동일 기계·제품 필터링 재실험
2. Denoising AE 미구현 → labeled 양품 / labeled+unlabeled 두 버전 비교 예정
3. 시계열성 미활용 → 슬라이딩 윈도우 K=10 기반 1D-CNN
4. CN7/RG3 분리 미구현 → 제품별 분리 모델 ablation
5. Pseudo-labeling 미시도 → 가이드북 §2.2.2 재현 (Future Work)
6. SMOTE 한계 → Focal Loss/Cost-sensitive 탐색

---

### Slide 15 — 결론

"단계별 ablation이 '어느 단계가 성능을 좌우했는가'를 수치로 답했다. 가이드북이 SVM best라 주장한 것을 QDA로 반박했고, unlabeled 795K 활용 가능성을 확인 + 도메인 불일치 문제도 발견했다."

주요 수치 요약:
- Phase 1~6 best: MLP ROC 0.9497 / PR 0.4710 (가이드북 DNN 초과)
- 이상탐지: OC-SVM ROC 0.8814 (labeled 양품만으로 달성)
- AE (구현 예정) + K-Means 개선 실험으로 Proposal에서 방향성 제시

---

### Slide 16 — Q&A 백업

예상 질문:
1. "SMOTE를 fold 안에서만 fit했는가?" → OOF SMOTE 코드 스니펫
2. "SVM-RBF가 PR-AUC 최하위인 이유?" → probability calibration 없이 decision_function 사용
3. "K-Means 이상탐지가 0.47로 나쁜 이유?" → 도메인 불일치 설명 + 개선 방향
4. "CNN 입력이 강제적이지 않은가?" → 가능성 탐색 목적, 한계 인정

백업 자료:
- 전처리 10조합 전체 결과표
- Phase 3~5 모든 모델 결과표
- 이상탐지 3방법 설계 도표

---

## 발표 시간 배분 (15분 기준)

| 슬라이드 | 예상 시간 |
|---|---|
| 1~2 (표지, 문제 정의) | 1분 |
| 3~4 (데이터, 차별점) | 1분 30초 |
| 5~6 (EDA, 전처리) | 2분 |
| 7~9 (Phase 3~4) | 3분 |
| 10~12 (Phase 5, 운용점, 기여도) | 2분 30초 |
| 13 (이상탐지) | 2분 |
| 14~15 (한계, 결론) | 2분 |
| 16 (Q&A) | 질문 대응 |

---

*v1: 2026-05-10 / v2: 2026-05-16 — 비지도 이상탐지 슬라이드(13) 추가, 한계 업데이트*
