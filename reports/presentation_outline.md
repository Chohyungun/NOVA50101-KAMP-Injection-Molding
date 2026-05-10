# 발표 PPT 슬라이드 아웃라인 (14슬라이드)

> **작성:** 감독관 에이전트, 2026-05-10  
> **근거:** TERM_PROJECT_GUIDELINE.md §6.2, reports/phase_results_summary.md  
> **발표 예상 시간:** 10~15분 (슬라이드당 평균 45초~1분)

---

## 슬라이드 구성 개요

| # | 제목 | 핵심 메시지 | Figure |
|---|---|---|---|
| 1 | 표지 | 팀 정보 + 연구 제목 | — |
| 2 | 문제 정의 | 불량 예측 + 가이드북 비판 | — |
| 3 | 데이터셋 개요 | 7,996행 × 25 변수, 불량 0.89% | Fig: 클래스 분포 |
| 4 | 우리의 차별점 | 단계별 Ablation = Originality | Fig: 설계 도표 |
| 5 | EDA 핵심 | 분산 0 변수 제거, CN7≠RG3 | Fig: 상관 히트맵 |
| 6 | Phase 2 — 전처리 Ablation | SMOTE가 PR-AUC 분산 절반으로 | Fig: Ablation 히트맵 |
| 7 | Phase 3 — 선형 베이스라인 | QDA가 SVM "best"를 뒤집음 | Fig: PR 곡선 |
| 8 | Phase 4-A — 차원축소 | PCA 손해, 트리Top-15 소폭 우위 | Fig: 차원축소 비교 |
| 9 | Phase 4-B — 앙상블·NN | MLP가 가이드북 DNN 초과 | Fig: ROC/PR 앙상블 |
| 10 | Phase 5 — CNN + Stacking | Stacking이 MLP를 못 넘은 반례 | Fig: Phase5 종합 |
| 11 | 운용점 분석 | Prec≥0.99에서 MLP Recall 32.4% | Fig: PR 곡선 운용점 |
| 12 | 단계별 기여도 요약 | Phase 4 비선형 전환이 최대 기여 | Table: 기여도 표 |
| 13 | Limitation & Future Work | 5가지 한계 + 향후 방향 | — |
| 14 | Q&A 백업 | 상세 결과표 + 추가 그림 | — |

---

## 슬라이드별 상세 내용

---

### Slide 1 — 표지

핵심 한 문장:  
"KAMP 사출성형 불량 예측: 가이드북 단일 Best vs 단계별 Ablation"

들어갈 요소:
- 팀번호 / 팀원 이름 (조현건 외)
- 강좌명: NOVA50101 인공지능학개론
- 발표일: 2026년 학기말
- 부제: "강의 14개 알고리즘으로 어느 단계가 성능을 좌우하는지 분리 측정"

사용 Figure: 없음

---

### Slide 2 — 문제 정의

핵심 한 문장:  
"가이드북은 AE→SVM→DNN 순서로 하나를 골랐지만, 어느 처리 단계가 점수를 끌어올렸는지는 측정하지 않았다."

들어갈 요소:
- 사출성형 불량 예측의 산업 중요성 (자동차 앞유리 사이드 몰딩, PassOrFail 라벨)
- 가이드북(KAIST·UNIST §2.3) 흐름: AE → SVM → DNN → best 1개 선택
- 가이드북의 한계: 단계별 기여도 미분리, 불균형 처리 없음, 차원축소 비교 없음
- 우리의 접근: 4-Phase Ablation으로 기여도 분리 측정 → Originality

사용 Figure: 없음 (텍스트 + 화살표 다이어그램)

---

### Slide 3 — 데이터셋 개요

핵심 한 문장:  
"불량 71건/7,996건(0.89%)의 극단적 불균형 데이터에서 유효 변수 25개로 분석한다."

들어갈 요소:
- 데이터셋: KAMP 사출성형, labeled_data.csv, 7,996행 × 45컬럼(메타 9 + 수치 36)
- 유효 변수 25개 확정 과정 (분산=0 변수 10개 + near-zero 1개 + ID 1개 제거)
- 양품/불량 분포 수치: 7,925/71 = 99.11% / 0.89%
- 평가 지표: ROC-AUC (주), PR-AUC (주), Precision@Recall (보조)

사용 Figure: `eda_class_distribution.png` (클래스 분포 막대 그래프)

---

### Slide 4 — 우리의 차별점: 단계별 Ablation

핵심 한 문장:  
"각 Phase마다 독립 변수 하나만 바꾸고 나머지를 고정해 성능 기여도를 분리 측정한다."

들어갈 요소:
- 4-Phase 설계 도표:
  - Phase 2: 분류기 고정 → 전처리 10조합 비교
  - Phase 3: 전처리 고정 → 모델 패밀리 6종 비교
  - Phase 4: 모델 고정 → 차원축소 8종 / 앙상블·NN 비교
  - Phase 5: 1D-CNN 보조, OOF Stacking, 운용점 분석
- Stratified 5-fold CV 공유 (같은 fold split)
- 데이터 누수 방지: SMOTE/Scaler를 fold 내부에서 fit

사용 Figure: 4-Phase 설계 흐름 다이어그램 (PPT 직접 작성 권장)

---

### Slide 5 — EDA 핵심 발견

핵심 한 문장:  
"분산=0 변수 11개 제거, 강한 상관 짝 34쌍, CN7과 RG3는 22/24 변수에서 분포가 유의하게 다르다."

들어갈 요소:
- 분산=0 변수 목록 (Mold_Temperature 계열 10개, Barrel_Temp_7) — 가이드북 §2.1과 완전 일치
- 상관 히트맵: 피어슨 |r|>0.95 34쌍 (다중공선성 존재)
- CN7 vs RG3 KS 검정: 22/24 변수 p<0.05 → 제품별 분포 차이 유의
- 시계열 흐름: `eda_temporal_drift.png`에서 일부 변수 날짜별 드리프트 관찰

사용 Figure: `eda_correlation_heatmap.png` (상관행렬 히트맵)

---

### Slide 6 — Phase 2: 전처리 Ablation

핵심 한 문장:  
"StandardScaler + SMOTE가 ROC-AUC 0.9311로 최고이며, SMOTE는 점수 향상보다 PR-AUC 분산을 절반으로 줄여 결과 안정성에 기여한다."

들어갈 요소:
- 10조합(Scaler 2종 × Resample 5종) 5-fold CV 결과
- SMOTE 효과: ROC-AUC +0.024(None→SMOTE), PR-AUC std 0.1145→0.0554
- RobustScaler+None: ROC-AUC 0.3406 — saga solver 수렴 실패 반례
- 가이드북에 없는 단계: "가이드북은 단일 표준화만 적용"

사용 Figure: `preproc_ablation_heatmap.png` (10조합 히트맵)

---

### Slide 7 — Phase 3: 선형·생성적 베이스라인

핵심 한 문장:  
"가이드북이 'SVM best'라고 한 SVM-RBF의 PR-AUC는 0.0890으로 최하위이며, 가이드북 미시도 모델인 QDA(PR-AUC 0.3526)가 1위다."

들어갈 요소:
- 6종 모델(LR-L1, LR-L2, LDA, QDA, SVM-linear, SVM-RBF) 결과표
  - QDA(0.9344/0.3526) vs SVM-RBF(0.9297/0.0890) 대비
- PR-AUC 기준 순위: QDA > SVM-linear > LR > LDA > SVM-RBF
- Ablation의 가치: 체계적 탐색 없이 SVM을 선택한 가이드북의 위험성

사용 Figure: `baseline_pr_curves.png` (6종 PR 곡선 비교)

---

### Slide 8 — Phase 4-A: 차원축소 비교

핵심 한 문장:  
"PCA는 이 데이터에서 성능을 떨어뜨리고, 트리 중요도 Top-15가 원본 25 변수보다 소폭 우위를 보인다."

들어갈 요소:
- 차원축소 8종(PCA 4종 + 트리중요도 3종 + None) 비교표
  - PCA 80%(ROC 0.8061) vs PCA 99%(0.8915) vs 없음(0.9343) vs TreeTop-15(0.9380)
- PCA 성능 저하 원인: 강한 상관 34쌍이 PCA에 유리해 보이지만, 불균형 불량 패턴이 주성분에 녹아들지 않음
- 가이드북에 없는 단계 전체: 차원축소 ablation이 100% originality

사용 Figure: `dim_reduction_comparison.png` (차원축소 방법별 ROC/PR 비교)

---

### Slide 9 — Phase 4-B: 앙상블·NN (가이드북 DNN 비교)

핵심 한 문장:  
"MLP[256,128,64]+Dropout=0.3이 ROC-AUC 0.9497, PR-AUC 0.4710으로 전 Phase 최고; 가이드북 DNN(0.9468/0.4655) 대비 Dropout이 PR-AUC +0.005 기여."

들어갈 요소:
- 5종 앙상블·NN 결과: MLP > RF > Guidebook DNN > GBM > AdaBoost
- 가이드북 DNN 재현 결과를 동일 표에 병렬 제시 (1:1 비교)
- Phase3→4 향상: ROC-AUC +0.015, PR-AUC +0.12~+0.15
- AdaBoost PR-AUC ~0.50의 이변: class_weight stump reweighting 효과

사용 Figure: `ensemble_roc_curves.png` 또는 `ensemble_pr_curves.png`

---

### Slide 10 — Phase 5: 1D-CNN + Stacking

핵심 한 문장:  
"Stacking(ROC 0.9462)이 단일 MLP(0.9497)를 넘지 못한 것은 base learner 다양성 부족 때문이며, 이 반례 자체가 ablation의 발견이다."

들어갈 요소:
- 1D-CNN 구조: 25 피처 → Conv1d(ch=32,64, k=3) → MaxPool × 2 → FC(64) → Dropout → Output
  - CNN1D(32,64) ep=50: ROC 0.9472, PR 0.4501
- Stacking 구성: LR-L2 + QDA + RF + GBM + MLP → OOF 메타 LR
  - 결과: ROC 0.9462, PR 0.4484 (MLP 단독 미달)
- 원인 분석: LR-L2·QDA 상관 높아 다양성 부족 → 메타 학습기 학습 이점 없음

사용 Figure: `phase5_summary.png` (전 모델 통합 비교)

---

### Slide 11 — 운용점(Operating Point) 분석

핵심 한 문장:  
"Precision≥0.99로 운용하면 MLP가 불량 71건 중 약 23건(Recall 32.4%)을 탐지하며, 절반 이상은 현재 데이터로 탐지 불가다."

들어갈 요소:
- 운용점 분석 표:
  - MLP: Prec≥0.95 → Recall 33.8%, Prec≥0.99 → Recall 32.4%
  - Stacking: Prec≥0.95 → Recall 32.4%, Prec≥0.99 → Recall 23.9%
- "71건 중 48건 탐지 안 됨"의 현장 의미: 현재 데이터로 높은 Recall과 Precision 동시 달성 불가
- 산업 적용 시 "어느 수준의 Recall을 감수할 것인가"는 도메인 전문가 판단 필요

사용 Figure: `phase5_summary.png` 또는 `baseline_pr_curves.png`에 운용점 마커 추가 버전

---

### Slide 12 — 단계별 성능 기여도 종합

핵심 한 문장:  
"Phase 4 비선형 모델 전환이 PR-AUC +0.119로 최대 기여 단계이며, 불균형 처리(Phase 2)는 점수보다 안정성에 기여했다."

들어갈 요소:
- 단계별 기여도 표:

| Phase | 처리 | 기여 지표 | 기여 크기 |
|---|---|---|---|
| Phase 2 | None → SMOTE | ROC-AUC | +0.024 |
| Phase 3 | LR → QDA | PR-AUC | +0.084 |
| Phase 4 | QDA → MLP | PR-AUC | +0.119 |
| Phase 5 | 단일→Stacking | PR-AUC | −0.023 (반례) |

- 핵심 결론: "비선형 모델 전환 > 모델 내 최적화 > 불균형 처리 > Stacking(음의 기여)"
- 가이드북 비교: 가이드북 미탐색 단계(전처리 ablation, 차원축소, QDA)에서 가장 큰 발견

사용 Figure: 표 직접 PPT 작성 + 막대 그래프(optional)

---

### Slide 13 — Limitation & Future Work

핵심 한 문장:  
"시계열 미활용·CN7/RG3 미분리·AE 미재현·Stacking 다양성 부족·불균형 잔여, 다섯 가지 한계 각각에 강의 범위 내 향후 과제가 있다."

들어갈 요소:
1. 시계열성 미활용 → 슬라이딩 윈도우 K=10 기반 1D-CNN 재구성
2. CN7/RG3 분리 미완 → 제품별 분리 모델 vs product flag ablation
3. AE 미재현 → PCA 재구성 오차를 비지도 이상 점수로 활용
4. Stacking < MLP → SVM-RBF + CNN1D를 base에 추가
5. 불균형 잔여(fold당 불량 14건, PR-AUC std ~0.10) → 반지도 학습(PCA 기반), cost-sensitive 그리드

사용 Figure: 없음 (5-item 리스트, 각 항목에 수치 병기)

---

### Slide 14 — Q&A 대비 백업 슬라이드

핵심 한 문장:  
"예상 질문: (1) SMOTE를 fold 안에서만 fit했는가? (2) SVM-RBF가 PR-AUC 최하위인 이유? (3) CNN 입력 구성이 강제적이지 않은가?"

들어갈 요소:
- 상세 결과표 전체(전처리 10조합 + Phase 3~5 모든 모델) 백업
- 데이터 누수 방지 증거: fold 내부 SMOTE fit 코드 스니펫 또는 설명
- SVM-RBF PR-AUC 저하 원인: probability 보정(platt scaling) 없이 decision_function 기반 → 임계값 민감
- CNN 입력 구조 정당화: 강의 §CNN 구현 시연 가능성 확보가 목적이며, 시계열 입력의 한계는 Slide 13에서 인정
- 운용점 임계값 선택 근거: PR 곡선에서 Precision 목표값 고정 후 최대 Recall 점 탐색

사용 Figure: 해당 Phase 결과표 전체 + 필요 시 추가 그림

---

## 발표 시간 배분 참고 (15분 기준)

| 슬라이드 | 예상 시간 |
|---|---|
| 1~2 (표지, 문제 정의) | 1분 30초 |
| 3~4 (데이터, 차별점) | 2분 |
| 5~6 (EDA, 전처리) | 2분 |
| 7~9 (Phase 3, 4) | 3분 30초 |
| 10~11 (Phase 5, 운용점) | 2분 |
| 12 (기여도 종합) | 1분 |
| 13 (한계) | 1분 30초 |
| 14 (Q&A) | 질문 대응 시 사용 |

---

*작성: 감독관 에이전트 (2026-05-10)*
