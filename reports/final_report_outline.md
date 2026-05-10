# 최종 보고서 아웃라인 — KAMP 사출성형 단계별 Ablation

> **작성:** 감독관 에이전트, 2026-05-10  
> **대상 보고서:** reports/final_report.pdf (8~12쪽)  
> **근거 파일:** results/ablation_summary.md, results/decisions.md, reports/phase_results_summary.md

---

## 전체 구조 요약

| # | 섹션 | 예상 쪽수 | 핵심 Figure/Table |
|---|---|---|---|
| 1 | Abstract | 0.5 | — |
| 2 | Introduction | 1.0 | Table 1 (가이드북 vs 우리 비교) |
| 3 | Dataset & Preprocessing | 1.5 | Fig 1~3, Table 2 |
| 4 | Methodology | 1.0 | Fig 4 (Ablation 설계 도표) |
| 5 | Results | 3.0 | Table 3~6, Fig 5~9 |
| 6 | Discussion | 1.0 | Table 7 (단계별 기여도) |
| 7 | Limitation & Future Work | 1.5 | — (수치 근거 중심) |
| 8 | Conclusion | 0.5 | — |
| 9 | References | 0.5 | — |
| Appendix | 활동계획 (5% 배점) | 0.5~1.0 | Table A1 |
| **합계** | | **10~11쪽** | |

---

## 1. Abstract (예상 0.5쪽)

KAMP 사출성형 데이터(7,996행, 불량률 0.89%)에 강의 14개 알고리즘을 4단계 ablation으로 적용한 결과, 단계별 성능 기여도를 분리 측정했다. 가이드북(KAIST·UNIST §2.3)이 "AE/SVM/DNN 순 단일 best 선택"으로 제시한 접근법의 분석적 한계를 지적하고, 비선형 모델 전환(Phase 4)이 PR-AUC +0.119로 최대 기여 단계임을 정량화했다. 최종 best는 MLP[256,128,64]+SMOTE (ROC-AUC 0.9497, PR-AUC 0.4710)이며, Stacking이 단일 MLP를 넘지 못한 반례도 보고한다.

사용 Figure/Table: 없음 (순수 텍스트)

---

## 2. Introduction (예상 1.0쪽)

사출성형 불량 예측의 산업적 중요성(자동차 앞유리 사이드 몰딩, PassOrFail 라벨 기준 불량률 0.89%)을 서술한다. 가이드북 §2.3의 "AE→SVM→DNN 순서 학습 후 best 1개 선택" 흐름이 각 처리 단계의 독립적 기여도를 측정하지 않는다는 점을 비판하고, 4-Phase Ablation이 차별점(originality)임을 명확히 한다. 강의 14개 알고리즘을 모두 체계적으로 시연하는 구조를 연구 목적으로 제시한다.

사용 Figure/Table:  
- Table 1: 가이드북(KAIST·UNIST §2.3) vs 우리 접근 비교표 (데이터 전처리 / 차원축소 / 모델 선택 방식 / 운용점 분석 여부)

---

## 3. Dataset & Preprocessing (예상 1.5쪽)

KAMP 사출성형 데이터셋 구성(labeled_data.csv 7,996행 × 45컬럼, 메타 9개 + 수치 변수 36개)을 기술한다. 분산=0 변수 10개(Mold_Temperature_1,2,5~12) 및 Barrel_Temperature_7(near-zero variance), PART_FACT_SERIAL(ID leak) 제거 후 유효 변수 25개 확정 과정을 서술한다. EDA의 핵심 발견—강한 상관 짝 34쌍, CN7·RG3 KS 검정 22/24 변수 유의 차이—을 보고한다. 전처리 Ablation(10조합 5-fold CV) 결과 StandardScaler+SMOTE 확정(ROC-AUC 0.9311, PR-AUC std 0.0554)을 기술한다.

사용 Figure/Table:  
- Fig 1: `eda_class_distribution.png` — 양품/불량 분포 (99.11% / 0.89% 극단 불균형 시각화)
- Fig 2: `eda_correlation_heatmap.png` — 상관 행렬 히트맵 (|r|>0.95 34쌍 식별)
- Fig 3: `preproc_ablation_heatmap.png` — 전처리 Ablation 결과 히트맵 (Scaler × Resample 10조합)
- Table 2: 제거 변수 목록 (변수명 / 제거 사유 / 가이드북 일치 여부)

---

## 4. Methodology (예상 1.0쪽)

4-Phase Ablation의 전체 구조를 설명한다. 각 Phase는 독립 변수 1개(전처리 방식 / 모델 패밀리 / 차원축소 방법 / 메타 학습기)를 변경하고 나머지를 고정해 성능 기여도를 분리 측정한다. 평가 지표(주: ROC-AUC + PR-AUC, 보조: F1, Precision@Recall=0.5/0.7/0.9), Stratified 5-fold CV(fold_0~4.npy 공유), 데이터 누수 방지(SMOTE/Scaler를 fold 내부에서 fit) 방침을 명시한다.

사용 Figure/Table:  
- Fig 4: 4-Phase Ablation 설계 도표 (Phase 2→3→4→5 흐름, 고정/변경 변수 표시 — 수동 작성 또는 표 형태로 대체 가능)

---

## 5. Results (예상 3.0쪽)

### 5.1 Phase 2 — 전처리 Ablation (0.5쪽)

10조합 비교 결과를 표로 제시한다. StandardScaler+None 대비 SMOTE가 ROC-AUC +0.024 향상, PR-AUC 표준편차를 절반으로(0.1145→0.0554) 줄인 점을 강조한다. 가이드북에 없는 단계임을 명시한다.

- Table 3: 전처리 Ablation 전체 결과 (Scaler / Resample / ROC-AUC±std / PR-AUC±std)
- Fig 3 재사용

### 5.2 Phase 3 — 선형·생성적 베이스라인 (0.75쪽)

6종 모델(LR-L1, LR-L2, LDA, QDA, SVM-linear, SVM-RBF) 결과를 제시한다. 가이드북이 "SVM best"로 명시한 SVM-RBF가 PR-AUC 기준 최하위(0.0890)이고, 가이드북 미시도 모델인 QDA(PR-AUC 0.3526)가 1위임을 강조한다. 이것이 단계별 ablation의 originality 핵심 근거임을 서술한다.

- Table 4: Phase 3 결과표 (모델 / 최적 파라미터 / ROC-AUC / PR-AUC / F1 / 가이드북 비교)
- Fig 5: `baseline_roc_curves.png` — 6종 모델 ROC 곡선
- Fig 6: `baseline_pr_curves.png` — 6종 모델 PR 곡선

### 5.3 Phase 4 — 차원축소 + 앙상블 + NN (1.0쪽)

차원축소 Ablation(PCA 4종 vs 트리 중요도 3종) 결과: PCA는 성능 저하, TreeTop-15(0.9380)가 full(0.9343)을 소폭 초과했다. 앙상블·NN 결과: MLP[256,128,64]+Dropout=0.3이 ROC-AUC 0.9497, PR-AUC 0.4710으로 전 Phase 최고치 달성. 가이드북 DNN 재현(0.9468/0.4655) 대비 우리 MLP가 소폭 우위임을 비교 표에 제시한다.

- Table 5: 차원축소 Ablation 결과 (방법 / 차원 수 / ROC-AUC / PR-AUC / 가이드북 비교)
- Table 6: Phase 4-B 앙상블·NN 결과 (가이드북 DNN 포함 동일 표)
- Fig 7: `dim_reduction_comparison.png` — 차원축소 방법별 성능 비교
- Fig 8: `ensemble_roc_curves.png` + `ensemble_pr_curves.png` — 앙상블·NN ROC/PR 곡선

### 5.4 Phase 5 — CNN + Stacking + 운용점 (0.75쪽)

1D-CNN(25 피처 → Conv1d 2층 → FC → Dropout) 결과(ROC-AUC 0.9472, PR-AUC 0.4501). Stacking(OOF 메타 LR) 결과가 단일 best MLP(0.9497)를 넘지 못한 반례(0.9462)를 제시하고 원인(base learner 다양성 부족)을 분석한다. 운용점 분석: Precision≥0.99 운용 시 MLP Recall 32.4%.

- Table 7 (운용점): 모델 / Prec 목표 / 달성 Prec / Recall / 임계값
- Fig 9: `phase5_summary.png` — 전 Phase 모델 통합 ROC/PR 곡선 (운용점 마커 포함)

---

## 6. Discussion (예상 1.0쪽)

"어느 단계가 성능을 가장 끌어올렸는가?"를 Phase별 기여도 정량화로 답한다. 가장 큰 기여는 Phase 4 비선형 모델 전환(PR-AUC +0.119)이고, 두 번째는 Phase 3 모델 패밀리 내 최적화(PR-AUC +0.084)다. 불균형 처리(Phase 2)는 점수 향상보다 분산 안정화 역할을 했다. Stacking 반례가 "항상 앙상블이 낫다"는 통념을 반박함을 논의한다. 가이드북의 "AE→SVM→DNN 단일 best" 방식이 QDA 같은 숨겨진 강자를 놓치는 위험을 정리한다.

사용 Figure/Table:  
- Table 8: 단계별 성능 기여도 요약 (Phase / 처리 단계 / 기여 지표 / 기여 크기)

---

## 7. Limitation & Future Work [25% 배점] (예상 1.5쪽)

5가지 구체적 한계를 각각 수치 근거와 함께 서술한다. (상세 내용은 `reports/limitation_section.md` 참조.)

1. 시계열성 활용 미흡 — 초 단위 사이클 내 패턴 미분석
2. CN7/RG3 분리 분석 미완 — KS 검정 22/24 유의 차이에도 혼합 분석 유지
3. Autoencoder 미재현 — 비지도 이상 탐지 관점 미적용
4. Stacking이 단일 best 미초과 — base learner 다양성 부족
5. 극심한 클래스 불균형 잔여 문제 — fold당 불량 ~14건, PR-AUC std~0.10

사용 Figure/Table: 없음 (수치 근거 내재)

---

## 8. Conclusion (예상 0.5쪽)

4-Phase Ablation이 가이드북의 단일 best 선택 방식 대비 각 처리 단계의 기여도를 정량화한다는 점이 핵심 기여임을 정리한다. 최종 best 모델(MLP[256,128,64], ROC-AUC 0.9497), 가이드북 대비 성능 향상(PR-AUC 0.4710 vs 0.4655), 비선형 모델 전환이 최대 기여 단계임을 재확인한다. 향후 시계열 모델링 및 CN7/RG3 분리 분석 방향을 1~2문장으로 언급한다.

사용 Figure/Table: 없음

---

## 9. References

- 가이드북: KAIST·UNIST·㈜이피엠솔루션즈, "사출성형기 AI 데이터셋 가이드북", 2020.
- KAMP 플랫폼: https://www.kamp-ai.kr
- 데이터 출처: johnwslee/injection_molding_analysis (GitHub)
- scikit-learn, imbalanced-learn, PyTorch 공식 문서
- 강의 슬라이드 (L4~L16, NOVA50101)

---

## Appendix: 활동계획 [5% 배점] (예상 0.5~1.0쪽)

팀원별 Phase별 기여도를 표로 제시한다. 원안(제안서 v0 기준 역할 분담)과 실제 수행 결과를 비교한다. GitHub commit 이력 및 팀 회의 기록을 근거로 삼는다.

사용 Figure/Table:  
- Table A1: 팀원별 기여도 표 (멤버 / Phase 2~6 분담 / 주요 산출물)

---

## Figure 마스터 목록

| Fig 번호 | 파일명 | 사용 섹션 |
|---|---|---|
| Fig 1 | `eda_class_distribution.png` | §3 |
| Fig 2 | `eda_correlation_heatmap.png` | §3 |
| Fig 3 | `preproc_ablation_heatmap.png` | §3, §5.1 |
| Fig 4 | 4-Phase 설계 도표 (수동 작성) | §4 |
| Fig 5 | `baseline_roc_curves.png` | §5.2 |
| Fig 6 | `baseline_pr_curves.png` | §5.2 |
| Fig 7 | `dim_reduction_comparison.png` | §5.3 |
| Fig 8 | `ensemble_roc_curves.png` / `ensemble_pr_curves.png` | §5.3 |
| Fig 9 | `phase5_summary.png` | §5.4 |

---

*작성: 감독관 에이전트 (2026-05-10)*
