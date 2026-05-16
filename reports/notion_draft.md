# Notion 업로드 초안 — KAMP 사출성형 Phase 1~7 결과

> **업로드 대상:** NOVA50101 팀 Notion 워크스페이스  
> **최종 업데이트:** 2026-05-16

---

## [Callout] 프로젝트 요약

KAMP 사출성형기 데이터(7,996행 labeled / 795,315행 unlabeled)로 단계별 Ablation 분석 수행.
가이드북(KAIST·UNIST) DNN(ROC 0.9468) 초과 달성: **MLP ROC 0.9497, PR-AUC 0.4710.**
추가로 비지도 이상탐지(OC-SVM ROC 0.88 / K-Means ROC 0.47 + 실패 원인 분석) 1차 실험 완료.

---

## [Table] 데이터셋 개요

| dataset | rows | has_label | 역할 |
|---|---|---|---|
| labeled_data | 7,996 | ✅ | 지도학습 주 데이터 |
| unlabeled_data | 795,315 | ❌ | 비지도 이상탐지 학습용 |
| moldset_labeled | 2,607 | ✅ | 준지도학습 |
| supervised_label_cn7 | 6,736 | ✅ | CN7 전용 |
| moldset_labeled_cn7/rg3 | 2,393 | ✅ | CN7/RG3 semi |
| moldset_unlabeled_cn7/rg3 | 71,180 | ❌ | CN7/RG3 semi unlabeled |

유효 피처: **25개** (분산=0 변수 11개 제거)  
불량 비율: **0.89%** (양품 7,925 / 불량 71)

---

## [Table] Phase 1~6 단계별 최고 성능

| Phase | 대표 모델 | ROC-AUC | PR-AUC | 비고 |
|---|---|---|---|---|
| Phase 2 (전처리) | LR-L2 + SMOTE | 0.9311 | 0.2413 | StandardScaler 확정 |
| Phase 3 (선형) | QDA (reg=0.01) | 0.9344 | 0.3526 | SVM "best" 반박 |
| Phase 4-A (차원축소) | LR-L2 + TreeTop-15 | 0.9380 | 0.2899 | PCA 성능 저하 확인 |
| Phase 4-B (앙상블) | MLP[256,128,64] | **0.9497** | **0.4710** | Guidebook DNN 초과 |
| Phase 5 (CNN) | CNN1D(32,64) | 0.9472 | 0.4501 | |
| Phase 5 (Stacking) | meta-LR | 0.9462 | 0.4484 | 단일 MLP 미초과 (반례) |
| 가이드북 DNN | DNN[128,64,32] | 0.9468 | 0.4655 | 우리가 직접 재현 |

---

## [Table] 비지도 이상탐지 1차 실험 결과 (2026-05-16)

| 방법 | 학습 데이터 | ROC-AUC (5-fold) | PR-AUC | 비고 |
|---|---|---|---|---|
| One-Class SVM (nu=0.02) | labeled 양품 7,925 | **0.8814 ± 0.027** | 0.1765 | L7-8 강의 직결 |
| K-Means (k=5) | unlabeled 795K 전체 | 0.4697 ± 0.063 | 0.1444 | 도메인 불일치로 실패 |
| K-Means (k=10~50) | unlabeled 795K 전체 | 0.31~0.40 | 0.06~0.07 | |
| Denoising AE | 구현 예정 | — | — | 가이드북 §2.2.1 재현 |

**K-Means 실패 원인:** unlabeled는 다기계·다제품 데이터, labeled는 CN7/RG3 단일 기계. 도메인 불일치로 centroid가 정상 패턴을 대표하지 못함 → 동일 기계·제품 필터링 후 재실험 예정.

---

## [Table] 가이드북 vs 우리 차별점 6가지

| 항목 | 가이드북 | 우리 |
|---|---|---|
| 불균형 처리 | 개념 1단락 (코드 없음) | 4종 10조합 5-fold 정량 ablation |
| 평가 지표 | accuracy + recall | ROC-AUC + PR-AUC + 운용점 |
| 전처리 비교 | 단일 표준화 | Scaler 2종 × Resample 5종 |
| 모델 탐색 | AE/SVM/DNN 3개 흐름 | 6종 선형 → 4종 앙상블 → CNN → Stacking |
| SVM 주장 | SVM best | QDA가 PR-AUC 기준 SVM 초과 |
| K-Means 이상탐지 | 없음 | 1차 시도 + 실패 원인 분석 + 개선 계획 |

---

## [Table] 단계별 기여도

| Phase | 처리 | 기여 지표 | 크기 |
|---|---|---|---|
| Phase 2 | None → SMOTE | ROC-AUC | +0.024 |
| Phase 3 | LR → QDA | PR-AUC | +0.084 |
| Phase 4 | QDA → MLP | PR-AUC | +0.119 |
| Phase 5 | 단일→Stacking | PR-AUC | −0.023 (반례) |

---

## [Table] 운용점 분석

| model | precision 목표 | 달성 precision | recall | 불량 탐지 |
|---|---|---|---|---|
| MLP[256,128,64] | ≥0.95 | 0.9600 | 33.8% | 71건 중 24건 |
| MLP[256,128,64] | ≥0.99 | 1.0000 | **32.4%** | **71건 중 23건** |
| Stacking | ≥0.99 | 1.0000 | 23.9% | 71건 중 17건 |

---

## [Toggle] 한계 및 Future Work

<details>
<summary>클릭하여 펼치기</summary>

1. **K-Means 이상탐지 도메인 불일치** — unlabeled를 동일 기계·제품으로 필터링 후 재실험 필요
2. **Denoising AE 미구현** — labeled 양품만 vs labeled+unlabeled 두 버전 비교 예정
3. **시계열성 미활용** — 사이클 간 공정 드리프트(EDA 10-27 전후 변화 확인) 미반영
4. **CN7/RG3 분리 미구현** — KS 검정 22/24 변수 유의차에도 혼합 학습 진행
5. **SMOTE 한계** — 불량 71개로 보간 품질 의심, Focal Loss/Cost-sensitive 미시도
6. **Pseudo-labeling 미시도** — 가이드북 §2.2.2(pseudo-labeling+entropy minimization) Future Work

</details>

---

## [Gallery] 주요 Figure

- `NB00_fig1_eda_class_distribution.png` — 양품/불량 분포
- `NB00_fig3_eda_correlation_heatmap.png` — 상관행렬 히트맵
- `NB01_fig1_preproc_ablation_heatmap.png` — 전처리 10조합 히트맵
- `NB02_fig1_baseline_roc_curves.png` — 선형 베이스라인 ROC
- `NB04_fig3_ensemble_nn_comparison.png` — 앙상블·NN 비교
- `NB05_fig1_phase5_summary.png` — Phase 5 종합 비교
- `anomaly_kmeans_k_ablation.png` — K-Means k 튜닝 결과
- `anomaly_detection_roc.png` — OC-SVM vs K-Means ROC/PR 곡선
- `anomaly_score_distribution.png` — 이상 점수 분포 (양품 vs 불량)

---

## Notion 업로드 방법 (mcp 도구 사용)

```
도구: mcp__claude_ai_Notion__notion-create-pages
상위 페이지: [사용자가 Notion workspace URL/ID 제공 필요]
블록 구성: Callout → Table × 5 → Toggle → Gallery
```

> 이미지는 Notion Image 블록으로 별도 업로드 (results/figures/ 경로에서)
