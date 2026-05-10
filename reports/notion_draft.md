# Notion 업로드 준비 완료 — 시니어 에이전트 최종 승인 대기

## 업로드 예정 페이지 제목
KAMP 사출성형 텀프로젝트 — Phase 2~5 Ablation 결과 (NOVA50101)

## 블록 구성 계획

### 데이터 개요
- Callout 블록: "886,227행 8개 CSV, 유효 변수 25개, 불량률 0.89%"
- Table 블록: eda_dataset_summary.csv 내용

| dataset | rows | cols | memory_MB | max_null_pct | has_label |
|---|---|---|---|---|---|
| labeled_data | 7,996 | 45 | 6.36 | 99.1% | True |
| moldset_labeled | 2,607 | 46 | 2.23 | 98.0% | True |
| unlabeled_data | 795,315 | 45 | 657.32 | 0.0% | False |
| supervised_label_cn7 | 6,736 | 25 | 1.35 | 0.0% | True |
| moldset_labeled_cn7 | 1,211 | 25 | 0.24 | 0.0% | True |
| moldset_unlabeled_cn7 | 35,239 | 24 | 6.77 | 0.0% | False |
| moldset_labeled_rg3 | 1,182 | 25 | 0.24 | 0.0% | True |
| moldset_unlabeled_rg3 | 35,941 | 24 | 6.90 | 0.0% | False |

### 단계별 성능 요약 표
| Phase | 대표 모델 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Phase 2 (전처리 기준) | LR-L2+SMOTE | 0.9311 | 0.2413 |
| Phase 3 (선형 best) | QDA(reg=0.01) | 0.9344 | 0.3526 |
| Phase 4 (앙상블 best) | MLP[256,128,64] | 0.9497 | 0.4710 |
| Phase 5 (CNN) | CNN1D(32,64) | 0.9472 | 0.4501 |
| Phase 5 (Stacking) | meta-LR | 0.9462 | 0.4484 |
| 가이드북 DNN 재현 | DNN[128,64,32] | 0.9468 | 0.4655 |

### 운용점 분석
- Precision≥0.95: MLP Recall=33.8%
- Precision≥0.99: MLP Recall=32.4% (불량 71건 중 23건 탐지)

### 첨부 이미지 (업로드 시 Notion Image 블록)
1. eda_enhanced_boxplots.png
2. preproc_ablation_heatmap.png
3. baseline_roc_curves.png
4. ensemble_roc_curves.png
5. phase5_summary.png
6. eda_pca_2d_scatter.png

### 한계 및 향후 과제 (Toggle 블록)
- 시계열성 활용 미흡 (TimeStamp 단위 미분석)
- CN7/RG3 분리 분석 미완 (KS sig 22/24)
- AE 미재현 (강의 범위 외, D-013)
- Stacking < MLP 단일 (base learner 다양성 부족)
- 클래스 불균형 잔여 (fold당 불량 ~14건)

## 업로드 전 확인 필요 사항
- [ ] Notion workspace 접속 권한 확인
- [ ] 업로드할 상위 페이지 URL/ID 입력 (사용자 제공 필요)
- [ ] 이미지 파일 Notion 업로드 허용 용량 확인
- [ ] 시니어 에이전트 최종 검토 및 승인

## Notion 업로드 명령 (승인 후 실행)
mcp__claude_ai_Notion__notion-create-pages 도구 사용 예정
