# NOVA50101 텀프로젝트 진행 가이드라인 — KAMP 사출성형

> **작성자:** 조현건 | **v1:** 2026-05-10 | **v2 업데이트:** 2026-05-16  
> **데이터셋:** KAMP 사출성형기 AI 데이터셋 (886,227행, 25개 유효 피처)  
> **현재 상태:** Phase 1~6 완료 / Phase 7(비지도 이상탐지) 진행 중 / Proposal 5/17 마감

---

## 0. 전체 상태 한눈에

| Phase | 기간 | 상태 | 핵심 결과 |
|---|---|---|---|
| Phase 1 (EDA) | 5/11~12 | ✅ 완료 | 유효 피처 25개, 불량률 0.89% |
| Phase 2 (전처리 ablation) | 5/13~19 | ✅ 완료 | StandardScaler+SMOTE ROC 0.9311 |
| Phase 3 (선형 베이스라인) | 5/20~26 | ✅ 완료 | QDA ROC 0.9344, PR 0.3526 |
| Phase 4-A (차원축소) | 5/27~6/1 | ✅ 완료 | TreeTop-15 최고, PCA 역효과 |
| Phase 4-B (앙상블·NN) | 5/27~6/2 | ✅ 완료 | MLP ROC **0.9497**, PR **0.4710** |
| Phase 5 (CNN·Stacking) | 6/3~9 | ✅ 완료 | Stacking < 단일 MLP (반례) |
| Phase 6 (통합 보고) | 6/10~ | ✅ 완료 | 운용점 Prec≥0.99 → Recall 32.4% |
| **Phase 7 (이상탐지)** | **5/16~** | **⏳ 진행 중** | OC-SVM ROC 0.8814 / K-Means ROC 0.47 |
| **Proposal 제출** | **5/17 마감** | **⏳ 초안 완성** | `docs/Proposal_NOVA50101.md` |

---

## 1. 핵심 수치 요약

### 1.1 Phase별 최고 성능

| Phase | 모델 | ROC-AUC | PR-AUC | vs 가이드북 |
|---|---|---|---|---|
| Phase 2 | LR-L2 + SMOTE | 0.9311 | 0.2413 | 가이드북 미비교 |
| Phase 3 | QDA (reg=0.01) | 0.9344 | 0.3526 | SVM "best" 반박 |
| Phase 4-B | **MLP[256,128,64]** | **0.9497** | **0.4710** | 가이드북 DNN +0.003/+0.005 |
| Phase 5 | CNN1D(32,64) | 0.9472 | 0.4501 | |
| 가이드북 DNN | 재현 | 0.9468 | 0.4655 | 기준값 |

### 1.2 단계별 성능 기여도

| 단계 | 처리 | 기여 |
|---|---|---|
| Phase 2 | None → SMOTE | ROC +0.024 |
| Phase 3 | LR → QDA | PR +0.084 |
| Phase 4 | QDA → MLP | PR +0.119 ← **최대** |
| Phase 5 | 단일→Stacking | PR −0.023 (반례) |

### 1.3 비지도 이상탐지 1차 결과

| 방법 | ROC-AUC | 비고 |
|---|---|---|
| One-Class SVM | **0.8814** | 구현 완료, L7-8 |
| K-Means (k=5) | 0.4697 | 도메인 불일치 실패 → 개선 예정 |
| Denoising AE | — | 구현 예정 (가이드북 §2.2.1) |

---

## 2. 현재 할 일 (Phase 7)

### 2.1 Denoising AE 구현 (`run_autoencoder.py`)

```python
# 비교 구조
Config 1: AE(labeled 양품 7,925행만)        ← 가이드북 §2.2.1 재현
Config 2: AE(labeled 양품 + unlabeled 795K) ← 우리 확장

임계값 A: +5σ  (가이드북)
임계값 B: PR 곡선 운용점 (우리)

# 구조
Encoder: FC(25→16→8) + Dropout(0.1) + ReLU
Decoder: FC(8→16→25) + ReLU
Loss: MSELoss, Optimizer: Adam(1e-3), Epochs: 50

# 평가
labeled val fold에서 ROC-AUC, PR-AUC 측정
```

### 2.2 K-Means 개선 실험

```python
# 실패 원인
unlabeled 전체(795K) = 여러 기계·제품 혼재
labeled = CN7/RG3 단일 기계만 → centroid 도메인 불일치

# 개선
1. unlabeled_data.csv에 EQUIP_NAME/PART_NAME 컬럼 존재 확인
2. 있으면 → 동일 기계·제품으로 필터링 후 K-Means 재학습
3. 1차(ROC 0.47) vs 2차 비교 결과 보고
```

---

## 3. Proposal (5/17 마감)

- 제출 파일: `docs/Proposal_NOVA50101.md` → PDF 변환 후 Blackboard 업로드
- 팀장(조현건)이 제출
- 팀원 이름·역할 표 실명 채우기 필요
- 상세 방향성: `docs/Proposal_방향성_팀가이드.md`

---

## 4. 데이터 인벤토리

| 파일 | 행수 | 레이블 | 역할 |
|---|---|---|---|
| `labeled_data.csv` | 7,996 | ✅ | 지도학습 주 데이터 |
| `unlabeled_data.csv` | 795,315 | ❌ | 이상탐지 학습 (현재 접근 가능) |
| `moldset_labeled.csv` | 2,607 | ✅ | 준지도학습 |
| `supervised_label_cn7.csv` | 6,736 | ✅ | CN7 전용 |
| `moldset_labeled_cn7/rg3` | 2,393 | ✅ | 제품별 semi |
| `moldset_unlabeled_cn7/rg3` | 71,180 | ❌ | 제품별 unlabeled |

**PassOrFail:** Y=7,925(양품) / N=71(불량) → 불량률 0.89%  
**유효 피처:** 25개 (분산=0 11개 + ID 1개 제거)

---

## 5. 실험 설계 원칙

### 5.1 데이터 누수 방지

- SMOTE·스케일링은 반드시 **fold 안**에서 fit
- K-Means를 unlabeled로 학습할 때는 leakage 없음 (레이블 불필요)
- AE 학습 시 labeled val fold 양품 포함하지 않음
- 5-fold split: `data/splits/fold_*.npy` 재사용

### 5.2 Figure 저장 규칙

- 노트북 생성: `NB{nn}_fig{n}_{설명}.png`
- 스크립트 생성: `{주제}_{설명}.png` (`anomaly_*`, `dim_reduction_*`)
- **반드시 `sns.set_theme()` 이후 `setup_korean_font()` 호출** (한글 깨짐 방지)

### 5.3 평가 지표

- 주: ROC-AUC, PR-AUC
- 보조: F1, Precision@Recall (운용점)
- **Accuracy 단독 보고 금지** (불량 0.89% 환경에서 무의미)
- 이상탐지 평가: labeled val fold 기준

---

## 6. 강의 알고리즘 커버리지 (완료 + 예정)

| 강의 | 알고리즘 | 상태 |
|---|---|---|
| L4-6 | LR(L1/L2), LDA, QDA | ✅ |
| L7-8 | SVM(linear/RBF), **One-Class SVM** | ✅ |
| L8-10 | RF, AdaBoost, GBM, Stacking | ✅ |
| L9-10 | **K-Means** (이상탐지), GMM·DBSCAN (Future) | ⚠️ |
| L11 | PCA, **UMAP** (Future) | ✅ / Future |
| L11-14 | MLP+Dropout, **Denoising AE** | ✅ / ⏳ |
| L15-16 | 1D-CNN | ✅ |

---

## 7. 산출물 마스터 체크리스트

### 즉시 (5/17)
- [ ] `docs/Proposal_NOVA50101.md` → PDF → Blackboard 제출
- [ ] 팀원 이름 실명 채우기

### 최종 제출 (학기말)
- [ ] `reports/final_report.pdf` (8~12페이지)
- [ ] `final_presentation.pptx` (16슬라이드, `reports/presentation_outline.md` 참조)
- [ ] `Activity_Appendix.pdf` (팀원별 기여 표)
- [ ] Denoising AE 구현 완료 (`run_autoencoder.py`)
- [ ] K-Means 개선 실험 완료 (unlabeled 필터링)
- [ ] `pytest tests/ -v` 전 통과 상태 유지

---

## 8. 자주 빠지는 함정

1. **데이터 누수** — SMOTE/스케일링은 fold 내부에서 fit. 외부이면 ROC 0.05+ 부풀림
2. **Accuracy 보고** — 불량 0.89%라 의미없음. ROC-AUC + PR-AUC 사용
3. **fold 불일치** — 모든 Phase가 동일 `fold_*.npy` 사용해야 비교 가능
4. **한글 폰트** — `sns.set_theme()` 이후 `setup_korean_font()` 호출
5. **K-Means 도메인** — unlabeled 전체 학습 시 다기계·다제품 혼재 주의
6. **AE val leakage** — labeled val fold 양품을 AE 학습에 포함하지 않기

---

## 9. 파일 위치 참조

| 파일 | 역할 |
|---|---|
| `docs/Proposal_NOVA50101.md` | **Proposal 제출 초안** |
| `docs/Proposal_방향성_팀가이드.md` | 팀 내부 방향성 상세 |
| `docs/NOVA50101_Default_Final_Project_EN.md` | 과제 명세 |
| `docs/04.-Guidebook_InjectionMolding_EN.md` | 가이드북 전문 |
| `reports/project_report_full.md` | Phase 1~7 통합 보고서 |
| `reports/presentation_outline.md` | 발표 아웃라인 (16슬라이드) |
| `results/ablation_summary.md` | Phase별 결과 누적 |
| `results/decisions.md` | 기술 결정 로그 |
| `AGENT_INSTRUCTIONS.md` | 개발 에이전트 지침 |

---

*v1: 2026-05-10 (Phase 0 제안서 마감 기준) | v2: 2026-05-16 (Phase 1~6 완료, Phase 7 진행 중)*
