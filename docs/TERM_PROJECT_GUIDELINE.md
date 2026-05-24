# NOVA50101 텀프로젝트 진행 가이드라인 — KAMP 사출성형

> **작성자:** 조현건 | **v1:** 2026-05-10 | **v2:** 2026-05-16 | **v3:** 2026-05-24 | **v4:** 2026-05-24  
> **데이터셋:** KAMP 사출성형기 AI 데이터셋 (886,227행, 25개 유효 피처)  
> **현재 상태:** Phase 1~7 진행 중 / 방법 C v3 심사 비판 수용 후 재실행 중

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
| **Phase 7 (이상탐지)** | **5/16~** | **⏳ 진행 중** | OC-SVM ROC **0.8814** / K-Means ROC 0.4697 / 방법 C v3 실행 중 (심사 비판 수용) |
| **Proposal 제출** | **5/17 마감** | **✅ 제출 완료** | 팀장 전정은 제출 / 방법 A·B·C 포함 |

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

### 1.3 비지도 이상탐지 + 방법 C 결과

| 방법 | ROC-AUC | PR-AUC | 비고 |
|---|---|---|---|
| One-Class SVM | **0.8814** | 0.1765 | 구현 완료, L7-8 |
| K-Means (k=5) | 0.4697 | — | 도메인 불일치 실패 (795K 혼재) |
| 방법 C v2 RF+B | 0.9613 | 0.4735 | 버그수정 후 결과 (스케일러 불일치 잔존) |
| 방법 C v2 RF+D | 0.9608 | 0.4653 | 동일 |
| **방법 C v3** | 실행 중 | 실행 중 | 심사 비판 4개 FATAL 전면 수정 후 |
| Denoising AE | — | — | 구현 예정 (가이드북 §2.2.1) |

### 1.4 방법 C 심사 비판 수용 목록 (v3)

| # | 심각도 | 비판 | 처리 |
|---|---|---|---|
| 1 | FATAL | 스케일러 공간 불일치 (unlabeled-fit → labeled-train 불일치) | sc_sel=labeled-only 통일 |
| 2 | FATAL | 전략 B 소프트 누수 (teacher RF가 val fold 포함) | fold별 teacher RF |
| 3 | FATAL | align_score 동등 가중치 (크기 2380배 피처 동일 취급) | 표준화 공간 + RF importance 가중 |
| 4 | FATAL | 전략 C 방향벡터 raw space (Clamp 99% 지배) | 표준화 공간 방향벡터 |
| 5 | MAJOR | 통계 검정 없이 IMPROVED 주장 | Wilcoxon signed-rank (n=5) |
| 6 | MAJOR | n ablation 전략 C만 수행 | B·D 전략도 ablation |
| 7 | MAJOR | ST threshold=0.80 근거 없음 | threshold sweep [0.5~0.8] |
| 8 | MINOR | fillna(0) 검증 없음 | 결측 패턴 분석 셀 추가 |

---

## 2. 현재 할 일 (Phase 7 — 방법 A·B·C)

> **팀 회의 결정 (2026-05-24):** 방법 C → 방법 A 순으로 조현건이 진행. 방법 B는 다른 팀원 담당.

### 2.1 방법 C — Semi-supervised Pseudo-labeling (조현건, 1순위)

**현재: v3 재설계 완료 + 실행 중 (심사 비판 4개 FATAL 전면 수용)**

#### 구현 현황 (노트북: `08_method_c_v2.ipynb`)

| 버전 | 상태 | 핵심 내용 |
|---|---|---|
| v1 | ✅ 완료·실패 | DBSCAN noise → 기계정지 오염 28% 방향일치 |
| v2 | ✅ 완료 (버그수정) | 기계정지 필터 + 4전략 + run_cv 버그 수정 |
| v3 | ⏳ 실행 중 | FATAL 비판 4개 + MAJOR 3개 + MINOR 1개 전면 수용 |

#### v3 핵심 설계 원칙

```
[스케일러 분리]
  sc_vis = StandardScaler().fit(X_unl)   ← 시각화(PCA)만 사용
  sc_sel = StandardScaler().fit(X_lab)   ← 전략 선택 + CV 학습 일관성

[표준화 공간 방향벡터 (전략 C)]
  defect_dir_sc = Xl_sc_sel[mask_d].mean() - Xl_sc_sel[mask_n].mean()
  projC = Xu_sc_sel @ defect_dir_sc_n  ← 25피처 균등 기여

[align_score 이중 지표]
  uniform: 동등 가중치 (비교 기준)
  weighted: RF importance 가중 (실질 중요도 반영)

[전략 B 누수 차단]
  run_cv_B_clean(): 각 fold에서 fold-train만으로 teacher RF → 누수 없는 pseudo 선택

[통계 검정]
  Wilcoxon signed-rank (n=5 fold 쌍별 차이)
  p<0.05 → IMPROVED* / p≥0.05 → trend or ns

[ablation]
  B·C·D 전략 모두 n=[50,100,200,354,500,750,1000,2000]

[Self-training]
  threshold sweep [0.5, 0.6, 0.7, 0.8]
  → threshold 무관하게 실패 시: 구조적 실패 확증 (DR=0.89% 극단 불균형 원인)
```

#### 핵심 실험 설계 제약

- **pseudo-label 생성**: val fold 배제 (B는 fold별 teacher로 완전 차단)
- **SMOTE**: labeled train에만 (pseudo는 SMOTE 후 append)
- **Scaler**: labeled train fold로 fit → val, pseudo 모두 transform
- **평가**: Wilcoxon p<0.05 기준, ROC+PR 양쪽 확인

### 2.2 방법 A — Autoencoder 기반 이상탐지 (조현건, 2순위)

**방법 C 결과 확인 후 진행. Proposal 방법 A = 가이드북 §2.2.1 재현 및 확장**

```python
# 3×2 ablation 구조
Config 1: OC-SVM           ← labeled 양품만, 구현 완료 (ROC 0.8814)
Config 2: AE (labeled만)   ← 가이드북 §2.2.1 재현
Config 3: AE (labeled + EQUIP/PART 필터링 unlabeled) ← 우리 확장

임계값 A: +5σ  (가이드북 방식)
임계값 B: PR 곡선 val fold 운용점 (우리 방식)

# AE 구조
Encoder: FC(25→16→8) + Dropout(0.1) + ReLU
Decoder: FC(8→16→25) + ReLU
Loss: MSELoss, Optimizer: Adam(1e-3), Epochs: 50
```

### 2.3 방법 B — K-Means 거리 기반 이상탐지 (다른 팀원 담당)

```
개선 핵심: unlabeled_data.csv에서 EQUIP_NAME/PART_NAME으로 필터링 후 K-Means 재학습
1차(ROC 0.47, 전체 795K) vs 2차(필터링 후) 비교 → 도메인 불일치 가설 직접 검증
```

---

## 3. Proposal (✅ 5/17 제출 완료)

- 제출 파일: `docs/Proposal_NOVA50101.md` → PDF 변환 후 Blackboard 업로드 완료
- 팀장(전정은, 20268528)이 제출
- 최종 팀원: 전정은(팀장, Phase 1·3) / 박상은(Phase 4·5) / 조현건(Phase 2·5·6·7)
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

| 강의 | 알고리즘 | 상태 | 위치 |
|---|---|---|---|
| L4-6 | LR(L1/L2), LDA, QDA | ✅ | Phase 2~3 |
| L7-8 | SVM(linear/RBF), **One-Class SVM** | ✅ | Phase 3 / Phase 7(방법 A 비교군) |
| L8-10 | RF, AdaBoost, GBM, Stacking | ✅ | Phase 4~5 |
| L9-10 | **K-Means** (이상탐지), **DBSCAN** (방법 C) | ✅ / ⏳ | Phase 7(방법 B·C) |
| L9-10 | GMM (Future) | Future | §5 Future Work |
| L11 | PCA, **UMAP** (방법 C 시각화) | ✅ / ⏳ | Phase 4-A / 방법 C |
| L11-14 | MLP+Dropout, **Denoising AE** (방법 A) | ✅ / ⏳ | Phase 4 / Phase 7(방법 A) |
| L15-16 | 1D-CNN | ✅ | Phase 5 |

---

## 7. 산출물 마스터 체크리스트

### 5/17 마감 ✅ 완료
- [x] `docs/Proposal_NOVA50101.md` → PDF → Blackboard 제출 (팀장 전정은)
- [x] 팀원 이름 실명 채우기 (전정은·박상은·조현건)

### 진행 중 (Phase 7)
- [x] 방법 C v1: DBSCAN 기반 (실패 — 기계정지 오염 확인)
- [x] 방법 C v2: 4전략 + run_cv 버그수정 (`notebooks/08_method_c_v2.ipynb`)
- [x] 방법 C v2 심사 비판: 4개 FATAL 비판 수용 + 재설계
- [ ] **방법 C v3 실행 중**: 심사 비판 전면 수용 버전 (`notebooks/08_method_c_v2.ipynb`)
  - [ ] Wilcoxon 검정 포함 최종 결과 확인
  - [ ] B·D 전략 n ablation 결과 확인
  - [ ] ST threshold sweep 결과 확인
- [ ] 방법 A: Denoising AE 구현 (`notebooks/09_autoencoder.ipynb`) — 방법 C 완료 후
- [ ] 방법 B: K-Means 개선 실험 (다른 팀원) — 필터링 후 재학습

### 최종 제출 (학기말)
- [ ] `reports/final_report.pdf` (8~12페이지)
- [ ] `final_presentation.pptx` (16슬라이드, `reports/presentation_outline.md` 참조)
- [ ] `Activity_Appendix.pdf` (팀원별 기여 표)
- [ ] `pytest tests/ -v` 전 통과 상태 유지

---

## 8. 자주 빠지는 함정

1. **데이터 누수** — SMOTE/스케일링은 fold 내부에서 fit. 외부이면 ROC 0.05+ 부풀림
2. **Accuracy 보고** — 불량 0.89%라 의미없음. ROC-AUC + PR-AUC 사용
3. **fold 불일치** — 모든 Phase가 동일 `fold_*.npy` 사용해야 비교 가능
4. **한글 폰트** — `sns.set_theme()` 이후 `setup_korean_font()` 호출
5. **K-Means 도메인** — unlabeled 전체 학습 시 다기계·다제품 혼재 주의
6. **AE val leakage** — labeled val fold 양품을 AE 학습에 포함하지 않기
7. **기계정지 오염** — unlabeled에서 Barrel_Temp=0 & RPM=0 샘플이 44%: 필터 없이는 pseudo-label = 기계정지
8. **스케일러 공간 불일치** — pseudo-label 선택 스케일러와 모델 학습 스케일러가 달라선 안 됨. sc_sel=labeled-fit 통일
9. **align_score raw 공간** — raw 공간에서 계산하면 대형 피처(Clamp_Open ~500)가 지배. 반드시 표준화 공간에서 계산
10. **Teacher RF 누수** — 전략 B teacher는 ALL labeled로 학습 시 val fold 정보 포함 → fold별 teacher로 차단
11. **통계 검정 생략** — 5-fold 결과에서 mean > std만으로 IMPROVED 주장은 과도. Wilcoxon signed-rank 필수
12. **n ablation 부분 수행** — 전략별로 최적 n이 다를 수 있음. 모든 전략에 동일한 n sweep 적용할 것

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

| `notebooks/08_method_c_v2.ipynb` | 방법 C v3 (심사 비판 수용 — 현재 실행 중) |
| `_gen_mc2.py` | 방법 C v3 노트북 생성 스크립트 |

---

*v1: 2026-05-10 | v2: 2026-05-16 | v3: 2026-05-24 (방법 C v2) | v4: 2026-05-24 (심사 비판 수용 v3)*
