# 인수인계 문서 — TSFM 검열수요 연구 (실행 세션용)

> 작성: 2026-08-08, 설계 세션(Claude Code). 이 문서만 읽으면 이전 대화 없이 실행을 시작할 수 있도록 작성됨.
> 동반 문서: `research-design-tsfm-censored-demand.md` (연구 설계서 v1.1 — 실험 매트릭스·평가지표·저널 전략의 원본), `project-CLAUDE-draft.md` (새 프로젝트 폴더에 넣을 CLAUDE.md 초안)

---

## 1. 프로젝트 한 줄 요약

시계열 파운데이션 모델(TSFM)을 **품절로 검열된 간헐 소매 수요**(FreshRetailNet-50K)에서 zero-shot/LoRA/full-FT로 평가하고, 정확도뿐 아니라 **뉴스벤더 재고 시뮬레이션**으로 의사결정 가치까지 측정하는 논문. 타깃: SCIE — **1순위 ESWA, 리젝 시 하향 캐스케이드(C&IE → EAAI/Neurocomputing → Applied Intelligence/J. of Forecasting → IEEE Access). IJF·IJPE는 심사 기간 문제로 제외** (2026-08-08 사용자 확정, 설계서 7절 참조).

## 2. 사용자 컨텍스트

- 목표: **최대한 빠른 SCIE(급) 논문 게재**. 이공계식 빠른 결과 지향
- 배경: 원래 GEM(창업) 데이터 연구를 탐색하다가, 딥러닝+GPU가 정당화되는 수요예측으로 방향 전환 (설계 세션에서 합의됨)
- GPU 사양: **미확정 — 실행 시작 전 사용자에게 확인할 것** (실험 범위 조정에 필요)

## 3. 확정된 설계 결정과 이유 (변경 시 사용자와 상의)

| 결정 | 이유 |
|---|---|
| 주 데이터 = FreshRetailNet-50K (HF: `Dingdong-Inc/FreshRetailNet-50K`, CC-BY-4.0) | 2025.5 공개 → 모든 TSFM 사전학습 이후 = 유일한 누출-무결 리테일 벤치마크. 품절 라벨 보유 |
| **M5는 zero-shot 평가 금지** | M5가 Chronos 계열 학습 코퍼스에 포함된 것 확정. M5는 (a) 파인튜닝 후 transfer 검증 (b) 누출 효과 정량화에만 사용 |
| 적응 전략 = {zero-shot, LoRA(r=8/16/32), full FT} | Energy and Buildings(2025) 선행이 이 구조로 SCIE 게재 — 성공 템플릿 복제 |
| 검열 처리 2×3 팩토리얼 = {원 판매량, two-stage 복원수요} × 적응 3종 | 단순 벤치마킹 논문과의 novelty 차별점 |
| 재고 평가 = 뉴스벤더 분위수 발주, 비용 시나리오 3종 | EJOR 2024(M5 정확도-재고성과)·Syntetos&Boylan(2006) 계보로 방법론 방어 |
| 통계 검정 = MCB + Friedman–Nemenyi, 시드 3회 | 예측 문헌 표준. 좋은 저널 관문 |
| 지표 = WAPE/MASE/WQL/CRPS/WPE (**MAPE 금지**) | 간헐수요 0-분모 문제. WPE(편향)가 검열 스토리의 핵심 지표 |
| 어떤 결과가 나와도 성립하는 4-시나리오 설계 | 설계서 6절 참조. TSFM이 져도 논문이 됨 |

## 4. 완료된 검증 (재조사 불필요)

- [x] 갭 확인: 검열·간헐 수요 TSFM 평가의 피어리뷰 저널 논문 부재 (2026-08 기준)
- [x] FRN-50K 원 논문 베이스라인: TimesNet WAPE 27.62% / DLinear 29.99%, two-stage로 편향 7.37%→~0 — 재현 목표치
- [x] Chronos 학습 코퍼스에 M5 포함 (확정)
- [x] FRN-50K 라이선스 CC-BY-4.0 (학술 이용 가능, 인용 필수)
- [x] TimesFM-2.5 코퍼스 상세 비공개 → 논문에서 한계로 기술
- [x] 재고 평가 방법론 근거: EJOR 2024 + Syntetos & Boylan 2006 + Kourentzes 계열

## 5. 실행 로드맵 (우선순위 순)

### Phase 0 — 시작 전 (사용자 확인 필요)
1. GPU 사양 확인 → 모델 수 조정 (VRAM 16GB 이하면 TTM·Chronos-Bolt 등 경량부터)
2. 새 프로젝트 폴더 생성 + `project-CLAUDE-draft.md`를 `CLAUDE.md`로 복사

### Phase 1 — 데이터·베이스라인 (주 1–4)
3. FRN-50K 로드, EDA, ADI×CV² 사분면 분류 (Syntetos-Boylan-Croston: ADI 경계 1.32, CV² 경계 0.49)
4. rolling origin 분할 3개 (테스트 마지막 14일, 매장 단위 그룹 분할)
5. 전통(Croston/SBA/TSB/ETS/SNaive) + LightGBM 베이스라인
6. **FRN-50K 원 논문 재현** (GitHub: `Dingdong-Inc/frn-50k-baseline`) — TimesNet WAPE ~27.6% 재현 실패 시 즉시 원인 규명 (프로젝트 최대 리스크)
7. **잔여 감사**: GIFT-Eval 구성 목록에서 M5·리테일 데이터 대조 → 설계서 부록 A 표 완성 (Chronos-2/Moirai-2/TTM-R2의 "대조 필요" 칸)

### Phase 2 — TSFM 실험 (주 5–10)
8. zero-shot 5종 (Chronos-2, Chronos-Bolt, TimesFM-2.5, Moirai-2, TTM-R2) — **모델 체크포인트 해시 기록 필수**
9. M5 누출 정량화 실험
10. LoRA/full FT + few-shot(10/30/100%) 곡선
11. two-stage 복원 팩토리얼

### Phase 3 — 재고·분석·집필 (주 11–16)
12. 뉴스벤더 시뮬레이션 (비용 시나리오 3종, 정확도-비용 순위 Kendall τ)
13. MCB/Friedman–Nemenyi, ablation (공변량/해상도/컨텍스트 길이/사분면/프로모션 구간)
14. 초고 → **ESWA 투고** (데스크 리젝 시 하향 캐스케이드 — 설계서 7절). 서론·Figure 1은 응용 문제(품절·폐기)로 시작할 것 (ESWA 데스크 리젝 회피)

## 6. 기술 스택 제안

- Python 3.11+, PyTorch, `datasets`(HF), `chronos-forecasting`, `timesfm`, `uni2ts`(Moirai), `granite-tsfm`(TTM), `peft`(LoRA), `statsforecast`(전통 베이스라인), `lightgbm`, `scikit-posthocs`(Nemenyi)
- 실험 추적: 결과를 CSV/parquet + 설정 YAML로 버전 관리 (재현 패키지가 논문 제출물)
- 난수 시드 {0, 1, 2} 고정

## 7. 리스크 알림 (설계서 9절의 실행 관점 요약)

- **최대 리스크**: FRN-50K 원 논문 재현 실패 → Phase 1에서 조기 검증하도록 순서 배치함
- hourly 5만 시계열은 전처리·추론 비용 큼 → 파일럿은 대표 표본(사분면별 층화 5천 개)으로 파이프라인 검증 후 전체 확장
- TSFM 라이브러리 API가 빠르게 변함 → 설치 시점에 버전 고정(`requirements.txt` lock)
- FRN-50K가 향후 모델 사전학습에 포함될 수 있음 → 사용한 체크포인트 버전·해시를 실험 로그에 남길 것

## 8. 핵심 레퍼런스 (논문 서론·방법론에서 인용할 것들)

| 용도 | 문헌 |
|---|---|
| 데이터 원 논문 | FreshRetailNet-50K, arXiv:2505.16319 |
| 파인튜닝 템플릿 선행 | Energy and Buildings (2025), S0378778825011764 |
| decision-focused 선행 | Energy and AI (2025), S2666546825000655 |
| 정확도-재고 관계 | EJOR (2024), S0377221724009755 |
| 재고 평가 프로토콜 | Syntetos & Boylan (2006); Kourentzes (2013); Kourentzes et al. (2020) |
| 누출 비판 | Rethinking Evaluation, arXiv:2510.13654 |
| **ESWA 게재 선례 (검열수요)** | **Sousa, Loureiro & Miguéis (2025), ESWA vol.259, S0957417424021808** — 패션 리테일 검열수요 two-stage(EM 복원→ML 예측). 우리와 동일 문제구조가 ESWA에 게재된 직접 선례. 서론·저널 적합성 방어의 핵심 인용 |
| 모델 원 논문 | Chronos arXiv:2403.07815; Moirai-2 arXiv:2511.11698; Lag-Llama arXiv:2310.08278 |
| 벤치마크 | It's TIME arXiv:2602.12147; GIFT-Eval |
| 간헐수요 고전 | Croston (1972); Syntetos & Boylan (2005); Teunter et al. (2011) |
