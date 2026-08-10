# 연구 설계서 (v1.1)

> v1.1 변경: 사전학습 코퍼스 감사 결과 반영(부록 A), FRN-50K 라이선스 확정(CC-BY-4.0), 10절 확인 사항 상태 갱신

## 제목(안)

**"Time Series Foundation Models for Censored and Intermittent Retail Demand: A Rigorous Evaluation of Zero-Shot, Fine-Tuning, and Inventory Implications"**

(국문: 검열·간헐 소매 수요에 대한 시계열 파운데이션 모델의 엄밀한 평가 — Zero-shot, 파인튜닝, 재고관리 관점)

---

## 1. 연구 질문 (RQ)

| # | 연구 질문 | 검증 방법 |
|---|---|---|
| **RQ1** | TSFM은 품절로 검열된(censored) 간헐 수요를 zero-shot으로 얼마나 정확히 예측하는가? 전통 간헐수요 기법(Croston/SBA/TSB), 트리 기반 ML(LightGBM), 최신 딥러닝(TimesNet, PatchTST, DLinear) 대비 우위가 있는가? | 실험 1 |
| **RQ2** | LoRA 파인튜닝은 TSFM의 검열·간헐 수요 성능을 얼마나 개선하는가? full fine-tuning 대비 비용-성능 트레이드오프는? | 실험 2 |
| **RQ3** | 잠재수요 복원(latent demand recovery)을 선행한 뒤 TSFM을 적용하면(two-stage) 원 판매량 직접 학습 대비 성능·편향이 개선되는가? | 실험 3 |
| **RQ4** | 예측 정확도의 우위가 재고 의사결정 성과(뉴스벤더 비용, 서비스수준, 폐기율)의 우위로 이어지는가? | 실험 4 |

**핵심 논지**: "TSFM 평가는 (i) 사전학습 누출이 통제된 데이터에서, (ii) 검열이라는 현실 조건 하에, (iii) 하류 의사결정 가치로 이루어져야 한다" — 평가 프레임 자체가 기여.

---

## 2. 기여 주장 (Contributions)

1. **최초의 검열·간헐 수요 조건 TSFM 체계적 평가.** FreshRetailNet-50K(2025.5 공개)는 모든 주요 TSFM의 사전학습 이후 공개된 품절 라벨링 데이터로, 누출 없는 깨끗한 테스트베드. 원 논문 베이스라인(TimesNet WAPE 27.62%)에 TSFM 계열을 최초로 추가.
2. **검열 처리 × 적응 전략의 팩토리얼 설계.** {원 판매량, 복원 수요} × {zero-shot, LoRA, full FT}의 교차 실험으로 "무엇이 성능을 만드는가"를 분리 식별. 단순 벤치마킹 논문과의 차별점.
3. **재고 의사결정 기반 평가.** 뉴스벤더 시뮬레이션으로 정확도-재고성과의 비선형 관계(EJOR 2024 선행)를 TSFM 맥락에서 최초 검증. 특히 신선식품 특유의 비대칭 비용(폐기 vs 품절) 구조에서.
4. **사전학습 누출의 명시적 통제와 정량화.** M5(사전학습 포함) vs FRN-50K(미포함) 성능 격차로 누출 효과를 실증 — 평가 방법론 문헌(Rethinking Evaluation, 2025)에 실증 근거 추가.

---

## 3. 선행연구 지형과 갭 (검증 완료)

### 3.1 TSFM 원 모델
- Chronos (Amazon, 2024, arXiv:2403.07815) — 토큰화 + T5. **주의: M5가 사전학습/보조학습 코퍼스에 포함됨 (검증됨)**
- Chronos-2 (2025.10) — 공변량 지원 시작
- TimesFM-2.5 (Google), Moirai-2 (Salesforce), TTM (IBM), Lag-Llama
- 벤치마크: It's TIME (2026, 12개 TSFM × 50 데이터셋), GIFT-Eval

### 3.2 도메인 응용 (SCIE 게재 선례 = 우리의 템플릿)
- **Energy and Buildings (2025)**: 건물 에너지에서 TSFM zero-shot은 부족하나 LoRA로 TFT 능가 → *같은 구조를 리테일에서 아무도 안 함*
- **Energy and AI (2025)**: decision-focused fine-tuning → *재고관리 버전 부재*
- 의료(바이탈사인 LoRA, arXiv:2405.10216), 금융(실현변동성) 등 — **리테일 수요예측 응용의 피어리뷰 저널 논문 공백 (검증됨)**

### 3.3 검열수요 문헌
- FreshRetailNet-50K 원 논문(arXiv:2505.16319): two-stage(복원→예측)로 편향 7.37%→~0. 베이스라인은 TimesNet/DLinear/GPVAE/CSDI — **TSFM 미포함 (갭 확정)**
- 고전: Croston(1972), SBA(Syntetos-Boylan 2005), TSB(Teunter et al. 2011)

### 3.4 예측-재고 연결 문헌
- **EJOR (2024)** "Forecast accuracy and inventory performance: M5 data" — 정확도와 재고성과의 비선형 관계
- Syntetos & Boylan (2006) 재고량-서비스수준 동시 비교, Kourentzes (2013), Kourentzes et al. (2020) 재고 목적 예측 최적화
- 간헐수요 조합예측의 재고 함의 (arXiv:2204.08283)

### 3.5 평가 방법론 문헌
- Rethinking Evaluation / 정보 누출 (arXiv:2510.13654) — Monash 계열 데이터의 사전학습-평가 중복 문제 제기. **우리 설계의 정당화 근거**

---

## 4. 데이터 설계

### 4.1 주 데이터: FreshRetailNet-50K
- 출처: Hugging Face `Dingdong-Inc/FreshRetailNet-50K` (무료, 베이스라인 코드 공개)
- 규모: 898개 매장 × 863개 신선 SKU = 50,000개 시계열, 시간(hourly) 단위, 품절 라벨 + 프로모션·강수·기온 공변량
- 역할: **zero-shot 및 파인튜닝의 주 평가 대상** (모든 TSFM 사전학습 이후 공개 → 누출 없음)
- 집계 수준: hourly → daily 집계본과 hourly 원본 두 해상도 모두 실험 (리뷰어 방어)

### 4.2 보조 데이터: M5 (Kaggle)
- 역할 재정의: zero-shot 평가용이 **아니라** —
  - (a) 파인튜닝 후 일반화(cross-dataset transfer) 검증
  - (b) **누출 효과 정량화**: "사전학습에 포함된 데이터(M5)에서의 zero-shot 성능 vs 미포함 데이터(FRN-50K)" 격차 측정 → 기여 4
- M5에는 품절 라벨이 없으므로 검열 실험에서는 제외하고, 간헐성 실험에만 사용

### 4.3 데이터 분할
- 시간 기반 분할 (rolling origin): 학습 → 검증 → 테스트를 시간순으로. 테스트 구간 최소 28일(M5 관례) / FRN-50K는 마지막 14일
- 파인튜닝 시 매장 단위 그룹 분할로 매장 간 누출 차단 (같은 매장의 과거→미래만 허용)
- Rolling origin evaluation 3개 이상의 origin으로 반복 → 결과 안정성 확보

### 4.4 간헐성 세분화 (사전 등록식 서브그룹)
- Syntetos-Boylan-Croston 분류: ADI(평균수요간격) × CV²(수요크기 변동)로 4사분면(smooth / erratic / intermittent / lumpy) 사전 분류
- 모든 결과를 사분면별로 보고 → "TSFM은 어떤 수요 패턴에서 강한가"라는 세부 발견 확보

---

## 5. 실험 설계

### 5.1 모델군 (4계열, 총 12~14개)

| 계열 | 모델 | 근거 |
|---|---|---|
| TSFM | Chronos-2, Chronos-Bolt, TimesFM-2.5, Moirai-2, TTM-R2 | It's TIME 상위권 + 아키텍처 다양성(인코더-디코더/디코더/MLP) |
| 딥러닝 (지역학습) | TimesNet, PatchTST, DLinear | FRN-50K 원 논문 SOTA 재현 + 비교 연속성 |
| 트리 ML | LightGBM (공변량 포함) | M5 우승 계열, 실무 표준 |
| 전통 통계 | Croston, SBA, TSB, ETS, Seasonal Naïve | 간헐수요 표준 + 최소 기준선 |

### 5.2 실험 매트릭스

**실험 1 (RQ1): Zero-shot 평가**
- FRN-50K에서 TSFM 5종 zero-shot vs 지역학습 모델 전체
- 두 가지 입력 조건: 관측 판매량 그대로 / 품절 마스킹(품절 시간대 결측 처리)

**실험 2 (RQ2): 적응 전략**
- {zero-shot, LoRA(r=8,16,32), full FT} × TSFM
- 보고: 성능 + GPU-시간 + 학습 파라미터 수 → 비용-성능 파레토 곡선
- few-shot 조건(학습 데이터 10%/30%/100%)으로 데이터 효율성 곡선

**실험 3 (RQ3): 검열 처리 팩토리얼**
- 2×3 설계: {원 판매량, two-stage 복원 수요(FRN-50K 원 논문 방법 재현)} × {zero-shot, LoRA, full FT}
- 종속변수: 정확도 + **체계적 편향(WPE)** — 검열 무시 시 과소예측 편향이 핵심 스토리

**실험 4 (RQ4): 재고 시뮬레이션**
- 뉴스벤더 프레임: 각 모델의 분위수 예측(q = CR = cu/(cu+co))으로 발주량 결정
- 비용 시나리오 3개: 폐기비용 우세(신선식품 현실), 대칭, 품절비용 우세
- 보고 지표: 총비용, 달성 서비스수준(α, β), 폐기율, 품절률 + 재고량-서비스수준 트레이드오프 곡선 (Syntetos-Boylan 2006 방식)
- 부가 분석: 정확도 순위 vs 재고비용 순위의 순위상관(Kendall τ) → "정확도가 곧 가치인가" 질문에 직접 답

### 5.3 평가지표
- 점 정확도: WAPE, MASE (간헐수요에서 MAPE 금지 — 0 분모 문제)
- 확률 정확도: WQL(가중 분위수 손실), CRPS
- 편향: WPE (검열 스토리의 핵심)
- 재고: 5.2 실험 4 참조

### 5.4 통계적 엄밀성 (리뷰어 방어의 핵심)
- **MCB (Multiple Comparisons with the Best)** + Friedman–Nemenyi 검정: 예측 대회 문헌 표준 (Koning et al. 2005, M5 분석 논문들 관례)
- 시계열별 순위 기반 → 5만 개 시계열이므로 검정력 충분
- 딥러닝/파인튜닝은 시드 3개 반복, 평균±표준편차 보고
- 모든 하이퍼파라미터 탐색 범위·선택 기준 명시, 코드·설정 GitHub 공개 (재현성)

### 5.5 Ablation & 강건성
1. 공변량 유무 (Chronos-2·Moirai-2의 공변량 처리 능력 — 에너지 문헌에서 Moirai가 공변량을 못 쓴다는 선행 발견의 리테일 검증)
2. 시간 해상도 (hourly vs daily)
3. 컨텍스트 길이 민감도
4. 사분면(ADI×CV²)별 성능 분해
5. 프로모션 구간 vs 평시 구간 분해

---

## 6. 예상 결과 시나리오 — 어떤 결과가 나와도 논문이 성립하는가

| 시나리오 | 결과 | 논문 스토리 |
|---|---|---|
| S1 | TSFM(LoRA)이 전면 우세 | "파인튜닝된 TSFM은 간헐·검열 수요의 새 SOTA" — 실무 처방 명확 |
| S2 | 트리/지역 딥러닝이 우세 | "TSFM의 한계 실증: 사전학습 분포와 다른 검열·간헐 패턴엔 전이 실패" — 에너지 문헌과 대조, 부정적 결과도 IJF 스타일 기여 |
| S3 | 정확도 우위 ≠ 재고비용 우위 | "TSFM 평가의 정확도 편중 비판" — EJOR 2024 계보의 방법론 기여 |
| S4 | 복원 수요 학습이 결정적 | "데이터 전처리가 모델 선택보다 중요" — two-stage 프레임 강화 |

어느 시나리오든 4개 기여 중 3개 이상이 살아남음 → 결과 의존성 낮은 안전한 설계.

---

## 7. 타깃 저널 (SCIE, 우선순위)

| 순위 | 저널 | IF(참고) | 적합성 | 비고 |
|---|---|---|---|---|
| 1 | **International Journal of Forecasting** | ~7 | 예측 방법론+재고 함의 논문의 본진. M5 특집 계보 | 심사 김. 방법론 엄밀성 요구 최고 |
| 2 | **International Journal of Production Economics** | ~9 | 재고 시뮬레이션 비중 키우면 최적 | RQ4를 주 기여로 재프레임 필요 |
| 3 | **European Journal of Operational Research** | ~6 | EJOR 2024 정확도-재고 논문의 직접 후속 포지셔닝 가능 | |
| 4 | **Expert Systems with Applications** | ~7.5 | 응용 AI 관대, 심사 상대적으로 빠름 | 속도 우선 시 1순위로 승격 |
| 백업 | Applied Soft Computing, Neurocomputing, IEEE Access | | | |

**전략**: IJF 투고 → 리젝 시 리뷰 반영해 IJPE/ESWA. 커버레터에서 "M5 누출 통제 + 재고 기반 평가"를 전면에.

---

## 8. 실행 계획 (T0 = 시작일)

| 주차 | 작업 |
|---|---|
| 1–2 | 데이터 파이프라인(FRN-50K 로드, ADI/CV² 분류, rolling origin 분할), 전통·트리 베이스라인 완성 |
| 3–4 | FRN-50K 원 논문 베이스라인(TimesNet 등) 재현 — **재현 실패 시 조기 발견 필수** |
| 5–6 | TSFM zero-shot 전체 (실험 1) + M5 누출 정량화 |
| 7–9 | LoRA/full FT (실험 2) — GPU 집중 구간 |
| 10 | two-stage 복원 팩토리얼 (실험 3) |
| 11 | 재고 시뮬레이션 (실험 4) + 통계 검정 |
| 12–14 | Ablation, 그림·표, 초고 작성 |
| 15–16 | 내부 리뷰, 재현 패키지 정리, 투고 |

**컴퓨팅 예산 추정**: LoRA 파인튜닝 중심이므로 A100 1장 기준 총 100–200 GPU-시간 내외. RTX 4090 1장으로도 가능(TTM·Chronos-Bolt 등 경량 모델부터).

---

## 9. 위험 요소와 방어

| 리뷰어 예상 공격 | 방어 |
|---|---|
| "단일 데이터셋 일반화 한계" | M5 transfer 실험 + (여유 시) Favorita 추가. FRN-50K 자체가 898매장×18도시로 내부 이질성 큼 |
| "M5가 사전학습에 포함됐는데 왜 썼나" | 역으로 설계 포인트: 누출 정량화 목적으로만 사용, zero-shot 주장에서 배제 — 본문에 명시 |
| "FRN-50K도 곧 사전학습에 들어갈 것" | 실험 시점의 모델 버전·체크포인트 해시 고정 명시 |
| "LoRA 설정 튜닝 불공정" | 모든 계열에 동일 검증 예산(HPO 시도 횟수) 배정, 부록에 전체 탐색 기록 |
| "중국 신선식품 특수성" | 한계 절에 명시 + 사분면 분해로 패턴 수준의 일반화 주장 |
| "novelty 부족 (그냥 벤치마킹)" | 팩토리얼 설계(검열×적응)와 재고 평가 프레임이 단순 벤치마킹과의 차별점 — 서론에서 선제 방어 |

---

## 10. 확인 사항 상태 (2026-08-08 기준)

1. ~~각 TSFM의 사전학습 코퍼스 문서 조사~~ → **1차 완료, 부록 A 참조.** 잔여: GIFT-Eval 구성 목록에서 M5·리테일 데이터 직접 대조 (실행 세션 과제)
2. ~~FRN-50K 라이선스 확인~~ → **완료: CC-BY-4.0.** 학술 이용 가능, 논문에 원 논문(arXiv:2505.16319) 인용 필수
3. FRN-50K 원 논문의 전처리·분할 프로토콜 코드 검토 → **실행 세션 과제** (GitHub: Dingdong-Inc/frn-50k-baseline)
4. 사용 가능한 GPU 사양 확정 → **사용자 확인 필요**

---

## 부록 A. TSFM 사전학습 코퍼스 감사 (1차, 논문 부록 표의 초안)

| 모델 | 사전학습 코퍼스 | 리테일/판매 데이터 | M5 포함 | FRN-50K 포함 | 판정 |
|---|---|---|---|---|---|
| Chronos / Chronos-Bolt | 공개 시계열 코퍼스 + 보조 학습셋 | 포함 (retail 도메인 명시) | **포함 확정** (M3·Tourism과 함께 보조 학습셋으로 샘플링) | 불가능 (2025.5 공개 > 학습 시점) | M5 zero-shot 평가 **금지** |
| Chronos-2 | GIFT-Pretrain(230B pts) + Chronos-Dataset | 포함 추정 | GIFT-Eval 구성 대조 필요 | 불가능 | 동일하게 보수적 처리 |
| TimesFM-2.5 | Google Trends, Wikipedia pageviews, 합성 (~100B pts), 상세 비공개 | **미확인** (비공개) | 미확인 | 불가능 | 비공개성 자체를 논문 한계·논점으로 기술 |
| Moirai-2 | GIFT-Pretrain + GIFT-Eval train split + Chronos-Mixup + KernelSynth + Salesforce 텔레메트리 (36M series / ~295B pts) | sales 도메인 포함 | GIFT-Eval train split 대조 필요 | 불가능 | 대조 후 확정 |
| TTM-R2 | Monash 계열 등 | 대조 필요 | 대조 필요 | 불가능 | 실행 세션 과제 |

**공통 결론**: FRN-50K는 모든 대상 모델의 학습 시점 이후 공개 → **유일하게 누출-무결한(leakage-free) 평가 대상**. 이 표의 "대조 필요" 항목 확정이 실행 세션 최우선 과제 중 하나이며, 완성된 표는 논문 부록 표 1이 된다.
