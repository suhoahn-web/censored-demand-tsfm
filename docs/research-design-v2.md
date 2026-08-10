# 연구 설계서 v2.0 (2026-08-08)

> v1.1 → v2.0: 병렬 문헌 조사 6건(`Literature Review/01~06`) + 핵심 논문 8편 전문 정독(`Literature Review/notes/`)에 기반한 전면 개정.
> 감사 결과는 `Literature Review/00-audit-synthesis.md`, 승인 기록은 `docs/decisions.md` 참조. v1.1 원본은 `연구설계/`에 보존.

## 제목(안)

**"Do Time Series Foundation Models Create Inventory Value under Demand Censoring? A Systematic Evaluation on Fresh Retail Data"**

(국문: 수요 검열 하에서 시계열 파운데이션 모델은 재고 가치를 만드는가 — 신선 소매 데이터 기반 체계적 평가)

---

## 1. 중심 가설과 연구 질문

**중심 가설 (v2.1 — 2026-08-08 실험 결과로 개정)**:

품절 검열은 '관측된 0'(실제로는 수요가 있었으나 팔지 못해 기록된 0)을 만들고, 이를 구분하지 못하는 모든 예측 방법은 체계적 하향 편향(WPE<0)을 갖는다. 이 편향의 재고 성과 영향은 **발주 정책에 조건부**다 — 서비스수준(분위수)이 고정된 정책에서는 편향이 곧 비용이지만, 분위수를 선택할 수 있으면 편향은 상쇄되고 **분포 형태의 품질(WQL)**이 성과를 좌우한다. 따라서 검열 하 예측 평가는 점 정확도만으로 불충분하며, 분포 품질과 하류 의사결정 비용을 함께 보아야 한다.

*부수 가설(확인됨)*: 검열의 해악은 수요 패턴에 따라 다르다 — 간헐·불규칙(intermittent/lumpy) 수요일수록 '가짜 0'의 비중이 커서, 품절 마스킹의 이득이 크고 smooth 수요에서는 오히려 해가 된다.

> v1.1의 "시간 분해(hourly)가 만드는 구조적 0" 축은 fast-track 범위 축소로 미검증 — 본 논문에서는 주장하지 않으며 후속 연구/부록으로 이관 (decisions.md 참조).

| # | 연구 질문 | 실험 |
|---|---|---|
| RQ1 | TSFM 다계열(5종)은 검열된 신선 소매 수요를 zero-shot으로 얼마나 정확히, 얼마나 편향 없이 예측하는가? 전통(Croston/SBA/TSB/ETS/SNaive), 트리(LightGBM-Tweedie), 지역 딥러닝(TFT 등) 대비? | 1 |
| RQ2 | LoRA/full FT는 TSFM의 성능·편향을 얼마나 개선하며, 비용-성능 파레토는 어떤가? | 2 |
| RQ3 | 잠재수요 복원(two-stage)과 시간 집계(hourly→daily)는 각각 '관측된 0'과 '구조적 0'을 제거한다 — 각 처리가 TSFM 성능·편향에 미치는 효과를 분리 식별하면? | 3 |
| RQ4 | 정확도·분포 지표의 우위가 재고 성과(비용·서비스수준·폐기)의 우위로 이어지는가? 편향(WPE)이 정확도(WAPE)보다 재고 성과를 더 잘 설명하는가? | 4 |

## 2. 기여 (문헌 검증 완료 버전)

1. **공개 검열 벤치마크에서 TSFM 다계열의 체계적 평가.** FreshRetailNet-LT v3가 Chronos-2 단일 모델을 평가했으나, 다계열(아키텍처 5종)×적응 3종×검열 처리의 팩토리얼은 부재. 경쟁 프리프린트(SSRN "A Bitter Lesson for Retail Demand Forecasting")는 비공개 데이터·검열 미처리 — 우리는 공개·재현가능·검열 인지.
2. **검열×집계 팩토리얼로 '두 종류의 0' 분리 식별.** ADIDA 계보(Nikolopoulos et al. 2011)의 temporal aggregation 논리를 검열 조건·TSFM으로 확장.
3. **TSFM 분위수의 재고 의사결정 가치 최초 평가** (완전한 공백 확인됨). Theodorou et al. (2025) §7의 future work("판매≠수요 왜곡", "신경망 확률예측")를 정확히 실행. 편향 중심 인과 분석(WPE→재고성과)은 Kourentzes et al. (2020)의 TSFM 확장.
4. **누출 감사 기반 평가 방법론.** Meyer et al. (arXiv:2510.13654)의 권고를 실증 적용: 코퍼스 문서 감사 표 + M5-오염 모델(Moirai-1.1) vs 비오염 모델의 격차 정량화.

## 3. 데이터

| 데이터 | 역할 | 규모 |
|---|---|---|
| **FreshRetailNet-LT** (주) | 전 실험 | 22,939 시계열, 1,057매장×576 SKU, hourly, 2023-06-05~2025-07-13 (770일, 불균형 패널), 공식 eval = 10K 시계열 × 2025-07-14~20 (7일) |
| FreshRetailNet-50K (보조) | 강건성 검증 (균형 패널 90일) | 50K 시계열 × 2024-03-28~06-25 |
| M5 (보조) | transfer 검증 + 누출 정량화 | Moirai-1.1(LOTSA에 M5 포함) vs 비오염 모델 |

- 분할: 공식 7일 eval 존중 + rolling origin 3개(7일 horizon, decisions.md 기록). 매장 단위 그룹 분할로 매장 간 누출 차단.
- ADI×CV² 사분면(경계 1.32/0.49): **방법 선택 규칙이 아니라 성능 층화용 기술 통계** (출처: Syntetos, Boylan & Croston 2005 *JORS*; 정정: Kostenko & Hyndman 2006 *JORS*). 해상도별(hourly/daily) 재분류 — 간헐성은 품목×해상도 속성.

## 4. 모델 (문헌 검증 완료 로스터)

| 계열 | 모델 | 비고 |
|---|---|---|
| TSFM 주력 | Chronos-2 (120M), TimesFM-2.5 (200M), Moirai-2.0-R-small (11.4M, CC-BY-NC 명시), TTM-r2.1, TiRex (35M) | GIFT-Eval 상위 + 아키텍처 다양성. TiRex는 FT 미지원→ZS 전용 |
| 누출 실험 전용 | Moirai-1.1-R | LOTSA에 m5·favorita 포함 — "오염 모델" |
| 공변량 축 | Chronos-2, TimesFM-2.5(XReg), TTM(FT 주입) | Moirai-2는 단변량 회귀로 제외 |
| 지역 딥러닝 | TFT(원 논문 주 베이스라인), TimesNet, DLinear, PatchTST | v3 비교 연속성 |
| 트리 | LightGBM — **Tweedie 손실**, 풀링 cross-learning, 가격·캘린더 공변량, 마지막 4×7일 롤링 CV | M5 우승 레시피 (상위 50팀 전원 LightGBM 사용) |
| 전통 | Croston, **SBA**(Y′=(1−α/2)z′/p′), **TSB**(β≤pα 가이드, 일별이므로 소값 β 격자), ETS, SNaive | 수식·파라미터는 notes/ 참조 |

체크포인트 해시 고정·기록 (절대 규칙 3).

## 5. 실험 매트릭스

- **실험 1 (ZS)**: TSFM 5종 zero-shot vs 지역학습 전체. 입력 2조건: 관측 판매량 그대로 / 품절 마스킹.
- **실험 2 (적응)**: {ZS, LoRA(r=8/16/32), full FT} × 파인튜닝 가능 TSFM. few-shot 곡선(10/30/100%). GPU-시간·파라미터 수 보고.
- **실험 3 (2×2×3 팩토리얼)**: {원 판매량, two-stage 복원수요} × {hourly→daily 집계, daily 직접} × {ZS, LoRA, full FT}. 종속변수: WAPE + **WPE**. 복원은 원 논문 Stage-1 재현(PyPOTS, 영업시간 6~22시, 복원치 음수 클리핑).
- **실험 4 (재고)**: 아래 6절.
- **누출 실험**: M5에서 Moirai-1.1 vs 비오염 TSFM의 ZS 격차 + 코퍼스 감사 표(부록).

## 6. 실험 4: 재고 시뮬레이션 (정독 기반 확정 프로토콜)

- **정책**: 주기적 리뷰 order-up-to + **lost sales** (Theodorou et al. 2025 식 1–7 코드화). S_t = Σ_{t..t+R+L} D̂_w + ss_t. R=1일, L∈{0,1,2}(신선식품). 초기재고 0 → 4주 burn-in 제외.
- **신선식품 확장**: 폐기비용 항 C_W + shelf-life FIFO 소진 (우리 추가 기여).
- **발주 분위수**: (A) 뉴스벤더 CR ∈ {0.5, 0.7, 0.8, 0.9, 0.95} (B) 목표 서비스수준 스윕 → (달성 CSL, 평균재고) trade-off curve (Syntetos & Boylan 2006 계보).
- **비용**: C_Tot = C_H + C_LS + C_O + C_W, **h/b 비율 민감도 스윕 필수** (Theodorou: 비용비에 따라 정확도-비용 순위 역전).
- **분위수 원천 ablation**: TSFM 네이티브 분위수 vs 점예측+경험 오차 분위수 — "분위수 품질의 재고 가치" 분리 식별.
- **핵심 분석**: ① 정확도 순위 vs 재고비용 순위 (Kendall τ, 역전 사례) ② **WPE×재고성과 회귀** (편향 중심 인과 사슬 — 중심 가설 검증) ③ 사분면·비용 시나리오 층화.

## 7. 평가지표 (방어 논리 완결판)

- WAPE(집계 수준), MASE(GIFT-Eval 표준·0-안전), WQL(RPS의 분위수 분해; Gneiting & Ranjan 2011), CRPS(=이산 RPS, Kolassa 2016 권장 proper scoring rule), **WPE**(편향 — 중심 가설 지표).
- MAPE 금지: 실측 0에서 미정의 + (−1)-중앙값 최적 편향 (Kolassa 2016 §2.2 직접 인용).
- **필수 방어 문단**: WAPE/MASE도 MAD 상수배라 중앙값-최적 비판의 사정권(간헐 시계열에서 중앙값=0 퇴화 가능; 정확히는 λ<0.693일 때). 방어: (a) LT 일별 수요는 대부분 smooth (b) 원 논문·GIFT-Eval 비교 가능성 (c) WPE·CRPS/WQL이 취약 축 보완.
- 검열 동기 인용구: "zero inflation could well result from items being out-of-stock" (Kolassa 2016, p.800–801).

## 8. 통계 검정

- MCB (Koning, Franses, Hibon & Stekler 2005 *IJF*) + Friedman–Nemenyi (Demšar 2006 *JMLR*) 병행 — M5는 MCB만 썼고 계층 시계열 독립성 위배를 한계로 자인 → 동일 주의 문구 + 병행 검정이 보강점.
- 시드 {0,1,2}, 평균±표준편차. "최고 예측은 지표에 의존"(M5: WRMSSE 우승자가 MCB 13위) → 다지표+순위 검정 병행 보고.

## 9. 예상 시나리오 (모든 결과가 논문이 됨)

| 시나리오 | 스토리 |
|---|---|
| TSFM(FT) 우세 | 검열·간헐 수요의 새 SOTA — 실무 처방 |
| 지역모델 우세 | Bitter Lesson의 공개 데이터 재현+검열 조건 확장 — 부정적 결과 기여 |
| 정확도≠재고가치 | Theodorou/Kourentzes 계보의 TSFM 확장 — 방법론 기여 |
| 복원/집계가 결정적 | "전처리>모델 선택" — two-stage 프레임 강화 |

## 10. 타깃 저널 (decisions.md 확정)

ESWA(1순위, 응용 문제 전면 배치) → Computers & Industrial Engineering → EAAI/Neurocomputing → Applied Intelligence/Journal of Forecasting → IEEE Access. 인용은 SCI/SCIE급만, MDPI 등 배제.

## 11. 실행 계획 및 리스크

- 일정: 데이터 파이프라인(LT 기준 재작업) → v3 원 논문 재현(TFT, 최대 리스크·최우선) → ZS 전체 → FT → 팩토리얼 → 재고 시뮬 → 집필. **6개월 내 선점 필수** (FRN 피인용 가속: 7편 중 5편이 2026년).
- GPU: lab06 A100 80GB — 전 범위 가능. HF 차단 → 체크포인트 로컬 전송.
- 리스크: v3 재현 실패(조기 검증), LT 불균형 패널 처리(시계열별 길이 상이 — 최소 길이 필터 기준을 사전 등록), TSFM 라이브러리 API 변동(버전 고정), SSRN Bitter Lesson의 저널 게재 시 서론 대조 강화.

## 부록 A. 사전학습 코퍼스 감사 (문헌 검증 완료)

| 모델 | 리테일 데이터 | M5 | FRN | 판정 |
|---|---|---|---|---|
| Chronos/Chronos-2 | Chronos 코퍼스에 M5 없음 (M5는 ZS 벤치마크) | **미포함** | 문서상 없음 | ZS 평가 가능 |
| Moirai-1.1 (LOTSA) | m5·favorita_sales 등 **포함** | **포함** | 없음 | 누출 실험 전용 |
| Moirai-2.0 | GiftEvalPretrain(88 서브셋 전수 확인 — 리테일 판매 전무) + 내부 데이터 | 문서상 없음 | 없음 | 내부 데이터 불확실성 각주 |
| TimesFM-2.5 | 상세 비공개 | 미확인 | 없음(시점상 불가능성+문서) | 비공개성을 한계 기술 |
| TTM-r2.1 / TiRex | Monash 계열 / 문서 확인 | 대조 완료 기준 없음 | 없음 | ZS 평가 가능 |

상세 근거·URL: `Literature Review/01-tsfm-landscape.md`.
