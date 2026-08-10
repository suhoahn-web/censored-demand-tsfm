# CLAUDE.md — TSFM Censored Demand Research

> 이 파일을 새 프로젝트 폴더 루트에 `CLAUDE.md`로 복사할 것.
> 상세 설계·인수인계: `docs/research-design-tsfm-censored-demand.md`, `docs/HANDOVER-tsfm-research.md` (두 파일을 `docs/`로 함께 복사)

## 프로젝트 목적

TSFM(Chronos-2, TimesFM-2.5, Moirai-2, TTM-R2 등)을 검열·간헐 소매 수요(FreshRetailNet-50K)에서 평가하는 SCIE 논문 연구. 실험 코드는 전부 재현 가능해야 함 — 재현 패키지가 논문 제출물의 일부다.

## 절대 규칙 (설계 세션 확정, 2026-08-08 문헌 감사로 개정. 위반 시 논문이 무너짐)

1. **누출 통제**: M5를 사전학습에 쓴 모델은 Chronos가 아니라 **Moirai 1.x**(LOTSA에 m5·favorita 포함)다. Moirai-1.x로 M5/FRN zero-shot 성능을 "일반화 능력"으로 주장하지 말 것 — Moirai-1.1은 누출 정량화 실험 전용. 누출-무결 주장의 근거는 "공개 시점"이 아니라 **코퍼스 문서 감사**(GiftEvalPretrain 등에 리테일 판매 데이터 부재)로 할 것.
2. **MAPE 사용 금지** (간헐수요 0-분모). 지표는 WAPE, MASE, WQL, CRPS, WPE.
3. 모든 TSFM 추론·학습 시 **체크포인트 버전/해시를 실험 로그에 기록**할 것.
4. 데이터 분할은 시간 기반(rolling origin 3개) + 매장 단위 그룹 분할. 미래→과거 누출 금지.
5. 딥러닝/파인튜닝 실험은 시드 {0,1,2} 3회 반복, 평균±표준편차 보고.
6. 설계 변경(모델 추가·제거, 지표 변경, 분할 방식 변경)은 반드시 사용자와 상의 후 `docs/decisions.md`에 기록.

## 데이터

- 주: `Dingdong-Inc/FreshRetailNet-LT` (Hugging Face) — v3, 22,939 hourly 시계열 × 2년(2023-06~2025-07), 품절 라벨, 불균형 패널, 공식 eval = 10K 시계열 × 마지막 7일
- 보조 1: `Dingdong-Inc/FreshRetailNet-50K` (v1/v2, CC-BY-4.0) — 강건성 검증용
- 보조 2: M5 (Kaggle `m5-forecasting-accuracy`) — transfer/누출 실험 전용
- 원본 데이터는 `data/raw/`에 두고 절대 수정하지 않음. 전처리 산출물은 `data/processed/` + 생성 스크립트 버전 기록

## 권장 구조

```
├── CLAUDE.md
├── docs/                  # 설계서, 인수인계, decisions.md
├── data/{raw,processed}/
├── src/
│   ├── data/              # 로드, ADI×CV² 분류, 분할
│   ├── models/            # 베이스라인·TSFM 래퍼 (공통 인터페이스: fit/predict_quantiles)
│   ├── eval/              # 지표, MCB/Nemenyi, 뉴스벤더 시뮬레이션
│   └── experiments/       # 실험별 진입 스크립트 (YAML 설정 구동)
├── configs/               # 실험 설정 YAML (실험 1개 = YAML 1개)
├── results/               # parquet/CSV + 실행 메타데이터(시드, 해시, 버전)
└── requirements.txt       # 버전 고정 필수
```

## 실행 순서

`docs/HANDOVER-tsfm-research.md` 5절의 Phase 0→3 순서를 따를 것. 특히 **Phase 1의 원 논문 재현(TimesNet WAPE ~27.6%)이 최우선** — 실패 시 이후 작업 중단하고 원인 규명.

## 스타일

- 실험 스크립트는 결정론적으로: 설정은 YAML로 주입, 하드코딩 금지
- 파일럿은 층화 표본 5천 시계열로 파이프라인 검증 후 전체 5만 확장
- GPU 메모리 부족 시 경량 모델(TTM, Chronos-Bolt)부터
