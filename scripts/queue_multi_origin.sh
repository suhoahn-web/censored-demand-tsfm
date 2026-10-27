#!/bin/bash
# P0-6: 다중 forecast origin zero-shot 평가.
#
# 근거: 단일 origin에서 Chronos-2 WAPE가 origin 이동만으로 1.9%p 움직이는데,
# 주요 기법 간 격차는 1.7%p에 불과하다. 따라서 단일 origin 순위는 결론을 지탱하지
# 못한다. 8개 origin(56일 간격, 12개월 = 전 계절)에서 다시 예측한다.
#
# 범위 제한(논문에 명시할 것): 파인튜닝 Chronos-2와 LightGBM은 LT train 패널 전체로
# 학습했으므로 과거 origin의 정답 구간이 학습에 포함된다 → 다중 origin 대상에서 제외.
# zero-shot TSFM은 이 데이터를 전혀 학습하지 않았으므로 origin을 옮겨도 누출이 없다.
# 고전 기법은 origin별 재적합이 필요해 CPU에서 따로 돌린다.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
MPY=~/miniconda3/envs/moirai/bin/python
LOG=~/tsfm-demand/logs/queue_multi_origin.log
mkdir -p logs results/zeroshot
MINCTX=90   # 이력 90일 미만 시계열 제외 (이른 origin에는 미개점 매장-상품이 있다)

run() {
  name=$1; shift
  if [ -f "results/zeroshot/${name}.parquet" ]; then
    echo "=== $name SKIP (exists) ===" | tee -a $LOG; return
  fi
  echo "=== $name start $(date) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/mo_${name}.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

ORIGINS="2024-07-07 2024-09-01 2024-10-27 2024-12-22 2025-02-16 2025-04-13 2025-05-11 2025-06-08"

# 모델을 바깥 루프에 둔다: 빠른 모델(20-40초)을 전 origin에서 먼저 끝내야
# 느린 모델(TimesFM ~30분/origin)을 기다리지 않고 중간 결과를 쓸 수 있다.
for CUT in $ORIGINS; do
  TAG=mo${CUT//-/}
  run chronos2_lt_${TAG}_raw $PY -m src.experiments.run_zeroshot --model chronos2     --model-path data/checkpoints/amazon__chronos-2 --dataset lt     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/chronos2_lt_${TAG}_raw.parquet
  run chronos2_lt_${TAG}_mask $PY -m src.experiments.run_zeroshot --model chronos2     --model-path data/checkpoints/amazon__chronos-2 --dataset lt --mask-stockout     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/chronos2_lt_${TAG}_mask.parquet
done
for CUT in $ORIGINS; do
  TAG=mo${CUT//-/}
  run tirex_lt_${TAG}_raw $PY -m src.experiments.run_zeroshot --model tirex     --model-path data/checkpoints/NX-AI__TiRex --dataset lt     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/tirex_lt_${TAG}_raw.parquet
  run ttm_lt_${TAG}_raw $PY -m src.experiments.run_zeroshot --model ttm     --model-path data/checkpoints/ibm-granite__granite-timeseries-ttm-r2 --dataset lt     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/ttm_lt_${TAG}_raw.parquet
done
for CUT in $ORIGINS; do
  TAG=mo${CUT//-/}
  run moirai2_lt_${TAG}_raw $MPY -m src.experiments.run_zeroshot --model moirai2     --model-path data/checkpoints/Salesforce__moirai-2.0-R-small --dataset lt     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/moirai2_lt_${TAG}_raw.parquet
done
for CUT in $ORIGINS; do
  TAG=mo${CUT//-/}
  run timesfm_lt_${TAG}_raw $PY -m src.experiments.run_zeroshot --model timesfm     --model-path data/checkpoints/google__timesfm-2.5-200m-pytorch --dataset lt     --cutoff-date $CUT --min-context-days $MINCTX     --out results/zeroshot/timesfm_lt_${TAG}_raw.parquet
done

echo "=== MULTI ORIGIN DONE $(date) ===" | tee -a $LOG
