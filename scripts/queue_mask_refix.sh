#!/bin/bash
# 마스킹 조건 재실행 (all-NaN 컨텍스트 안전장치 반영).
# 기존 50K/LT 마스킹 결과는 전 기간 품절 시계열에서 붕괴했을 수 있어 전부 갱신.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
MPY=~/miniconda3/envs/moirai/bin/python
LOG=~/tsfm-demand/logs/queue_maskfix.log
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

C2=data/checkpoints/amazon__chronos-2
TX=data/checkpoints/NX-AI__TiRex
TT=data/checkpoints/ibm-granite__granite-timeseries-ttm-r2
MO=data/checkpoints/Salesforce__moirai-2.0-R-small
TF=data/checkpoints/google__timesfm-2.5-200m-pytorch

for DS in frn50k lt; do
  run c2_${DS}_mask   $PY -m src.experiments.run_zeroshot --model chronos2 --model-path $C2 --dataset $DS --mask-stockout --out results/zeroshot/chronos2_${DS}_mask.parquet
  run tirex_${DS}_mask $PY -m src.experiments.run_zeroshot --model tirex --model-path $TX --dataset $DS --mask-stockout --out results/zeroshot/tirex_${DS}_mask.parquet
  run ttm_${DS}_mask   $PY -m src.experiments.run_zeroshot --model ttm --model-path $TT --dataset $DS --mask-stockout --out results/zeroshot/ttm_${DS}_mask.parquet
  run moirai_${DS}_mask $MPY -m src.experiments.run_zeroshot --model moirai2 --model-path $MO --dataset $DS --mask-stockout --out results/zeroshot/moirai2_${DS}_mask.parquet
done
run timesfm_50k_mask $PY -m src.experiments.run_zeroshot --model timesfm --model-path $TF --dataset frn50k --mask-stockout --out results/zeroshot/timesfm_50k_mask.parquet
echo "=== MASKFIX QUEUE DONE $(date) ===" | tee -a $LOG
