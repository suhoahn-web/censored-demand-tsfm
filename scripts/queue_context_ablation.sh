#!/bin/bash
# 이력 길이 ablation: LT(2년)를 90/180/365일로 절단 → 50K(90일)와의 차이가
# "데이터셋 차이"가 아니라 "이력 길이" 때문인지 분리 검증.
# 같은 시계열·같은 평가구간이므로 통제된 비교.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
MPY=~/miniconda3/envs/moirai/bin/python
LOG=~/tsfm-demand/logs/queue_ctxabl.log
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

for D in 90 180 365; do
  run c2_lt_ctx${D} $PY -m src.experiments.run_zeroshot --model chronos2 \
    --model-path data/checkpoints/amazon__chronos-2 --dataset lt \
    --max-context-days $D --out results/zeroshot/ctxabl_chronos2_lt_${D}d.parquet
  run tirex_lt_ctx${D} $PY -m src.experiments.run_zeroshot --model tirex \
    --model-path data/checkpoints/NX-AI__TiRex --dataset lt \
    --max-context-days $D --out results/zeroshot/ctxabl_tirex_lt_${D}d.parquet
  run moirai_lt_ctx${D} $MPY -m src.experiments.run_zeroshot --model moirai2 \
    --model-path data/checkpoints/Salesforce__moirai-2.0-R-small --dataset lt \
    --max-context-days $D --out results/zeroshot/ctxabl_moirai2_lt_${D}d.parquet
done
echo "=== CTX ABLATION DONE $(date) ===" | tee -a $LOG
