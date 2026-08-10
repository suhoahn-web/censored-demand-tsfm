#!/bin/bash
# Moirai-2 zero-shot 4 runs (moirai env, torch 2.4 격리). 소형 모델(11.4M)이라 병행 실행.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/moirai/bin/python
CKPT=data/checkpoints/Salesforce__moirai-2.0-R-small
LOG=~/tsfm-demand/logs/queue_moirai.log
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a $LOG
  $PY -m src.experiments.run_zeroshot "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

run moirai_lt_raw   --model moirai2 --model-path $CKPT --dataset lt --out results/zeroshot/moirai2_lt_raw.parquet
run moirai_lt_mask  --model moirai2 --model-path $CKPT --dataset lt --mask-stockout --out results/zeroshot/moirai2_lt_mask.parquet
run moirai_50k_raw  --model moirai2 --model-path $CKPT --dataset frn50k --out results/zeroshot/moirai2_50k_raw.parquet
run moirai_50k_mask --model moirai2 --model-path $CKPT --dataset frn50k --mask-stockout --out results/zeroshot/moirai2_50k_mask.parquet
run moirai_50krec   --model moirai2 --model-path $CKPT --dataset frn50krec --out results/zeroshot/moirai2_50krec_raw.parquet
echo "=== MOIRAI QUEUE DONE $(date) ===" | tee -a $LOG
