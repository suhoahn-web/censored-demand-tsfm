#!/bin/bash
# TiRex zero-shot 4 runs (LT/50K × raw/mask). TFT 학습과 병행 (GPU 여유 충분).
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
CKPT=data/checkpoints/NX-AI__TiRex
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue_tirex.log
  $PY -m src.experiments.run_zeroshot "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue_tirex.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue_tirex.log
}

run tirex_lt_raw   --model tirex --model-path $CKPT --dataset lt --out results/zeroshot/tirex_lt_raw.parquet
run tirex_lt_mask  --model tirex --model-path $CKPT --dataset lt --mask-stockout --out results/zeroshot/tirex_lt_mask.parquet
run tirex_50k_raw  --model tirex --model-path $CKPT --dataset frn50k --out results/zeroshot/tirex_50k_raw.parquet
run tirex_50k_mask --model tirex --model-path $CKPT --dataset frn50k --mask-stockout --out results/zeroshot/tirex_50k_mask.parquet
echo "=== TIREX QUEUE DONE $(date) ===" | tee -a logs/queue_tirex.log
