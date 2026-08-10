#!/bin/bash
# TimesFM-2.5 zero-shot 4 runs. 복원(TimesNet)과 GPU 병행.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
CKPT=data/checkpoints/google__timesfm-2.5-200m-pytorch
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a ~/tsfm-demand/logs/queue_timesfm.log
  $PY -m src.experiments.run_zeroshot "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a ~/tsfm-demand/logs/queue_timesfm.log \
    || echo "=== $name FAIL $(date) ===" | tee -a ~/tsfm-demand/logs/queue_timesfm.log
}

run timesfm_lt_raw   --model timesfm --model-path $CKPT --dataset lt --out results/zeroshot/timesfm_lt_raw.parquet
run timesfm_lt_mask  --model timesfm --model-path $CKPT --dataset lt --mask-stockout --out results/zeroshot/timesfm_lt_mask.parquet
run timesfm_50k_raw  --model timesfm --model-path $CKPT --dataset frn50k --out results/zeroshot/timesfm_50k_raw.parquet
run timesfm_50k_mask --model timesfm --model-path $CKPT --dataset frn50k --mask-stockout --out results/zeroshot/timesfm_50k_mask.parquet
echo "=== TIMESFM QUEUE DONE $(date) ===" | tee -a ~/tsfm-demand/logs/queue_timesfm.log
