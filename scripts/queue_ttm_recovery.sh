#!/bin/bash
# TTM zero-shot 4 runs (tsfm env) → Stage-1 잠재수요 복원 TimesNet (repro env).
# 순차 실행 (TTM은 몇 분, 복원이 장기).
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
RPY=~/miniconda3/envs/repro/bin/python
CKPT=data/checkpoints/ibm-granite__granite-timeseries-ttm-r2
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue_ttm_rec.log
  "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue_ttm_rec.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue_ttm_rec.log
}

run ttm_lt_raw   $PY -m src.experiments.run_zeroshot --model ttm --model-path $CKPT --dataset lt --out results/zeroshot/ttm_lt_raw.parquet
run ttm_lt_mask  $PY -m src.experiments.run_zeroshot --model ttm --model-path $CKPT --dataset lt --mask-stockout --out results/zeroshot/ttm_lt_mask.parquet
run ttm_50k_raw  $PY -m src.experiments.run_zeroshot --model ttm --model-path $CKPT --dataset frn50k --out results/zeroshot/ttm_50k_raw.parquet
run ttm_50k_mask $PY -m src.experiments.run_zeroshot --model ttm --model-path $CKPT --dataset frn50k --mask-stockout --out results/zeroshot/ttm_50k_mask.parquet

# Stage-1 복원 (원 논문 최고 성능 모델 TimesNet, seed 1024 내장)
cd ~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp
run recovery_timesnet $RPY app.py --model TimesNet --missing_rate 0

echo "=== TTM+RECOVERY QUEUE DONE $(date) ===" | tee -a ~/tsfm-demand/logs/queue_ttm_rec.log
