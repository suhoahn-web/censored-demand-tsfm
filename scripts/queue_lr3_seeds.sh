#!/bin/bash
# LoRA r16 lr=1e-3 시드 1,2 보강 (seed 0은 v2 큐에서 완료)
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
mkdir -p logs results/finetune_v2

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue_lr3.log
  $PY -m src.experiments.run_finetune_chronos2 "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue_lr3.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue_lr3.log
}

for seed in 1 2; do
  run v2_lora_r16_lr1e-3_s${seed} --mode lora --rank 16 --seed $seed \
      --learning-rate 1e-3 --out results/finetune_v2/lora_r16_lr1e-3_s${seed}
done
echo "=== LR3 QUEUE DONE $(date) ===" | tee -a logs/queue_lr3.log
