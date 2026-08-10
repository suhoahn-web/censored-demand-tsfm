#!/bin/bash
# 주말 파인튜닝 큐 (lab06, nohup). 실험 2: Chronos-2 적응 전략.
# LoRA r={8,16,32}×seed{0,1,2} + full×seed{0,1,2} + 마스킹 LoRA r16×seed{0,1,2} = 15 runs.
# 실패해도 다음으로 진행. 로그: ~/tsfm-demand/logs/
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
mkdir -p logs results/finetune

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue_ft.log
  $PY -m src.experiments.run_finetune_chronos2 "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue_ft.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue_ft.log
}

for seed in 0 1 2; do
  for r in 8 16 32; do
    run "c2_lora_r${r}_s${seed}" --mode lora --rank $r --seed $seed \
        --out results/finetune/c2_lora_r${r}_s${seed}
  done
done

for seed in 0 1 2; do
  run "c2_full_s${seed}" --mode full --seed $seed \
      --out results/finetune/c2_full_s${seed}
done

for seed in 0 1 2; do
  run "c2_lora_r16_mask_s${seed}" --mode lora --rank 16 --seed $seed --mask-stockout \
      --out results/finetune/c2_lora_r16_mask_s${seed}
done

echo "=== FT QUEUE DONE $(date) ===" | tee -a logs/queue_ft.log
