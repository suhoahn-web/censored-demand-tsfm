#!/bin/bash
# FT 재실행 v2 (시드 수정 + LoRA lr 1e-4). 결과: results/finetune_v2/
# lr sweep 2 + LoRA 그리드 9 + full 3 + 마스킹 LoRA 3 = 17 runs.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
mkdir -p logs results/finetune_v2

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue_ft2.log
  $PY -m src.experiments.run_finetune_chronos2 "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue_ft2.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue_ft2.log
}

# LoRA lr sweep (부록용; 1e-4는 본 그리드에 포함)
run v2_lora_r16_lr1e-5_s0 --mode lora --rank 16 --seed 0 --learning-rate 1e-5 --out results/finetune_v2/lora_r16_lr1e-5_s0
run v2_lora_r16_lr1e-3_s0 --mode lora --rank 16 --seed 0 --learning-rate 1e-3 --out results/finetune_v2/lora_r16_lr1e-3_s0

# LoRA 본 그리드 (lr 기본 1e-4)
for seed in 0 1 2; do
  for r in 8 16 32; do
    run v2_lora_r${r}_s${seed} --mode lora --rank $r --seed $seed \
        --out results/finetune_v2/lora_r${r}_s${seed}
  done
done

# full FT (chronos 기본 lr)
for seed in 0 1 2; do
  run v2_full_s${seed} --mode full --seed $seed --out results/finetune_v2/full_s${seed}
done

# 품절 마스킹 LoRA r16
for seed in 0 1 2; do
  run v2_lora_r16_mask_s${seed} --mode lora --rank 16 --seed $seed --mask-stockout \
      --out results/finetune_v2/lora_r16_mask_s${seed}
done

echo "=== FT V2 QUEUE DONE $(date) ===" | tee -a logs/queue_ft2.log
