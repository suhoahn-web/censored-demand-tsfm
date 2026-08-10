#!/bin/bash
# 실험 3 TSFM 셀: 복원수요 패널 생성 → Chronos-2 {ZS, LoRA r16×3seed} on recovered.
# 마지막에 (recovered TFT 체인 종료 대기 후) censored predictTFT 재실행 — 예측 저장 패치 반영.
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
RPY=~/miniconda3/envs/repro/bin/python
CKPT=data/checkpoints/amazon__chronos-2
LOG=~/tsfm-demand/logs/queue_exp3.log
mkdir -p logs results/zeroshot results/finetune_v2

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

run build_rec_panel $PY scripts/build_recovered_panel.py
run c2_rec_zs $PY -m src.experiments.run_zeroshot --model chronos2 --model-path $CKPT --dataset frn50krec --out results/zeroshot/chronos2_50krec_raw.parquet
for seed in 0 1 2; do
  run c2_rec_lora_s${seed} $PY -m src.experiments.run_finetune_chronos2 --mode lora --rank 16 --seed $seed --dataset frn50krec --out results/finetune_v2/c2_rec_lora_r16_s${seed}
done
# 비교용: 검열 그대로의 50K LoRA (실험 3의 대응 셀)
for seed in 0 1 2; do
  run c2_50k_lora_s${seed} $PY -m src.experiments.run_finetune_chronos2 --mode lora --rank 16 --seed $seed --dataset frn50k --out results/finetune_v2/c2_50k_lora_r16_s${seed}
done

# recovered TFT 체인 종료 대기 후 censored predictTFT 재실행 (예측 저장 목적)
until grep -qE "TFT recovered predict (OK|FAIL)" ~/tsfm-demand/logs/queue_tft.log 2>/dev/null; do sleep 300; done
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
run tft_censored_repredict $RPY predictTFT.py
echo "=== EXP3 QUEUE DONE $(date) ===" | tee -a $LOG
