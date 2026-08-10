#!/bin/bash
# 주말 zero-shot 큐 (lab06, nohup으로 실행).
# Chronos-2 × {LT 공식 + rolling origin 2개 + 50K 공식} × {raw, 품절마스킹} = 10 runs.
# 한 실행이 실패해도 다음으로 진행. 로그: ~/tsfm-demand/logs/
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
CKPT=data/checkpoints/amazon__chronos-2
mkdir -p logs results/zeroshot

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a logs/queue.log
  $PY -m src.experiments.run_zeroshot "$@" >> "logs/$name.log" 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a logs/queue.log \
    || echo "=== $name FAIL $(date) ===" | tee -a logs/queue.log
}

# LT 공식 origin (2025-07-13 컷) — 주 결과
run c2_lt_raw       --model chronos2 --model-path $CKPT --dataset lt --out results/zeroshot/chronos2_lt_raw.parquet
run c2_lt_mask      --model chronos2 --model-path $CKPT --dataset lt --mask-stockout --out results/zeroshot/chronos2_lt_mask.parquet
# LT rolling origin 2개 (컨텍스트 컷: 공식 -7일, -14일; 정답은 train 구간에 존재)
run c2_lt_o2_raw    --model chronos2 --model-path $CKPT --dataset lt --cutoff-date 2025-07-06 --out results/zeroshot/chronos2_lt_o2_raw.parquet
run c2_lt_o2_mask   --model chronos2 --model-path $CKPT --dataset lt --cutoff-date 2025-07-06 --mask-stockout --out results/zeroshot/chronos2_lt_o2_mask.parquet
run c2_lt_o1_raw    --model chronos2 --model-path $CKPT --dataset lt --cutoff-date 2025-06-29 --out results/zeroshot/chronos2_lt_o1_raw.parquet
run c2_lt_o1_mask   --model chronos2 --model-path $CKPT --dataset lt --cutoff-date 2025-06-29 --mask-stockout --out results/zeroshot/chronos2_lt_o1_mask.parquet
# 50K 강건성 (공식 eval 2024-06-26~07-02)
run c2_50k_raw      --model chronos2 --model-path $CKPT --dataset frn50k --out results/zeroshot/chronos2_50k_raw.parquet
run c2_50k_mask     --model chronos2 --model-path $CKPT --dataset frn50k --mask-stockout --out results/zeroshot/chronos2_50k_mask.parquet
# 50K rolling origin 1개 (2024-06-18 컷)
run c2_50k_o2_raw   --model chronos2 --model-path $CKPT --dataset frn50k --cutoff-date 2024-06-18 --out results/zeroshot/chronos2_50k_o2_raw.parquet
run c2_50k_o2_mask  --model chronos2 --model-path $CKPT --dataset frn50k --cutoff-date 2024-06-18 --mask-stockout --out results/zeroshot/chronos2_50k_o2_mask.parquet

echo "=== QUEUE DONE $(date) ===" | tee -a logs/queue.log
