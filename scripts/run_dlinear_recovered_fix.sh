#!/bin/bash
# DLinear 복원수요 조건 (수정판).
# 주의: 원 리포의 train_predict_on_recovered.sh와 동일하게
#   --data recovered_dataset --target sale_amount_pred 를 써야 한다.
#   (censored_dataset + sale_amount 로 돌리면 demand.parquet의 원본 판매 열을 읽어
#    검열 조건과 동일해진다 — 2026-08-08 실수 교정)
set -u
RPY=~/miniconda3/envs/repro/bin/python
LOG=~/tsfm-demand/logs/queue_dlinear.log
REC=~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp/demand/demand.parquet
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/DLinear

echo "=== dlinear_recovered_fix start $(date) ===" | tee -a $LOG
$RPY -u main.py \
  --is_training 1 --train_only True --model DLinear --scale True --loss mae \
  --features MS --data recovered_dataset --data_path $REC --target sale_amount_pred \
  --revin False --total_seq_len 90 --seq_len 62 --pred_len 7 --moving_avg 28 \
  --enc_in 7 --des dlin_recovered_fix --do_predict --itr 1 --batch_size 1024 \
  --train_epochs 6 --num_workers 8 --learning_rate 0.001 \
  >> ~/tsfm-demand/logs/dlinear_recovered_fix.log 2>&1 \
  && echo "=== dlinear_recovered_fix OK $(date) ===" | tee -a $LOG \
  || echo "=== dlinear_recovered_fix FAIL $(date) ===" | tee -a $LOG
