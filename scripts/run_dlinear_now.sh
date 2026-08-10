#!/bin/bash
# DLinear 즉시 실행 (TimesFM 대기 없이 병렬). 검열 + 복원 두 조건.
set -u
RPY=~/miniconda3/envs/repro/bin/python
LOG=~/tsfm-demand/logs/queue_dlinear.log
RAW=~/tsfm-demand/data/raw/FreshRetailNet-50K/data/train.parquet
REC=~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp/demand/demand.parquet
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/DLinear

common="--is_training 1 --train_only True --model DLinear --scale True --loss mae \
  --features MS --target sale_amount --revin False --total_seq_len 90 --seq_len 62 \
  --pred_len 7 --moving_avg 28 --enc_in 7 --do_predict --itr 1 --batch_size 1024 \
  --train_epochs 6 --num_workers 8 --learning_rate 0.001"

echo "=== dlinear_censored start $(date) ===" | tee -a $LOG
$RPY -u main.py $common --data censored_dataset --data_path $RAW --des dlin_censored \
  >> ~/tsfm-demand/logs/dlinear_censored.log 2>&1 \
  && echo "=== dlinear_censored OK $(date) ===" | tee -a $LOG \
  || echo "=== dlinear_censored FAIL $(date) ===" | tee -a $LOG

echo "=== dlinear_recovered start $(date) ===" | tee -a $LOG
$RPY -u main.py $common --data censored_dataset --data_path $REC --des dlin_recovered \
  >> ~/tsfm-demand/logs/dlinear_recovered.log 2>&1 \
  && echo "=== dlinear_recovered OK $(date) ===" | tee -a $LOG \
  || echo "=== dlinear_recovered FAIL $(date) ===" | tee -a $LOG

echo "=== DLINEAR QUEUE DONE $(date) ===" | tee -a $LOG
