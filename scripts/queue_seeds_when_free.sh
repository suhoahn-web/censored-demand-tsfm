#!/bin/bash
# TFT·DLinear 시드 보강 (절대 규칙 5). GPU가 비면 자동 시작.
#
# 다른 프로젝트의 vLLM 작업이 A100 메모리 대부분을 점유 중이므로, 여유 메모리가
# 확보될 때까지 대기한 뒤 실행한다. 남의 작업을 OOM으로 죽이지 않는 것이 우선.
set -u
RPY=~/miniconda3/envs/repro/bin/python
LOG=~/tsfm-demand/logs/queue_seeds.log
RAW=~/tsfm-demand/data/raw/FreshRetailNet-50K/data/train.parquet
REC=~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp/demand/demand.parquet
NEED_MB=20000     # TFT 학습에 필요한 여유 (관측 ~10GB + 안전여유)

free_mb() {
  local total used
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo $((total - used))
}

echo "=== waiting for GPU (need ${NEED_MB} MiB free) $(date) ===" | tee -a $LOG
while [ "$(free_mb)" -lt "$NEED_MB" ]; do sleep 300; done
echo "=== GPU free ($(free_mb) MiB), starting $(date) ===" | tee -a $LOG

run() {
  name=$1; shift
  echo "=== $name start $(date) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/$name.log 2>&1 \
    && echo "=== $name OK $(date) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date) ===" | tee -a $LOG
}

# DLinear: 빠름(에폭당 ~90초) — 먼저
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/DLinear
DL="--is_training 1 --train_only True --model DLinear --scale True --loss mae \
 --features MS --revin False --total_seq_len 90 --seq_len 62 --pred_len 7 \
 --moving_avg 28 --enc_in 7 --do_predict --itr 1 --batch_size 1024 \
 --train_epochs 6 --num_workers 8 --learning_rate 0.001"
for S in 1 2; do
  run dlinear_censored_s$S env FRN_SEED=$S $RPY -u main.py $DL \
      --data censored_dataset --data_path $RAW --target sale_amount --des dlin_cen_s$S
  run dlinear_recovered_s$S env FRN_SEED=$S $RPY -u main.py $DL \
      --data recovered_dataset --data_path $REC --target sale_amount_pred --des dlin_rec_s$S
done

# TFT: 시드당 학습 ~1.1h + 예측 ~6분
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
for S in 1 2; do
  run tft_censored_s$S env FRN_SEED=$S $RPY -u trainTFT.py
  run tft_predict_s$S env FRN_SEED=$S $RPY -u predictTFT.py
done

echo "=== SEED QUEUE DONE $(date) ===" | tee -a $LOG
