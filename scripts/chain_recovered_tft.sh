#!/bin/bash
# Stage-1 복원(demand.parquet) 완료 대기 → TFT 복원수요 학습 → predict. detached용.
set -u
RPY=~/miniconda3/envs/repro/bin/python
DEMAND=~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp/demand/demand.parquet
LOG=~/tsfm-demand/logs/queue_tft.log

# 복원 프로세스 종료 + demand.parquet 생성 대기
until [ -f "$DEMAND" ] && ! pgrep -f "app.py --model TimesNet" > /dev/null; do
  sleep 120
done
sleep 30

cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
echo "=== TFT recovered train start $(date) ===" | tee -a $LOG
$RPY trainTFT.py --demand >> ~/tsfm-demand/logs/tft_recovered_train.log 2>&1 \
  && echo "=== TFT recovered train OK $(date) ===" | tee -a $LOG \
  || { echo "=== TFT recovered train FAIL $(date) ===" | tee -a $LOG; exit 1; }

echo "=== TFT recovered predict start $(date) ===" | tee -a $LOG
$RPY predictTFT.py --demand >> ~/tsfm-demand/logs/tft_recovered_predict.log 2>&1 \
  && echo "=== TFT recovered predict OK $(date) ===" | tee -a $LOG \
  || echo "=== TFT recovered predict FAIL $(date) ===" | tee -a $LOG
