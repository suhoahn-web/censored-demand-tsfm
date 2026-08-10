#!/bin/bash
# TFT 학습 완료 대기 → predictTFT 실행 (원 논문 지표 산출). detached용.
set -u
PY=~/miniconda3/envs/repro/bin/python
LOG=~/tsfm-demand/logs/queue_tft.log

until grep -q "TFT repro train OK" $LOG 2>/dev/null; do
  if grep -q "TFT repro train FAIL" $LOG 2>/dev/null; then
    echo "=== TFT predict skipped (train FAIL) $(date) ===" | tee -a $LOG
    exit 1
  fi
  sleep 60
done

cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
echo "=== TFT predict start $(date) ===" | tee -a $LOG
$PY predictTFT.py >> ~/tsfm-demand/logs/tft_predict.log 2>&1 \
  && echo "=== TFT predict OK $(date) ===" | tee -a $LOG \
  || echo "=== TFT predict FAIL $(date) ===" | tee -a $LOG
