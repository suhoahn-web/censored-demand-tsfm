#!/bin/bash
# TFT 재현 학습 (frn-50k-baseline 원 프로토콜, censored=raw 조건, 50K 데이터)
# repro env(torch 2.0.1) 사용. 로그: ~/tsfm-demand/logs/tft_repro.log
set -u
PY=~/miniconda3/envs/repro/bin/python
cd ~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
mkdir -p ~/tsfm-demand/logs

echo "=== TFT repro train start $(date) ===" | tee -a ~/tsfm-demand/logs/queue_tft.log
$PY trainTFT.py >> ~/tsfm-demand/logs/tft_repro.log 2>&1 \
  && echo "=== TFT repro train OK $(date) ===" | tee -a ~/tsfm-demand/logs/queue_tft.log \
  || echo "=== TFT repro train FAIL $(date) ===" | tee -a ~/tsfm-demand/logs/queue_tft.log
