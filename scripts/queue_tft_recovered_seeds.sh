#!/bin/bash
# TFT 복원수요 조건의 시드 1·2 보강 (검열 조건은 완료).
# 예측은 시드별 체크포인트를 지정해 수행 — predictTFT의 정렬 기반 선택 버그 회피.
set -u
RPY=~/miniconda3/envs/repro/bin/python
TFT=~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
CKDIR=$TFT/lightning_logs/recovered/checkpoints
LOG=~/tsfm-demand/logs/queue_tft_rec_seeds.log
cd $TFT

for S in 1 2; do
  echo "=== tft_rec_train_s$S start $(date) ===" | tee -a $LOG
  if env FRN_SEED=$S $RPY -u trainTFT.py --demand \
       >> ~/tsfm-demand/logs/tft_rec_train_s$S.log 2>&1; then
    echo "=== tft_rec_train_s$S OK $(date) ===" | tee -a $LOG
  else
    echo "=== tft_rec_train_s$S FAIL $(date) ===" | tee -a $LOG; continue
  fi

  # 방금 학습된 체크포인트(가장 최근 수정본)만 남기고 예측
  NEW=$(ls -t $CKDIR/epoch=4-*.ckpt 2>/dev/null | head -1)
  [ -z "$NEW" ] && { echo "no ckpt for s$S" | tee -a $LOG; continue; }
  STASH=$TFT/lightning_logs/recovered/_stash_s$S
  mkdir -p $STASH
  for f in $CKDIR/*.ckpt; do [ "$f" != "$NEW" ] && mv "$f" $STASH/; done

  echo "=== tft_rec_predict_s$S start $(date) ($(basename $NEW)) ===" | tee -a $LOG
  if env FRN_SEED=$S $RPY -u predictTFT.py --demand \
       >> ~/tsfm-demand/logs/tft_rec_predict_s$S.log 2>&1; then
    mv recovered_res_last_epoch.parquet "recovered_res_s$S.parquet"
    [ -f recovered_epoch4_predictions.parquet ] && \
      mv recovered_epoch4_predictions.parquet "recovered_predictions_s$S.parquet"
    echo "=== tft_rec_predict_s$S OK $(date) ===" | tee -a $LOG
  else
    echo "=== tft_rec_predict_s$S FAIL $(date) ===" | tee -a $LOG
  fi
  mv $STASH/*.ckpt $CKDIR/ 2>/dev/null || true
  rmdir $STASH 2>/dev/null || true
done
echo "=== TFT REC SEEDS DONE $(date) ===" | tee -a $LOG
