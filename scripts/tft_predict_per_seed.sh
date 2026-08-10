#!/bin/bash
# TFT 시드별 예측 재실행.
#
# 문제: predictTFT.py는 체크포인트 디렉토리에서 `sorted(files)[-1]`을 고른다.
# Lightning이 재실행 시 `-v1`/`-v2` 접미사를 붙이는데 ASCII상 '-'(0x2D) < '.'(0x2E)라
# 원본 `epoch=4-step=3415.ckpt`가 항상 마지막으로 정렬되어, 시드별 학습을 해도
# 예측은 늘 원본 체크포인트로 수행됐다. 학습 산출물은 남아 있으므로 예측만 재실행한다.
#
# 방법: 체크포인트 디렉토리를 시드별로 하나만 남기고 predict → 결과를 시드 태그로 보존.
set -eu
RPY=~/miniconda3/envs/repro/bin/python
TFT=~/tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT
CKDIR=$TFT/lightning_logs/censored/checkpoints
LOG=~/tsfm-demand/logs/queue_tft_seedpredict.log
cd $TFT

BACKUP=$TFT/lightning_logs/censored/_all_ckpts
mkdir -p $BACKUP
mv $CKDIR/*.ckpt $BACKUP/ 2>/dev/null || true

# 파일명 → 시드 매핑 (학습 시각 순: 원본=2025, v1=seed1, v2=seed2)
declare -A MAP=( ["epoch=4-step=3415.ckpt"]="s2025" \
                 ["epoch=4-step=3415-v1.ckpt"]="s1" \
                 ["epoch=4-step=3415-v2.ckpt"]="s2" )

for f in "${!MAP[@]}"; do
  tag=${MAP[$f]}
  [ -f "$BACKUP/$f" ] || { echo "missing $f" | tee -a $LOG; continue; }
  rm -f $CKDIR/*.ckpt
  cp "$BACKUP/$f" "$CKDIR/"
  echo "=== predict $tag ($f) start $(date) ===" | tee -a $LOG
  if $RPY -u predictTFT.py >> ~/tsfm-demand/logs/tft_predict_$tag.log 2>&1; then
    mv censored_res_last_epoch.parquet "censored_res_${tag}.parquet"
    [ -f censored_epoch4_predictions.parquet ] && \
      mv censored_epoch4_predictions.parquet "censored_predictions_${tag}.parquet"
    echo "=== predict $tag OK $(date) ===" | tee -a $LOG
  else
    echo "=== predict $tag FAIL $(date) ===" | tee -a $LOG
  fi
done

# 체크포인트 원상 복구
rm -f $CKDIR/*.ckpt
cp $BACKUP/*.ckpt $CKDIR/
echo "=== TFT SEED PREDICT DONE $(date) ===" | tee -a $LOG
