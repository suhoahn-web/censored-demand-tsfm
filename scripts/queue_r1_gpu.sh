#!/bin/bash
# JORS R1 GPU 큐 (lab06). 배양육 작업을 jmm_safe_stop_20261027.py로 중단한 뒤 실행.
#   E7-GPU  zero-shot 추론 피크 메모리·시간 (5 모델)
#   E2      origin별 Chronos-2 LoRA r16(lr 1e-3, seed 0) 재학습 × 8 origin
#   E6-a    복원 파이프라인 MNAR 검증 missing_rate 0.1, 0.5
# 끝나면 배양육 재개 실행을 --continue-resume으로 자동 재개한다 (GPU가 비어 있어야 스크립트가 시작됨).
set -u
cd ~/tsfm-demand
PY=~/miniconda3/envs/tsfm/bin/python
MPY=~/miniconda3/envs/moirai/bin/python
RPY=~/miniconda3/envs/repro/bin/python
CSM=/home/ubuntu/miniconda3/envs/csm/bin/python
LOG=~/tsfm-demand/logs/queue_r1_gpu.log
mkdir -p logs results/zeroshot results/gpu_cost results/finetune_mo

run() {
  name=$1; shift
  echo "=== $name start $(date -u) ===" | tee -a $LOG
  "$@" >> ~/tsfm-demand/logs/r1_${name}.log 2>&1 \
    && echo "=== $name OK $(date -u) ===" | tee -a $LOG \
    || echo "=== $name FAIL $(date -u) ===" | tee -a $LOG
}

# ── E7-GPU ──
run gpucost_chronos2 $PY scripts/measure_gpu_cost.py --model chronos2 --model-path data/checkpoints/amazon__chronos-2
run gpucost_tirex    $PY scripts/measure_gpu_cost.py --model tirex    --model-path data/checkpoints/NX-AI__TiRex
run gpucost_ttm      $PY scripts/measure_gpu_cost.py --model ttm      --model-path data/checkpoints/ibm-granite__granite-timeseries-ttm-r2
run gpucost_moirai2  $MPY scripts/measure_gpu_cost.py --model moirai2 --model-path data/checkpoints/Salesforce__moirai-2.0-R-small
run gpucost_timesfm  $PY scripts/measure_gpu_cost.py --model timesfm  --model-path data/checkpoints/google__timesfm-2.5-200m-pytorch --limit 1000

# ── E2: origin별 LoRA r16 (lr 1e-3, seed 0) ──
ORIGINS="2024-07-07 2024-09-01 2024-10-27 2024-12-22 2025-02-16 2025-04-13 2025-05-11 2025-06-08"
for CUT in $ORIGINS; do
  TAG=mo${CUT//-/}
  if [ -f results/zeroshot/c2lora_lt_${TAG}_raw.parquet ]; then echo "=== c2lora_${TAG} SKIP ===" | tee -a $LOG; continue; fi
  run c2lora_${TAG} $PY -m src.experiments.run_finetune_chronos2 --mode lora --rank 16 --learning-rate 1e-3 --seed 0 \
      --dataset lt --cutoff-date $CUT --min-context-days 90 \
      --out results/finetune_mo/c2_lora_r16_lr1e-3_${TAG}_s0 \
      --out-parquet results/zeroshot/c2lora_lt_${TAG}_raw.parquet
done

# ── E6-a: MNAR 검증 0.1 / 0.5 ──
cd ~/tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp
run mnar_0.1 $RPY app.py --model TimesNet --missing_rate 0.1
run mnar_0.5 $RPY app.py --model TimesNet --missing_rate 0.5
cd ~/tsfm-demand

echo "=== R1 GPU QUEUE DONE $(date -u) ===" | tee -a $LOG

# ── 배양육 재개 (문서화된 명령 + --continue-resume) ──
sleep 10
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "=== GPU still busy; NOT resuming JMM run — resume manually ===" | tee -a $LOG; exit 0
fi
cd /data/cultured_meat_jmm_r1
nohup $CSM jmm_r1_resume_32b.py production \
  --run runs/full641_32b_direct_repetition30_20261011_v1 \
  --out runs/full641_32b_direct_repetition30_20261011_v1_resume_20261022 \
  --interruption-record interruptions/full641_32b_direct_repetition30_20261011_v1_interruption_20261022.json \
  --continue-resume >> runs/full641_32b_direct_repetition30_20261011_v1_resume_20261022_continue_20261027.log 2>&1 &
echo "=== JMM resume relaunched pid $! $(date -u) ===" | tee -a $LOG
