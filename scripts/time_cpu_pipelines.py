"""E7 (JORS R1, Major 7): CPU 파이프라인의 공통 워크로드 wall-clock 측정.

워크로드: LT 공식 origin, 10,000 평가 시계열, 7일 horizon.
  (1) 고전 5종(Croston/SBA/TSB/AutoETS/sNaive) 적합+예측
  (2) 경험적 분위수 캘리브레이션: 검증 origin 재적합·예측 + 잔차 분위수 표 + 적용
  (3) LightGBM: Tweedie + 9 분위수 목적함수, 20 origin 학습표본, 300 라운드
출력은 scratch 디렉토리로 보내 기존 결과를 덮어쓰지 않는다.

산출: results/compute_cost_cpu_lt.csv
사용: python scripts/time_cpu_pipelines.py
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "results" / "_timing_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def timed(name: str, cmd: list[str]) -> dict:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    sec = time.time() - t0
    ok = r.returncode == 0
    print(f"{name}: {sec:.0f}s {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(r.stderr[-2000:])
    return {"pipeline": name, "wall_seconds": round(sec), "ok": ok}


rows = []
rows.append(timed("classical_5models_fit_forecast",
                  [PY, "-m", "src.experiments.run_traditional", "--dataset", "lt",
                   "--out-dir", "results/_timing_scratch"]))

# EQ 캘리브레이션: validation_errors (재적합+예측) + 분위수 표 + 적용. 파일을 쓰지 않도록
# 함수만 호출한다.
t0 = time.time()
code = (
    "import numpy as np, pandas as pd, time\n"
    "from src.experiments.add_empirical_quantiles import validation_errors, MODEL_COLS, QUANTILES, HORIZON\n"
    "val, scale = validation_errors('lt')\n"
    "val = val.join(scale, on=['store_id','product_id'])\n"
    "for short, col in MODEL_COLS.items():\n"
    "    e = ((val['sale_amount'] - val[col]) / val['scale']).to_numpy(); hs = val['h'].to_numpy()\n"
    "    qtab = {h: np.nanquantile(e[hs == h], QUANTILES) for h in range(1, HORIZON + 1)}\n"
    "print('eq ok')\n")
r = subprocess.run([PY, "-c", code], cwd=ROOT, env=ENV, capture_output=True, text=True)
sec = time.time() - t0
print(f"empirical_quantile_calibration_5models: {sec:.0f}s {'OK' if r.returncode == 0 else 'FAIL'}",
      flush=True)
if r.returncode != 0:
    print(r.stderr[-2000:])
rows.append({"pipeline": "empirical_quantile_calibration_5models",
             "wall_seconds": round(sec), "ok": r.returncode == 0})

rows.append(timed("lightgbm_tweedie_plus_9quantiles",
                  [PY, "-m", "src.experiments.run_lightgbm", "--dataset", "lt",
                   "--out-dir", "results/_timing_scratch"]))

df = pd.DataFrame(rows)
df["hardware"] = f"CPU {os.cpu_count()} threads, {platform.processor()}"
df["workload"] = "LT official origin, 10,000 series, h=7"
df.to_csv(ROOT / "results" / "compute_cost_cpu_lt.csv", index=False)
print(df.to_string(index=False))
