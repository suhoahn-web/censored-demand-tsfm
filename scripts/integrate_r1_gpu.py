"""JORS R1 GPU 큐 결과 통합 (lab06에서 scp로 가져온 뒤 로컬 실행).

  (1) E6-a MNAR 검증 로그(logs/r1/r1_mnar_0.1.log, r1_mnar_0.5.log) → results/recovery_mnar_validation_rates.csv
  (2) E7-GPU results/gpu_cost/*.json → results/tables/r1_C9_gpu_cost.md + compute_cost_table.md의 n/m 셀 채움
  (3) E2 origin별 LoRA·LightGBM 파일 존재 확인 → multi_origin_summary / multi_origin_inventory 재실행 안내

사용: python scripts/integrate_r1_gpu.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
LOGS = ROOT / "logs" / "r1"
REV = ROOT.parent / "Target Journal" / "Journal of the Operational Research Society" / "Revision-1"

# (1) MNAR
rows = [{"missing_rate": 0.3, "masked_hours": 9163425, "wape": 0.2761868810581497,
         "wpe": 0.014310895302210207, "source": "2026-08-16 run"}]
for rate in ["0.1", "0.5"]:
    f = LOGS / f"r1_mnar_{rate}.log"
    if not f.exists():
        print("missing", f)
        continue
    t = f.read_text(encoding="utf-8", errors="replace")
    w = re.findall(r"^wape\s+([0-9.]+)", t, re.M)
    p = re.findall(r"^wpe\s+(-?[0-9.]+)", t, re.M)
    n = re.findall(r"^(\d{6,})\s*$", t, re.M)  # generate_data의 valid_idx.sum() 출력
    if w and p:
        rows.append({"missing_rate": float(rate), "masked_hours": int(n[0]) if n else None,
                     "wape": float(w[-1]), "wpe": float(p[-1]), "source": "2026-10-27 run"})
mn = pd.DataFrame(rows).sort_values("missing_rate")
mn.to_csv(R / "recovery_mnar_validation_rates.csv", index=False)
print(mn.to_string(index=False))

# (2) GPU cost
gc = []
for f in sorted((R / "gpu_cost").glob("*.json")) if (R / "gpu_cost").exists() else []:
    gc.append(json.loads(f.read_text()))
if gc:
    g = pd.DataFrame(gc)
    md = ["| Model | Series measured | Wall-clock (s) | Seconds per 10k series | Peak GPU memory allocated (GB) | Peak reserved (GB) |",
          "|---|---|---|---|---|---|"]
    label = {"chronos2": "Chronos-2", "tirex": "TiRex", "ttm": "TTM-R2", "moirai2": "Moirai-2.0-R-small",
             "timesfm": "TimesFM-2.5"}
    for r in g.itertuples():
        md.append(f"| {label.get(r.model, r.model)} | {r.n_series:,} | {r.wall_seconds} | {r.seconds_per_10k_series} "
                  f"| {r.peak_gpu_mem_allocated_GB} | {r.peak_gpu_mem_reserved_GB} |")
    (R / "tables" / "r1_C9_gpu_cost.md").write_text(
        "**Table C9 (GPU part).** Zero-shot inference on the LT evaluation series, one A100 80GB, "
        "batch size 256: wall-clock and peak GPU memory (torch.cuda.max_memory_allocated / reserved). "
        "Measured 2026-10-27.\n\n" + "\n".join(md), encoding="utf-8")
    print("\n".join(md))
    # compute_cost_table.md의 n/m 셀 채우기 (행 라벨로 매칭)
    cc = REV / "tables" / "compute_cost_table.md"
    t = cc.read_text(encoding="utf-8")
    rowkey = {"chronos2": "| Chronos-2 zero-shot |", "tirex": "| TiRex zero-shot |",
              "ttm": "| TTM-R2 zero-shot |", "moirai2": "| Moirai-2.0-R-small zero-shot |",
              "timesfm": "| TimesFM-2.5 zero-shot |"}
    for r in g.itertuples():
        key = rowkey.get(r.model)
        if not key or key not in t:
            continue
        i = t.index(key)
        j = t.index("\n", i)
        line = t[i:j]
        cells = line.split("|")
        # 열: pipeline, train, hp, recovery, calib, inference, peak mem, hardware, det
        cells[6] = f" {r.peak_gpu_mem_allocated_GB} GB alloc / {r.peak_gpu_mem_reserved_GB} GB reserved "
        cells[5] = f" {r.seconds_per_10k_series} s per 10k series (measured on {r.n_series:,}) "
        t = t[:i] + "|".join(cells) + t[j:]
    t = t.replace("Peak GPU memory was not recorded during the original runs and is marked n/m",
                  "Peak GPU memory for zero-shot inference was re-measured on 2026-10-27; adaptation and "
                  "baseline runs are marked n/m")
    cc.write_text(t, encoding="utf-8")
    print("compute_cost_table.md updated")

# (3) origin runs
have = sorted(p.name for p in (R / "zeroshot").glob("c2lora_lt_mo*_raw.parquet"))
lg = sorted(p.name for p in (R / "zeroshot").glob("lightgbm_lt_mo*_raw.parquet"))
print(f"\nc2lora origins: {len(have)}  lightgbm origins: {len(lg)}")
if len(have) == 8 and len(lg) == 8:
    print("run: python -m src.eval.multi_origin_summary --dataset lt && python -m src.eval.multi_origin_inventory --dataset lt")
