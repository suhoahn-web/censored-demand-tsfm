"""P0 Task 1: 다중 origin의 정확한 날짜·간격·시계열 수를 실험 파일에서 복원한다.

원고가 "eight origins spaced 56 days apart between 2024-07-07 and 2025-06-08"라고
썼는데, 이 세 진술은 동시에 참일 수 없다(336일 / 56일 = 6구간 = 7개 origin).
meta.json이 진실의 원천이다.

산출: results/multi_origin_exact_dates.csv
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZS = ROOT / "results" / "zeroshot"

rows = {}
for mp in sorted(ZS.glob("*_mo*_*.meta.json")):
    meta = json.loads(mp.read_text())
    cut = meta.get("cutoff_date")
    if not cut:
        continue
    rec = rows.setdefault(cut, {"n_series": set(), "models": set()})
    rec["n_series"].add(meta.get("n_series"))
    rec["models"].add(meta.get("model"))

out = []
prev = None
for i, cut in enumerate(sorted(rows), 1):
    c = pd.Timestamp(cut)
    gap = (c - prev).days if prev is not None else None
    prev = c
    out.append({
        "origin_id": i,
        "forecast_origin": cut,
        "days_since_previous": gap,
        "train_end": cut,
        # 고전기법 EQ의 검증 원점: origin 직전 7일 (run_traditional_origins.py의 vcut)
        "validation_start": str((c - pd.Timedelta(days=6)).date()),
        "validation_end": cut,
        "forecast_start": str((c + pd.Timedelta(days=1)).date()),
        "forecast_end": str((c + pd.Timedelta(days=7)).date()),
        "n_series_zeroshot": max(s for s in rows[cut]["n_series"] if s),
        "models": ",".join(sorted(rows[cut]["models"])),
    })

df = pd.DataFrame(out)
df.to_csv(ROOT / "results" / "multi_origin_exact_dates.csv", index=False)
print(df.to_string(index=False))
print(f"\norigins: {len(df)}")
print("gaps:", df.days_since_previous.dropna().astype(int).tolist())
span = pd.Timestamp(df.forecast_origin.iloc[-1]) - pd.Timestamp(df.forecast_origin.iloc[0])
print(f"span: {df.forecast_origin.iloc[0]} ~ {df.forecast_origin.iloc[-1]} = {span.days} days")
