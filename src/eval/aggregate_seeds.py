"""TFT·DLinear 시드 반복 결과 집계 (절대 규칙 5).

원 리포 산출물은 실행 폴더(`--des` 태그)로 구분되므로, 시드별 폴더를 모아
평균±표준편차를 낸다. 시드 큐(`scripts/queue_seeds_when_free.sh`) 완료 후 실행.

사용 (lab06): python -m src.eval.aggregate_seeds
"""

import re
from pathlib import Path

import pandas as pd

HOME = Path.home()
BASE = HOME / "tsfm-demand/external/frn-50k-baseline/demand_forecasting"
OUT = HOME / "tsfm-demand/results/seed_summary.csv"

# (표시명, 결과 폴더 glob, group_type 필터)
TARGETS = [
    ("TFT (censored)", BASE / "TFT", "censored*_res_last_epoch.parquet"),
    ("TFT (recovered)", BASE / "TFT", "recovered*_res_last_epoch.parquet"),
    ("DLinear (censored)", BASE / "DLinear/results", "*dlin_cen*/metrics_res.parquet"),
    ("DLinear (recovered)", BASE / "DLinear/results", "*dlin_rec*/metrics_res.parquet"),
]


def collect(label: str, root: Path, pattern: str) -> list[dict]:
    rows = []
    for f in sorted(root.glob(pattern)):
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        g = d[d.group_type == "psd>=0"] if "group_type" in d.columns else d
        m = re.search(r"_s(\d+)", str(f))
        rows.append({"model": label, "seed": int(m.group(1)) if m else 0,
                     "wape": float(g.wape.mean()), "wpe": float(g.wpe.mean()),
                     "source": f.parent.name if f.name == "metrics_res.parquet" else f.name})
    return rows


if __name__ == "__main__":
    rows = []
    for label, root, pat in TARGETS:
        if root.exists():
            rows += collect(label, root, pat)
    if not rows:
        print("no seed results found yet")
        raise SystemExit(0)
    df = pd.DataFrame(rows).drop_duplicates(subset=["model", "source"])
    df.to_csv(OUT, index=False)
    summary = (df.groupby("model")
               .agg(n_seeds=("seed", "nunique"),
                    wape_mean=("wape", "mean"), wape_sd=("wape", "std"),
                    wpe_mean=("wpe", "mean"), wpe_sd=("wpe", "std"))
               .round(4))
    print(summary.to_string())
    print("\nper-run detail:")
    print(df[["model", "seed", "wape", "wpe", "source"]].round(4).to_string(index=False))
    print("\nsaved:", OUT)
