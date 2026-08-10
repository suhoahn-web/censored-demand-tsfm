"""FreshRetailNet(LT/50K) → 일 단위 실험용 패널 생성.

산출물 (data/processed/):
  {lt|frn50k}_daily_train.parquet — 공식 train 구간 일 단위 패널 (컨텍스트용)
  {lt|frn50k}_daily_eval.parquet  — 공식 eval 7일 정답
원본은 수정하지 않는다. lab06에서도 동일하게 실행 가능.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRS = {
    "lt": PROJECT_ROOT / "data" / "raw" / "FreshRetailNet-LT" / "data",
    "frn50k": PROJECT_ROOT / "data" / "raw" / "FreshRetailNet-50K" / "data",
}
OUT = PROJECT_ROOT / "data" / "processed"

COLS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt",
        "discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature"]


def run(dataset: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIRS[dataset]
    for split in ["train", "eval"]:
        df = pd.read_parquet(raw / f"{split}.parquet", columns=COLS)
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
        out_name = f"{dataset}_daily_{split}.parquet"
        df.to_parquet(OUT / out_name, index=False)
        n = df.groupby(["store_id", "product_id"]).ngroups
        print(f"{out_name}: {df.shape}, {n} series, {df.dt.min().date()}~{df.dt.max().date()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    args = ap.parse_args()
    run(args.dataset)
