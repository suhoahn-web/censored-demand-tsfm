"""ADI × CV² 기반 수요 패턴 사분면 분류 (Syntetos-Boylan-Croston).

경계값 (Syntetos & Boylan 2005):
  ADI 1.32, CV² 0.49
  smooth       : ADI < 1.32, CV² < 0.49
  erratic      : ADI < 1.32, CV² >= 0.49
  intermittent : ADI >= 1.32, CV² < 0.49
  lumpy        : ADI >= 1.32, CV² >= 0.49

주의: 관측 판매량 기준 분류(검열 포함). 복원 수요 기준 분류와의 차이는
ablation에서 다룬다. 분류는 train 구간(90일)만 사용 — 미래 누출 금지.
"""

import argparse

import numpy as np
import pandas as pd

from .loading import PROCESSED_DIR, SERIES_KEY, daily_panel

ADI_CUT = 1.32
CV2_CUT = 0.49


def classify_series(g: pd.Series) -> pd.Series:
    """단일 시계열의 일 판매량 벡터 → ADI, CV², 사분면 라벨."""
    v = g.to_numpy(dtype=float)
    nonzero = v[v > 0]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        return pd.Series({"adi": np.inf, "cv2": np.nan, "quadrant": "all_zero",
                          "zero_frac": 1.0, "mean_nonzero": 0.0})
    adi = len(v) / n_nonzero
    cv2 = (nonzero.std(ddof=0) / nonzero.mean()) ** 2 if nonzero.mean() > 0 else np.nan
    if adi < ADI_CUT:
        quadrant = "smooth" if cv2 < CV2_CUT else "erratic"
    else:
        quadrant = "intermittent" if cv2 < CV2_CUT else "lumpy"
    return pd.Series({"adi": adi, "cv2": cv2, "quadrant": quadrant,
                      "zero_frac": float((v == 0).mean()),
                      "mean_nonzero": float(nonzero.mean())})


def run(out_name: str = "series_classification.parquet") -> pd.DataFrame:
    df = daily_panel("train")
    result = (
        df.sort_values("dt")
        .groupby(SERIES_KEY)["sale_amount"]
        .apply(classify_series)
        .unstack()
        .reset_index()
    )
    stockout = (
        df.groupby(SERIES_KEY)["stock_hour6_22_cnt"]
        .agg(stockout_day_frac=lambda s: (s > 0).mean(),
             mean_stockout_hours="mean")
        .reset_index()
    )
    result = result.merge(stockout, on=SERIES_KEY)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / out_name
    result.to_parquet(out_path, index=False)
    print(f"saved: {out_path} ({len(result)} series)")
    print(result["quadrant"].value_counts())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="series_classification.parquet")
    args = parser.parse_args()
    run(args.out)
