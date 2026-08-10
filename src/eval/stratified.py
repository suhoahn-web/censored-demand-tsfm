"""ADI×CV² 사분면별 성능 층화 분석.

용법 주의 (설계서 v2.0 3절): 경계값 1.32/0.49는 Croston 대 SBA의 이론 MSE 우위 경계
(Syntetos, Boylan & Croston 2005, JORS)이며 보편적 수요 taxonomy가 아니다
(Kostenko & Hyndman 2006 정정). 여기서는 **방법 선택 규칙이 아니라 성능 층화용
기술 통계**로만 사용한다.

사용: python -m src.eval.stratified --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
ADI_CUT, CV2_CUT = 1.32, 0.49
TRUTH_OF = {"frn50krec": "frn50k"}


def classify(dataset: str) -> pd.DataFrame:
    """train 구간 관측 판매량 기준 ADI/CV²/사분면 + 품절 강도."""
    ds = TRUTH_OF.get(dataset, dataset)
    cache = PROCESSED / f"{ds}_classification.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_parquet(PROCESSED / f"{ds}_daily_train.parquet",
                         columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])

    def per(g: pd.DataFrame) -> pd.Series:
        v = g["sale_amount"].to_numpy(dtype=float)
        nz = v[v > 0]
        if len(nz) == 0:
            return pd.Series({"adi": np.inf, "cv2": np.nan, "quadrant": "all_zero",
                              "stockout_day_frac": (g["stock_hour6_22_cnt"] > 0).mean()})
        adi = len(v) / len(nz)
        cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if nz.mean() > 0 else np.nan
        quad = ("smooth" if cv2 < CV2_CUT else "erratic") if adi < ADI_CUT else \
               ("intermittent" if cv2 < CV2_CUT else "lumpy")
        return pd.Series({"adi": adi, "cv2": cv2, "quadrant": quad,
                          "stockout_day_frac": (g["stock_hour6_22_cnt"] > 0).mean()})

    out = df.groupby(KEY).apply(per, include_groups=False).reset_index()
    out.to_parquet(cache, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    args = ap.parse_args()

    cls = classify(args.dataset).set_index(KEY)
    mat = pd.read_parquet(RESULTS / f"per_series_mase_{args.dataset}.parquet")
    print("quadrant 분포:")
    print(cls.loc[mat.index, "quadrant"].value_counts().to_string())

    joined = mat.join(cls[["quadrant", "stockout_day_frac"]])

    print("\n=== 사분면별 평균 MASE (낮을수록 우수) ===")
    q = joined.groupby("quadrant")[mat.columns].mean().T
    q["overall"] = mat.mean()
    print(q.sort_values("overall").round(3).to_string())

    print("\n=== 사분면별 평균 순위 (모델 간 상대 위치) ===")
    ranks = mat.rank(axis=1).join(cls[["quadrant"]])
    rq = ranks.groupby("quadrant")[mat.columns].mean().T
    print(rq.round(2).sort_values(rq.columns[0]).to_string())

    print("\n=== 품절 강도 3분위별 평균 순위 ===")
    joined["stockout_tercile"] = pd.qcut(joined["stockout_day_frac"], 3,
                                         labels=["low", "mid", "high"])
    rs = (mat.rank(axis=1).join(joined[["stockout_tercile"]])
          .groupby("stockout_tercile", observed=True)[mat.columns].mean().T)
    rs["swing_high_minus_low"] = rs["high"] - rs["low"]
    print(rs.round(2).sort_values("swing_high_minus_low").to_string())

    q.to_csv(RESULTS / f"stratified_mase_{args.dataset}.csv")
    rs.to_csv(RESULTS / f"stratified_stockout_{args.dataset}.csv")
    print("\nsaved:", RESULTS / f"stratified_mase_{args.dataset}.csv")


if __name__ == "__main__":
    main()
