"""점예측 모델에 경험적 분위수를 부여해 분포예측 모델과 공정 비교.

동기 (리뷰어 방어): "TSFM은 분위수를 주고 전통 기법은 못 준다"는 비교는 불공정하다.
전통 기법도 잔차 분포로 예측구간을 만들 수 있으므로(표준 실무), 동일 조건에서 겨루게 한다.

절차:
  1) 검증 원점(train 마지막 −7일)에서 동일 모델로 7일 예측 → 실제값과 비교해 오차 수집
  2) 시계열 규모 차이를 없애기 위해 오차를 시계열 평균으로 스케일링
  3) horizon별로 풀링한 스케일 오차의 분위수 Q(q,h) 산출
  4) 테스트 예측에 적용: q_pred = max(0, point + scale_i × Q(q,h))

사용: python -m src.experiments.add_empirical_quantiles --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
HORIZON = 7
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_COLS = {"croston": "CrostonClassic", "sba": "CrostonSBA", "tsb": "TSB",
              "ets": "AutoETS", "snaive": "SeasonalNaive"}


def validation_errors(dataset: str, uncensored_only: bool = False
                      ) -> tuple[pd.DataFrame, pd.Series]:
    """검증 원점(마지막 −7일)에서의 모델별 오차와 시계열 스케일.

    uncensored_only=True면 검증창에서 품절시간이 0인 날의 잔차만 쓴다
    (JORS R1 Major 5: 기본 절차는 검열일 잔차를 포함한다 — LT 검증창 품절일 49%).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import (TSB, AutoETS, CrostonClassic, CrostonSBA,
                                      SeasonalNaive)

    train = pd.read_parquet(PROCESSED / f"{dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    eval_keys = (pd.read_parquet(PROCESSED / f"{dataset}_daily_eval.parquet",
                                 columns=KEY).drop_duplicates())
    train = train.merge(eval_keys, on=KEY)
    last = train["dt"].max()
    cutoff = last - pd.Timedelta(days=HORIZON)

    fit_df = train[train["dt"] <= cutoff].copy()
    holdout = train[train["dt"] > cutoff].copy()
    if uncensored_only:
        n0 = len(holdout)
        holdout = holdout[holdout["stock_hour6_22_cnt"] == 0]
        print(f"validation residuals restricted to uncensored days: "
              f"{len(holdout)}/{n0}", flush=True)
    fit_df["unique_id"] = fit_df["store_id"].astype(str) + "_" + fit_df["product_id"].astype(str)
    df = fit_df.rename(columns={"dt": "ds", "sale_amount": "y"})[["unique_id", "ds", "y"]]

    sf = StatsForecast(models=[CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.1, alpha_p=0.05),
                               AutoETS(season_length=7), SeasonalNaive(season_length=7)],
                       freq="D", n_jobs=-1)
    fc = sf.forecast(df=df, h=HORIZON).reset_index()
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    ids = fc["unique_id"].str.split("_", expand=True).astype(int)
    fc["store_id"], fc["product_id"] = ids[0], ids[1]
    fc = fc.sort_values(KEY + ["ds"])
    fc["h"] = fc.groupby(KEY).cumcount() + 1

    holdout = holdout.sort_values(KEY + ["dt"])
    holdout["h"] = holdout.groupby(KEY).cumcount() + 1
    merged = fc.merge(holdout[KEY + ["h", "sale_amount"]], on=KEY + ["h"], how="inner")

    scale = (train.groupby(KEY)["sale_amount"].mean().replace(0, np.nan)
             .fillna(1.0).rename("scale"))
    return merged, scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    ap.add_argument("--uncensored-only", action="store_true",
                    help="검증 잔차를 품절시간 0인 날로 한정 (출력 접미사 equ)")
    args = ap.parse_args()
    suffix = "equ" if args.uncensored_only else "eq"

    val, scale = validation_errors(args.dataset, args.uncensored_only)
    val = val.join(scale, on=KEY)
    print(f"validation rows: {len(val)}", flush=True)

    for short, col in MODEL_COLS.items():
        src = RESULTS / "zeroshot" / f"{short}_{args.dataset}_raw.parquet"
        if not src.exists():
            print(f"skip (no point preds): {src.name}")
            continue
        # horizon별 스케일 오차 분위수
        e = ((val["sale_amount"] - val[col]) / val["scale"]).to_numpy()
        hs = val["h"].to_numpy()
        qtab = {h: np.nanquantile(e[hs == h], QUANTILES) for h in range(1, HORIZON + 1)}

        pred = pd.read_parquet(src)
        pred = pred.join(scale, on=KEY)
        point = pred["q0.5"].to_numpy()
        out = pred[KEY + ["h"]].copy()
        for j, q in enumerate(QUANTILES):
            adj = np.array([qtab[h][j] for h in pred["h"].to_numpy()])
            out[f"q{q}"] = np.clip(point + pred["scale"].to_numpy() * adj, 0, None)
        # 분위수 교차 방지(단조화)
        qcols = [f"q{q}" for q in QUANTILES]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)

        dst = RESULTS / "zeroshot" / f"{short}{suffix}_{args.dataset}_raw.parquet"
        out.to_parquet(dst, index=False)
        meta = {"model": f"{short}+empirical-quantiles"
                         + ("-uncensored" if args.uncensored_only else ""),
                "dataset": args.dataset,
                "mask_stockout": False, "cutoff_date": None, "quantiles": QUANTILES,
                "horizon": HORIZON, "n_series": out.groupby(KEY).ngroups,
                "quantile_source": "empirical residuals at validation origin (train last -7d)"
                                   + (", uncensored validation days only"
                                      if args.uncensored_only else ", all validation days"),
                "validation_rows": int(len(val)),
                "scaled_by": "series mean of train sale_amount"}
        Path(str(dst) + ".meta.json").write_text(json.dumps(meta, indent=2))
        print("saved:", dst.name, flush=True)


if __name__ == "__main__":
    main()
