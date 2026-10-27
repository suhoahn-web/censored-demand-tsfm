"""P0-6 보조: 고전 기법 + 경험적 분위수를 rolling origin마다 재적합.

zero-shot TSFM은 origin을 옮겨도 학습 데이터가 바뀌지 않지만, 고전 기법은 origin마다
다시 적합해야 공정하다. 경험적 분위수도 각 origin의 **직전 7일**을 검증 원점으로 삼아
그때그때 다시 만든다(테스트 구간 정보를 쓰지 않는다).

산출: results/zeroshot/{model}[eq]_lt_mo{YYYYMMDD}_raw.parquet (+ meta)

사용: python -m src.experiments.run_traditional_origins --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
KEY = ["store_id", "product_id"]
HORIZON = 7
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NAME_MAP = {"CrostonClassic": "croston", "CrostonSBA": "sba", "TSB": "tsb",
            "AutoETS": "ets", "SeasonalNaive": "snaive"}
ORIGINS = ["2024-07-07", "2024-09-01", "2024-10-27", "2024-12-22",
           "2025-02-16", "2025-04-13", "2025-05-11", "2025-06-08"]
MIN_CONTEXT = 90


def _fit(df: pd.DataFrame):
    from statsforecast import StatsForecast
    from statsforecast.models import (TSB, AutoETS, CrostonClassic, CrostonSBA,
                                      SeasonalNaive)
    sf = StatsForecast(
        models=[CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.1, alpha_p=0.05),
                AutoETS(season_length=7), SeasonalNaive(season_length=7)],
        freq="D", n_jobs=-1)
    fc = sf.forecast(df=df, h=HORIZON)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    ids = fc["unique_id"].str.split("_", expand=True).astype(int)
    fc["store_id"], fc["product_id"] = ids[0], ids[1]
    fc = fc.sort_values(KEY + ["ds"])
    fc["h"] = fc.groupby(KEY).cumcount() + 1
    return fc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--out-dir", default="results/zeroshot")
    args = ap.parse_args()

    train = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount"])
    eval_keys = (pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                                 columns=KEY).drop_duplicates())
    train = train.merge(eval_keys, on=KEY)
    train["unique_id"] = (train["store_id"].astype(str) + "_"
                          + train["product_id"].astype(str))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for cut_s in ORIGINS:
        cut = pd.Timestamp(cut_s)
        tag = "mo" + cut_s.replace("-", "")
        hist = train[train.dt <= cut]
        n_obs = hist.groupby(KEY)["dt"].size()
        keep = set(map(tuple, np.array(n_obs[n_obs >= MIN_CONTEXT].index.tolist())))
        hist = hist[[t in keep for t in zip(hist.store_id, hist.product_id)]]
        if hist.empty:
            continue

        # 1) 검증 원점(cut-7)에서 적합 -> 잔차 분위수
        vcut = cut - pd.Timedelta(days=HORIZON)
        vfit = hist[hist.dt <= vcut].rename(columns={"dt": "ds", "sale_amount": "y"})
        vhold = hist[hist.dt > vcut].sort_values(KEY + ["dt"]).copy()
        vhold["h"] = vhold.groupby(KEY).cumcount() + 1
        vfc = _fit(vfit[["unique_id", "ds", "y"]])
        val = vfc.merge(vhold[KEY + ["h", "sale_amount"]], on=KEY + ["h"], how="inner")
        scale = (hist.groupby(KEY)["sale_amount"].mean().replace(0, np.nan)
                 .fillna(1.0).rename("scale"))
        val = val.join(scale, on=KEY)

        # 2) 테스트 원점(cut)에서 적합
        tfit = hist.rename(columns={"dt": "ds", "sale_amount": "y"})
        fc = _fit(tfit[["unique_id", "ds", "y"]]).join(scale, on=KEY)

        for col, short in NAME_MAP.items():
            base = fc[KEY + ["h"]].copy()
            point = np.clip(fc[col].to_numpy(), 0, None)
            base["q0.5"] = point
            p1 = out_dir / f"{short}_{args.dataset}_{tag}_raw.parquet"
            base.to_parquet(p1, index=False)
            Path(str(p1) + ".meta.json").write_text(json.dumps(
                {"model": short, "dataset": args.dataset, "mask_stockout": False,
                 "cutoff_date": cut_s, "quantiles": [0.5], "horizon": HORIZON,
                 "n_series": base.groupby(KEY).ngroups, "engine": "statsforecast",
                 "min_context_days": MIN_CONTEXT, "point_only": True}, indent=2))

            e = ((val["sale_amount"] - val[col]) / val["scale"]).to_numpy()
            vh = val["h"].to_numpy()
            qtab = {h: np.nanquantile(e[vh == h], QUANTILES)
                    for h in range(1, HORIZON + 1)}
            hs = fc["h"].to_numpy()
            eq = fc[KEY + ["h"]].copy()
            for j, q in enumerate(QUANTILES):
                adj = np.array([qtab[h][j] for h in hs])
                eq[f"q{q}"] = np.clip(point + fc["scale"].to_numpy() * adj, 0, None)
            qc = [f"q{q}" for q in QUANTILES]
            eq[qc] = np.sort(eq[qc].to_numpy(), axis=1)
            p2 = out_dir / f"{short}eq_{args.dataset}_{tag}_raw.parquet"
            eq.to_parquet(p2, index=False)
            Path(str(p2) + ".meta.json").write_text(json.dumps(
                {"model": f"{short}+empirical-quantiles", "dataset": args.dataset,
                 "mask_stockout": False, "cutoff_date": cut_s,
                 "quantiles": QUANTILES, "horizon": HORIZON,
                 "n_series": eq.groupby(KEY).ngroups,
                 "min_context_days": MIN_CONTEXT,
                 "quantile_source": "empirical residuals at this origin minus 7 days"},
                indent=2))
        print(f"origin {cut_s}: {fc.groupby(KEY).ngroups} series done", flush=True)


if __name__ == "__main__":
    main()
