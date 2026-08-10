"""전통 베이스라인 (statsforecast): Croston, SBA, TSB, AutoETS, SeasonalNaive.

점예측 → q0.5 단일 분위수로 저장 (zero-shot 러너와 동일 스키마 → 동일 평가 코드).
CPU 실행 (GPU 불필요). 마스킹 조건은 결측 미지원으로 제외 (raw만).
사용: python -m src.experiments.run_traditional --dataset lt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
KEY = ["store_id", "product_id"]
HORIZON = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    ap.add_argument("--out-dir", default="results/zeroshot")
    args = ap.parse_args()

    from statsforecast import StatsForecast
    from statsforecast.models import (TSB, AutoETS, CrostonClassic, CrostonSBA,
                                      SeasonalNaive)

    train = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount"])
    eval_keys = (pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                                 columns=KEY).drop_duplicates())
    train = train.merge(eval_keys, on=KEY)
    train["unique_id"] = train["store_id"].astype(str) + "_" + train["product_id"].astype(str)
    df = train.rename(columns={"dt": "ds", "sale_amount": "y"})[["unique_id", "ds", "y"]]

    models = [
        CrostonClassic(),
        CrostonSBA(),
        TSB(alpha_d=0.1, alpha_p=0.05),  # 일별 데이터 → 소값 β (Teunter et al. 2011 권고 반영)
        AutoETS(season_length=7),
        SeasonalNaive(season_length=7),
    ]
    sf = StatsForecast(models=models, freq="D", n_jobs=-1)
    t0 = time.time()
    fc = sf.forecast(df=df, h=HORIZON)
    print(f"forecast done in {time.time()-t0:.0f}s", flush=True)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc

    ids = fc["unique_id"].str.split("_", expand=True).astype(int)
    fc["store_id"], fc["product_id"] = ids[0], ids[1]
    fc = fc.sort_values(["store_id", "product_id", "ds"])
    fc["h"] = fc.groupby(KEY).cumcount() + 1

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = {"CrostonClassic": "croston", "CrostonSBA": "sba", "TSB": "tsb",
                "AutoETS": "ets", "SeasonalNaive": "snaive"}
    for col, short in name_map.items():
        rows = fc[KEY + ["h"]].copy()
        rows["q0.5"] = np.clip(fc[col].to_numpy(), 0, None)
        out_path = out_dir / f"{short}_{args.dataset}_raw.parquet"
        rows.to_parquet(out_path, index=False)
        meta = {"model": short, "dataset": args.dataset, "mask_stockout": False,
                "cutoff_date": None, "quantiles": [0.5], "horizon": HORIZON,
                "n_series": rows.groupby(KEY).ngroups, "engine": "statsforecast",
                "point_only": True}
        Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        print("saved:", out_path.name, flush=True)


if __name__ == "__main__":
    main()
