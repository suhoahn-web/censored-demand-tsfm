"""E5 (JORS R1, Major 5): 분위수 수준별 pinball loss와 커버리지.

WQL은 전 수준 평균이라 상단 꼬리의 품질을 따로 보여주지 못한다. 공식 origin의
모든 분위수 보유 기법에 대해 수준 q마다 (i) 정규화 pinball loss 2·mean(ρ_q)/mean(y),
(ii) 경험적 커버리지 P(y ≤ ŷ_q), (iii) 커버리지 오차를 비검열일에서 계산한다.

산출: results/quantile_losses_{ds}.csv (long), results/quantile_losses_{ds}_wide.csv

사용: python -m src.eval.quantile_losses --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import collect_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TRUTH_OF = {"frn50krec": "frn50k"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    args = ap.parse_args()

    truth = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    scored = truth[truth.stock_hour6_22_cnt == 0][KEY + ["h", "sale_amount"]]

    rows = []
    for name, pred in collect_predictions(args.dataset):
        qcols = sorted((float(c[1:]), c) for c in pred.columns if c.startswith("q"))
        if len(qcols) < 3:
            continue
        m = pred.merge(scored, on=KEY + ["h"], how="inner")
        y = m["sale_amount"].to_numpy()
        ybar = np.abs(y).mean()
        for q, c in qcols:
            yq = m[c].to_numpy()
            e = y - yq
            pin = np.maximum(q * e, (q - 1) * e).mean()
            rows.append({"run": name, "q": q, "n_obs": len(y),
                         "pinball_norm": float(2 * pin / ybar),
                         "coverage": float((y <= yq).mean()),
                         "coverage_error": float((y <= yq).mean() - q)})
        print(f"  {name}: {len(qcols)} levels", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"quantile_losses_{args.dataset}.csv", index=False)
    wide = df.pivot(index="run", columns="q", values="pinball_norm")
    cov = df.pivot(index="run", columns="q", values="coverage")
    wide.columns = [f"pin_{q}" for q in wide.columns]
    cov.columns = [f"cov_{q}" for q in cov.columns]
    out = wide.join(cov)
    out.to_csv(RESULTS / f"quantile_losses_{args.dataset}_wide.csv")
    show = [c for c in out.columns if c.split("_")[1] in ("0.5", "0.7", "0.9")]
    print(out[show].sort_values("pin_0.9").round(4).to_string())


if __name__ == "__main__":
    main()
