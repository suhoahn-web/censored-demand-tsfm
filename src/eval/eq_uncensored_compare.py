"""E4 (JORS R1, Major 5): 경험적 분위수의 검증 잔차에서 검열일을 제외하면 무엇이 바뀌나.

기본 EQ(eq)는 검증창의 모든 날 잔차를 쓰고(LT 검증창 품절일 49%), 변형(equ)은
품절시간 0인 날의 잔차만 쓴다. 두 변형을 정확도(비검열일)·커버리지·재고비용
(사전지정 셀)에서 나란히 비교한다.

산출: results/eq_uncensored_compare_{ds}.csv
사용: python -m src.eval.eq_uncensored_compare --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import BURN_IN, order_at_cr, simulate
from src.eval.metrics import crps_from_quantiles, wape, wpe, wql

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
PRESPEC = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
HOLD = 0.02
MODELS = ["croston", "sba", "tsb", "ets", "snaive"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    args = ap.parse_args()
    ds = args.dataset

    truth = pd.read_parquet(PROCESSED / f"{ds}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    scored = truth[truth.stock_hour6_22_cnt == 0][KEY + ["h", "sale_amount"]]
    dwide = truth.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    demand = dwide.to_numpy()
    tot_d = demand[:, BURN_IN:].sum()

    rows = []
    for short in MODELS:
        for variant in ["eq", "equ"]:
            f = RESULTS / "zeroshot" / f"{short}{variant}_{ds}_raw.parquet"
            if not f.exists():
                continue
            meta = json.loads(Path(str(f) + ".meta.json").read_text())
            pred = pd.read_parquet(f)
            m = pred.merge(scored, on=KEY + ["h"], how="inner")
            y = m["sale_amount"].to_numpy()
            qc = {float(c[1:]): m[c].to_numpy() for c in m.columns if c.startswith("q")}
            rec = {"model": short, "variant": variant,
                   "validation_rows": meta.get("validation_rows"),
                   "wape": wape(y, qc[0.5]), "wpe": wpe(y, qc[0.5]),
                   "wql": wql(y, qc), "crps": crps_from_quantiles(y, qc),
                   "cov0.7": float((y <= qc[0.7]).mean()),
                   "cov0.9": float((y <= qc[0.9]).mean())}
            ps = pred.sort_values(KEY + ["h"])
            for b, q in PRESPEC.items():
                flat = order_at_cr(ps, q)
                tgt = (ps[KEY + ["h"]].assign(v=flat)
                       .pivot_table(index=KEY, columns="h", values="v")
                       .reindex(dwide.index))
                r = simulate(demand, tgt.to_numpy(), 2, 0)
                cost = r["waste"].sum() + b * r["lost"].sum() + HOLD * r["hold"].sum()
                rec[f"cost_b{b}_q{q}"] = float(cost / tot_d)
                rec[f"fill_b{b}_q{q}"] = float(r["sales"].sum() / tot_d)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"eq_uncensored_compare_{ds}.csv", index=False)
    print(df.round(4).to_string(index=False))
    d = df.pivot(index="model", columns="variant")
    print("\nequ - eq (differences):")
    for col in ["wape", "wpe", "wql", "cov0.9", "cost_b1.0_q0.7", "cost_b3.0_q0.9"]:
        print(f"  {col}: " + ", ".join(f"{i}={d[col]['equ'][i]-d[col]['eq'][i]:+.4f}"
                                       for i in d.index))


if __name__ == "__main__":
    main()
