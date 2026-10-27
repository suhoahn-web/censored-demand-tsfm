"""P0-5: 평가 수요 경로에 대한 민감도.

문제: 재고 시뮬레이션의 비용은 수요에 의존하는데, 품절이 있었던 평가일의 참 수요는
관측되지 않는다. 관측 판매량을 수요로 쓰면 품절 비용이 과소평가된다. 따라서 결과를
"참 운영비용"이라고 부를 수 없고, 수요 경로를 바꿔 순위가 유지되는지 봐야 한다.

세 가지 평가 수요 경로:
  observed  관측 판매량 그대로 (검열됨; 하한 성격)
  uplift    품절 시간 비례 보정 y x 16/(16-s). 영업 16시간 동안 시간당 수요율이
            일정하다는 가정하의 단순 상한. s=16(종일 품절)이면 그 시계열의 무품절일
            평균으로 대체하고, 폭주를 막기 위해 무품절일 평균의 3배로 절단한다.
  clean     평가 주간에 품절이 전혀 없던 시계열만 (선택 표본; 검열 없음)

어느 것도 참값이 아니다. uplift는 추정 잠재수요 민감도 분석이지 정답이 아니다.

산출: results/demand_target_sensitivity_{dataset}.csv

사용: python -m src.eval.demand_target_sensitivity --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import (BURN_IN, HORIZON, collect_predictions,
                                    order_at_cr, simulate)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TRADING_HOURS = 16
QGRID = [0.5, 0.7, 0.9]
BGRID = [0.3, 1.0, 3.0]
HOLD = 0.02
UPLIFT_CAP = 3.0


def build_paths(dataset: str) -> dict[str, pd.DataFrame]:
    t = pd.read_parquet(PROCESSED / f"{dataset}_daily_eval.parquet",
                        columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    t = t.sort_values(KEY + ["dt"]).reset_index(drop=True)
    t["h"] = t.groupby(KEY).cumcount() + 1

    # 무품절일 평균 (uplift의 절단 기준·종일 품절 대체값)
    clean_mean = (t[t.stock_hour6_22_cnt == 0].groupby(KEY)["sale_amount"].mean()
                  .rename("clean_mean"))
    t = t.join(clean_mean, on=KEY)
    t["clean_mean"] = t["clean_mean"].fillna(t.groupby(KEY)["sale_amount"]
                                             .transform("mean"))

    s = t["stock_hour6_22_cnt"].clip(0, TRADING_HOURS).to_numpy()
    y = t["sale_amount"].to_numpy()
    open_h = TRADING_HOURS - s
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(open_h > 0, y * TRADING_HOURS / np.maximum(open_h, 1e-9),
                      t["clean_mean"].to_numpy())
    up = np.minimum(up, np.maximum(UPLIFT_CAP * t["clean_mean"].to_numpy(), y))
    t["uplift"] = np.maximum(up, y)

    paths = {}
    paths["observed"] = t.pivot_table(index=KEY, columns="h", values="sale_amount")
    paths["uplift"] = t.pivot_table(index=KEY, columns="h", values="uplift")
    clean_idx = (t.groupby(KEY)["stock_hour6_22_cnt"].max() == 0)
    paths["clean"] = paths["observed"].loc[clean_idx[clean_idx].index]
    for k in paths:
        paths[k] = paths[k].sort_index()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--shelf-life", type=int, default=2)
    args = ap.parse_args()

    paths = build_paths(args.dataset)
    for k, v in paths.items():
        print(f"{k:9s} series={len(v):,}  total demand={v.to_numpy()[:, BURN_IN:].sum():,.0f}")

    preds = collect_predictions(args.dataset)
    rows = []
    for path_name, dwide in paths.items():
        demand = dwide.to_numpy()
        tot_d = demand[:, BURN_IN:].sum()
        for name, pred in preds:
            ps = pred.sort_values(KEY + ["h"])
            for q in QGRID:
                flat = order_at_cr(ps, q)
                if flat is None:
                    continue
                tgt = (ps[KEY + ["h"]].assign(v=flat)
                       .pivot_table(index=KEY, columns="h", values="v")
                       .reindex(dwide.index))
                if tgt.isna().any().any():
                    continue
                r = simulate(demand, tgt.to_numpy(), args.shelf_life, 0)
                for b in BGRID:
                    cost = r["waste"].sum() + b * r["lost"].sum() + HOLD * r["hold"].sum()
                    rows.append({"demand_path": path_name, "run": name, "q": q, "b": b,
                                 "n_series": len(demand),
                                 "cost_per_unit_demand": float(cost / tot_d),
                                 "fill_rate": float(r["sales"].sum() / tot_d)})
        print(f"  path done: {path_name}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / f"demand_target_sensitivity_{args.dataset}.csv"
    df.to_csv(out, index=False)
    print("saved:", out.name)

    # 경로 간 순위 상관 (Spearman) — 순위가 유지되는가
    print("\n=== 수요 경로 간 비용 순위 상관 (Spearman) ===")
    for q in QGRID:
        for b in BGRID:
            s = df[np.isclose(df.q, q) & np.isclose(df.b, b)]
            pv = s.pivot(index="run", columns="demand_path",
                         values="cost_per_unit_demand").dropna()
            if pv.shape[1] < 2:
                continue
            c = pv.rank().corr(method="spearman")
            print(f"q={q}, b={b}: observed-uplift {c.loc['observed','uplift']:.3f}"
                  + (f", observed-clean {c.loc['observed','clean']:.3f}"
                     if "clean" in c.columns else ""))

    print("\n=== b=3.0, q=0.9 상위 5개 (경로별) ===")
    for path_name in paths:
        s = (df[(df.demand_path == path_name) & np.isclose(df.b, 3.0)
                & np.isclose(df.q, 0.9)].nsmallest(5, "cost_per_unit_demand"))
        print(f"[{path_name}] " + ", ".join(
            f"{r.run}({r.cost_per_unit_demand:.4f})" for r in s.itertuples()))


if __name__ == "__main__":
    main()
