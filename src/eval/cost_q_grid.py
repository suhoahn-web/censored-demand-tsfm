"""P0-1: 비용구조(b)와 발주분위수(q)를 분리한 2차원 격자.

문제: 기존 시나리오 3종은 b와 q를 **동시에** 바꾼다(b=0.3&q=0.5, b=1.0&q=0.7,
b=3.0&q=0.9). 그래서 "경제 파라미터 하나가 권고를 뒤집는다"는 주장이 식별되지 않는다.
q를 고정한 채 b만 바꾸고, b를 고정한 채 q만 바꿔야 각 축의 효과를 분리할 수 있다.

물리적 시뮬레이션 결과(폐기·품절·보유·발주량)는 (모델, q)에만 의존하고 b에는 의존하지
않는다. 따라서 (모델, q)마다 한 번만 시뮬레이션하고, 비용은 임의의 b에 대해 사후
계산한다. 부산물로 **시계열 단위 비용 분해**가 남아 군집 부트스트랩(P0-7)에 쓰인다.

산출:
  results/cost_q_outcomes_{tag}.parquet — run × q × 시계열 물리량
  results/cost_q_grid_{tag}.csv         — run × q × b 집계 비용·서비스

사용: python -m src.eval.cost_q_grid --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import (BURN_IN, HORIZON, TRUTH_OF, collect_predictions,
                                    order_at_cr, simulate)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]

QGRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BGRID = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
HOLD = 0.02   # 단위·일 보유비용 (구매원가 1로 정규화)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--shelf-life", type=int, default=2)
    ap.add_argument("--lead-time", type=int, default=0)
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--demand-path", default="observed",
                    choices=["observed", "uplift", "clean"],
                    help="평가 수요 경로 (P0-5 민감도)")
    args = ap.parse_args()

    if args.demand_path != "observed":
        from src.eval.demand_target_sensitivity import build_paths
        dwide = build_paths(args.dataset)[args.demand_path]
    else:
        truth = pd.read_parquet(
            PROCESSED / f"{TRUTH_OF.get(args.dataset, args.dataset)}_daily_eval.parquet",
            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
        truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
        truth["h"] = truth.groupby(KEY).cumcount() + 1
        if args.clean_only:
            clean = truth.groupby(KEY)["stock_hour6_22_cnt"].max() == 0
            truth = truth.set_index(KEY).loc[clean[clean].index].reset_index()
            print(f"clean subsample: {int(clean.sum())} series", flush=True)
        dwide = truth.pivot_table(index=KEY, columns="h",
                                  values="sale_amount").sort_index()
    demand = dwide.to_numpy()
    dem_scored = demand[:, BURN_IN:].sum(axis=1)   # 시계열별 집계 대상 수요

    per_series, agg = [], []
    for name, pred in collect_predictions(args.dataset):
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
            r = simulate(demand, tgt.to_numpy(), args.shelf_life, args.lead_time)
            per_series.append(pd.DataFrame({
                "run": name, "q": q,
                "store_id": dwide.index.get_level_values(0),
                "product_id": dwide.index.get_level_values(1),
                "demand": dem_scored, "sales": r["sales"], "lost": r["lost"],
                "waste": r["waste"], "hold": r["hold"], "ordered": r["ordered"],
                "n_orders": r["n_orders"]}))
            tot_d = dem_scored.sum()
            for b in BGRID:
                cost = (r["waste"].sum() + r["lost"].sum() * b + r["hold"].sum() * HOLD)
                agg.append({
                    "run": name, "q": q, "b": b, "h_cost": HOLD,
                    "q_star": round(b / (1 + b), 4),
                    "cost_per_unit_demand": float(cost / tot_d),
                    "fill_rate": float(r["sales"].sum() / tot_d),
                    "lost_rate": float(r["lost"].sum() / tot_d),
                    "waste_rate": float(r["waste"].sum() / max(r["ordered"].sum(), 1e-9)),
                    "avg_inventory": float(r["hold"].sum()
                                           / (len(demand) * (HORIZON - BURN_IN))),
                })
        print(f"  simulated: {name}", flush=True)

    suffix = ("_clean" if args.clean_only
              else ("" if args.demand_path == "observed" else f"_{args.demand_path}"))
    tag = f"{args.dataset}{suffix}_sl{args.shelf_life}_L{args.lead_time}"
    out1 = RESULTS / f"cost_q_outcomes_{tag}.parquet"
    pd.concat(per_series, ignore_index=True).to_parquet(out1, index=False)
    grid = pd.DataFrame(agg)
    out2 = RESULTS / f"cost_q_grid_{tag}.csv"
    grid.to_csv(out2, index=False)
    print(f"\nsaved: {out1.name}, {out2.name}")

    # A. q 고정, b 변화 -> 순위가 경제 파라미터만으로 바뀌는가
    print("\n=== A. q 고정, b 변화: 최저비용 모델 ===")
    for q in [0.5, 0.7, 0.9]:
        s = grid[np.isclose(grid.q, q)]
        best = s.loc[s.groupby("b")["cost_per_unit_demand"].idxmin()]
        print(f"q={q}: " + " | ".join(f"b={r.b}:{r.run}" for r in best.itertuples()))

    # B. b 고정, q 변화 -> 순위가 서비스 목표만으로 바뀌는가
    print("\n=== B. b 고정, q 변화: 최저비용 모델 ===")
    for b in [0.3, 1.0, 3.0]:
        s = grid[np.isclose(grid.b, b)]
        best = s.loc[s.groupby("q")["cost_per_unit_demand"].idxmin()]
        print(f"b={b}: " + " | ".join(f"q={r.q}:{r.run}" for r in best.itertuples()))

    # C. 분석적으로 정당화되는 q (뉴스벤더 임계비 b/(1+b))에서의 순위
    print("\n=== C. q = b/(1+b) (뉴스벤더 임계비)에서 최저비용 5개 ===")
    for b in [0.3, 1.0, 3.0]:
        qs = min(QGRID, key=lambda x: abs(x - b / (1 + b)))
        s = (grid[np.isclose(grid.b, b) & np.isclose(grid.q, qs)]
             .nsmallest(5, "cost_per_unit_demand"))
        print(f"\nb={b} -> q*={b/(1+b):.3f} (격자 {qs})")
        print(s[["run", "cost_per_unit_demand", "fill_rate", "waste_rate"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
