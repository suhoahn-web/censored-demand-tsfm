"""E1 (JORS R1, Major 6/2): 다중 forecast origin에서의 재고 시뮬레이션.

8개 origin(2024-07-07 ~ 2025-06-08)에서 저장된 zero-shot·고전(+EQ) 예측을
재고 시뮬레이션에 넣어 (i) 사전지정 셀(b,q)에서의 비용 순위가 origin 간에 얼마나
안정적인지, (ii) RQ1의 속성 이동(τ(|WPE|,cost) 하락·τ(WQL,cost) 상승)이 origin마다
재현되는지 본다. 세 평가수요 경로(observed/uplift/clean)를 모두 돌린다.

정답은 train 패널에서 origin 다음 7일로 구성한다(공식 평가주간은 쓰지 않는다).
같은 origin 안에서는 모든 기법에 공통인 시계열만 쓴다(공정 비교).

산출:
  results/multi_origin_inventory_{ds}.csv        — origin × path × method × q × b
  results/multi_origin_inventory_rank_{ds}.csv   — 사전지정 셀 순위 요약
  results/multi_origin_tau_{ds}.csv              — origin별 τ(속성, 비용) by q (b=1)

사용: python -m src.eval.multi_origin_inventory --dataset lt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.eval.inventory_sim import BURN_IN, HORIZON, order_at_cr, simulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TAG_RE = re.compile(r"_mo(\d{8})_")
QGRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BGRID = [0.3, 1.0, 3.0]
PRESPEC = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
HOLD = 0.02
TRADING_HOURS = 16
UPLIFT_CAP = 3.0


def truth_paths(dataset: str, cutoff: str) -> dict[str, pd.DataFrame]:
    """origin 다음 7일 정답을 세 수요 경로로 (wide: series × h)."""
    tr = pd.read_parquet(PROCESSED / f"{dataset}_daily_train.parquet",
                         columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    cut = pd.Timestamp(cutoff)
    t = tr[(tr.dt > cut) & (tr.dt <= cut + pd.Timedelta(days=HORIZON))]
    t = t.sort_values(KEY + ["dt"]).copy()
    t["h"] = t.groupby(KEY).cumcount() + 1
    cnt = t.groupby(KEY)["h"].transform("size")
    t = t[cnt == HORIZON]

    clean_mean = (t[t.stock_hour6_22_cnt == 0].groupby(KEY)["sale_amount"].mean()
                  .rename("clean_mean"))
    t = t.join(clean_mean, on=KEY)
    t["clean_mean"] = t["clean_mean"].fillna(
        t.groupby(KEY)["sale_amount"].transform("mean"))
    s = t["stock_hour6_22_cnt"].clip(0, TRADING_HOURS).to_numpy()
    y = t["sale_amount"].to_numpy()
    open_h = TRADING_HOURS - s
    up = np.where(open_h > 0, y * TRADING_HOURS / np.maximum(open_h, 1e-9),
                  t["clean_mean"].to_numpy())
    up = np.minimum(up, np.maximum(UPLIFT_CAP * t["clean_mean"].to_numpy(), y))
    t["uplift"] = np.maximum(up, y)

    obs = t.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    upl = t.pivot_table(index=KEY, columns="h", values="uplift").sort_index()
    clean_idx = (t.groupby(KEY)["stock_hour6_22_cnt"].max() == 0)
    clean = obs.loc[clean_idx[clean_idx].index]
    from src.eval.demand_paths import profile_path_origin
    prof, _ = profile_path_origin(dataset, cutoff)
    return {"observed": obs, "uplift": upl, "clean": clean,
            "profile": prof.reindex(obs.index)}


def load_origin_runs(dataset: str) -> dict[str, list[tuple[str, Path]]]:
    runs: dict[str, list] = {}
    for meta_path in sorted((RESULTS / "zeroshot").glob("*_mo*_*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("dataset") != dataset or not meta.get("cutoff_date"):
            continue
        name = meta_path.name.replace(".parquet.meta.json", "")
        if not TAG_RE.search(name):
            continue
        method = TAG_RE.sub("_", name)
        runs.setdefault(meta["cutoff_date"], []).append(
            (method, Path(str(meta_path).replace(".meta.json", ""))))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--shelf-life", type=int, default=2)
    args = ap.parse_args()

    runs = load_origin_runs(args.dataset)
    print(f"{len(runs)} origins", flush=True)
    rows = []
    for cutoff in sorted(runs):
        paths = truth_paths(args.dataset, cutoff)
        preds = {m: pd.read_parquet(p) for m, p in runs[cutoff]}
        # 공통 시계열: 모든 기법 × 정답에 존재
        common = paths["observed"].index
        for m, p in preds.items():
            idx = pd.MultiIndex.from_frame(p[KEY].drop_duplicates())
            common = common.intersection(idx)
        print(f"origin {cutoff}: {len(preds)} methods, {len(common)} common series",
              flush=True)
        for path_name, dwide in paths.items():
            dw = dwide.loc[dwide.index.intersection(common)].sort_index()
            demand = dw.to_numpy()
            tot_d = demand[:, BURN_IN:].sum()
            for method, pred in preds.items():
                ps = pred.sort_values(KEY + ["h"])
                for q in QGRID:
                    flat = order_at_cr(ps, q)
                    if flat is None:
                        continue
                    tgt = (ps[KEY + ["h"]].assign(v=flat)
                           .pivot_table(index=KEY, columns="h", values="v")
                           .reindex(dw.index))
                    if tgt.isna().any().any():
                        continue
                    r = simulate(demand, tgt.to_numpy(), args.shelf_life, 0)
                    n_days = demand.shape[0] * (demand.shape[1] - BURN_IN)
                    for b in BGRID:
                        cost = (r["waste"].sum() + b * r["lost"].sum()
                                + HOLD * r["hold"].sum())
                        rows.append({
                            "origin": cutoff, "demand_path": path_name,
                            "method": method, "q": q, "b": b,
                            "prespecified": bool(abs(PRESPEC[b] - q) < 1e-9),
                            "n_series": len(demand),
                            "cost_per_unit_demand": float(cost / tot_d),
                            "fill_rate": float(r["sales"].sum() / tot_d),
                            "lost_rate": float(r["lost"].sum() / tot_d),
                            "waste_rate": float(r["waste"].sum()
                                                / max(r["ordered"].sum(), 1e-9)),
                            "avg_inventory": float(r["hold"].sum() / n_days)})
            print(f"  path {path_name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"multi_origin_inventory_{args.dataset}.csv", index=False)

    # ── 사전지정 셀 순위 안정성 (observed 경로) ──
    pre = df[df.prespecified & (df.demand_path == "observed")].copy()
    pre["rank"] = pre.groupby(["origin", "b"])["cost_per_unit_demand"].rank()
    n_or = pre.origin.nunique()
    summ = (pre.groupby(["b", "q", "method"])
            .agg(cost_mean=("cost_per_unit_demand", "mean"),
                 cost_sd=("cost_per_unit_demand", "std"),
                 rank_mean=("rank", "mean"), rank_sd=("rank", "std"),
                 rank_min=("rank", "min"), rank_max=("rank", "max"),
                 n_origins=("origin", "nunique"))
            .reset_index())
    top1 = (pre[pre["rank"] == 1].groupby(["b", "method"]).size() / n_or
            ).rename("top1_freq").reset_index()
    summ = summ.merge(top1, on=["b", "method"], how="left").fillna({"top1_freq": 0.0})
    # 경로 간 순위 상관 (origin별, b=3/q=0.9)
    summ.to_csv(RESULTS / f"multi_origin_inventory_rank_{args.dataset}.csv", index=False)

    # ── origin별 τ(속성, 비용) by q ──
    acc = pd.read_csv(RESULTS / f"multi_origin_{args.dataset}.csv")
    acc["abs_wpe"] = acc.wpe.abs()
    taus = []
    for (origin, path_name, b, q), s in df.groupby(["origin", "demand_path", "b", "q"]):
        s = s.merge(acc[["origin", "method", "wape", "wpe", "abs_wpe", "wql"]],
                    on=["origin", "method"])
        s = s[s.method.isin(s.method)]  # no-op; keeps explicit
        if len(s) < 5:
            continue
        rec = {"origin": origin, "demand_path": path_name, "b": b, "q": q,
               "n_methods": len(s)}
        for prop in ["abs_wpe", "wpe", "wql", "wape"]:
            tau, p = kendalltau(s[prop], s.cost_per_unit_demand)
            rec[f"tau_{prop}"] = tau
            rec[f"p_{prop}"] = p
        taus.append(rec)
    tau_df = pd.DataFrame(taus)
    tau_df.to_csv(RESULTS / f"multi_origin_tau_{args.dataset}.csv", index=False)

    print("\n=== 사전지정 셀 순위 (observed) ===")
    for b in BGRID:
        s = summ[np.isclose(summ.b, b)].sort_values("rank_mean")
        print(f"\n[b={b}, q={PRESPEC[b]}]")
        print(s[["method", "cost_mean", "cost_sd", "rank_mean", "rank_sd",
                 "rank_min", "rank_max", "top1_freq"]].round(4).to_string(index=False))
    print("\n=== tau(|WPE|,cost) / tau(WPE,cost) / tau(WQL,cost) by q, b=1, observed, per origin ===")
    t1 = tau_df[(tau_df.demand_path == "observed") & np.isclose(tau_df.b, 1.0)]
    for col in ["tau_abs_wpe", "tau_wpe", "tau_wql"]:
        print(col)
        print(t1.pivot(index="origin", columns="q", values=col).round(2).to_string())


if __name__ == "__main__":
    main()
