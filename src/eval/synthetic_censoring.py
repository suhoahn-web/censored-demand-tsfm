"""E6-b (JORS R1, Major 1·2): 참수요를 아는 시계열에 인위 검열을 가한 통제 실험.

평가 주간에 품절이 전혀 없는 시계열(LT 340, 50K 5,874)은 7일 전부 참수요를 안다.
이들에 벤치마크의 실제 검열 강도(품절일 비율, 품절일의 품절시간 분포)를 따라
인위 품절을 가해 '검열 판매'를 만들고, 동일한 예측을
  (a) 참수요로 채점/시뮬레이션   (b) 검열 판매로 채점/시뮬레이션
했을 때 정확도·편향·재고비용·기법 순위가 얼마나 달라지는지 측정한다.
예측은 바꾸지 않으므로 이 차이는 순수하게 **평가 타깃의 검열이 만드는 인공물**이다.

검열 기제: 품절은 영업시간(06–22) 중 어느 시각에 시작해 그날 끝까지 지속한다고
가정한다(재고 소진 후 당일 재입고 없음). 품절시간 s를 실제 품절일의 s 분포에서
뽑아 마지막 s 시간의 판매를 0으로 만든다(hours_sale 사용). 실제 데이터에서
품절이 '한 번 시작해 끝까지' 지속한 비율도 함께 보고한다.

산출: results/synthetic_censoring_{ds}.csv (seed×method×q×b×target)
      results/synthetic_censoring_{ds}_summary.csv
      results/synthetic_censoring_{ds}_accuracy.csv

사용: python -m src.eval.synthetic_censoring --dataset lt --seeds 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eval.inventory_sim import (BURN_IN, HORIZON, collect_predictions,
                                    order_at_cr, simulate)
from src.eval.metrics import wape, wpe, wql

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = {"lt": PROJECT_ROOT / "data/raw/FreshRetailNet-LT/data",
       "frn50k": PROJECT_ROOT / "data/raw/FreshRetailNet-50K/data"}
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
OPEN, CLOSE = 6, 22
QGRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BGRID = [0.3, 1.0, 3.0]
PRESPEC = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
HOLD = 0.02


def censoring_profile(ds: str) -> tuple[float, np.ndarray, float]:
    """평가창의 품절일 비율, 품절일 s(1..16) 경험분포, 품절의 '종결형' 비율."""
    ev = pd.read_parquet(RAW[ds] / "eval.parquet",
                         columns=["stock_hour6_22_cnt", "hours_stock_status"])
    cnt = ev["stock_hour6_22_cnt"].to_numpy()
    p_c = float((cnt > 0).mean())
    s = cnt[(cnt > 0) & (cnt <= 16)]
    pmf = np.bincount(s, minlength=17)[1:17].astype(float)
    pmf /= pmf.sum()
    st = np.stack(ev.loc[cnt > 0, "hours_stock_status"].to_numpy())[:, OPEN:CLOSE]
    # 종결형: 상태가 0…0 1…1 (한 번 품절되면 마감까지 지속)
    terminal = ((np.diff(st, axis=1) >= 0).all(axis=1)).mean()
    return p_c, pmf, float(terminal)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shelf-life", type=int, default=2)
    args = ap.parse_args()
    ds = args.dataset

    p_c, pmf, terminal_share = censoring_profile(ds)
    print(f"{ds}: censored-day share {p_c:.3f}, E[s|censored] "
          f"{(pmf * np.arange(1, 17)).sum():.2f} h, terminal-stockout share "
          f"{terminal_share:.3f}", flush=True)

    ev = pd.read_parquet(RAW[ds] / "eval.parquet",
                         columns=KEY + ["dt", "sale_amount", "hours_sale",
                                        "stock_hour6_22_cnt"])
    ev["dt"] = pd.to_datetime(ev["dt"])
    ev = ev.sort_values(KEY + ["dt"]).reset_index(drop=True)
    ev["h"] = ev.groupby(KEY).cumcount() + 1
    clean_idx = ev.groupby(KEY)["stock_hour6_22_cnt"].max() == 0
    clean_keys = clean_idx[clean_idx].index
    ev = ev.set_index(KEY).loc[clean_keys].reset_index()
    hs = np.stack(ev["hours_sale"].to_numpy())  # (rows, 24)
    assert np.allclose(hs.sum(axis=1), ev["sale_amount"].to_numpy(), atol=1e-6)
    true_wide = ev.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    print(f"clean series: {len(true_wide)}", flush=True)

    preds = collect_predictions(ds)
    rows, acc_rows = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        n_rows = len(ev)
        cens = rng.random(n_rows) < p_c
        s = rng.choice(np.arange(1, 17), size=n_rows, p=pmf)
        cut_hour = np.where(cens, CLOSE - s, 24)
        hour_idx = np.arange(24)[None, :]
        y_cens = np.where(hour_idx < cut_hour[:, None], hs, 0.0).sum(axis=1)
        evc = ev[KEY + ["h", "sale_amount"]].copy()
        evc["y_cens"] = y_cens
        evc["is_cens"] = cens
        cens_wide = evc.pivot_table(index=KEY, columns="h", values="y_cens").sort_index()
        targets = {"true": true_wide, "censored": cens_wide}
        print(f"seed {seed}: censored share {cens.mean():.3f}, demand loss "
              f"{1 - y_cens.sum() / ev.sale_amount.sum():.3f}", flush=True)

        for name, pred in preds:
            ps = pred.sort_values(KEY + ["h"])
            m = ps.merge(evc, on=KEY + ["h"], how="inner")
            qc = {float(c[1:]): m[c].to_numpy() for c in m.columns if c.startswith("q")}
            yt, yc = m["sale_amount"].to_numpy(), m["y_cens"].to_numpy()
            unc = ~m["is_cens"].to_numpy()
            acc = {"seed": seed, "run": name}
            for tgt_name, yy, mask in [("true_all", yt, np.ones_like(unc)),
                                       ("censored_all", yc, np.ones_like(unc)),
                                       ("censored_uncensored_days", yc, unc)]:
                qm = {q: v[mask] for q, v in qc.items()}
                acc[f"wape_{tgt_name}"] = wape(yy[mask], qm[0.5])
                acc[f"wpe_{tgt_name}"] = wpe(yy[mask], qm[0.5])
                acc[f"wql_{tgt_name}"] = wql(yy[mask], qm) if len(qm) > 1 else np.nan
            acc_rows.append(acc)

            for q in QGRID:
                flat = order_at_cr(ps, q)
                if flat is None:
                    continue
                tgt = (ps[KEY + ["h"]].assign(v=flat)
                       .pivot_table(index=KEY, columns="h", values="v")
                       .reindex(true_wide.index))
                if tgt.isna().any().any():
                    continue
                for tgt_name, dw in targets.items():
                    demand = dw.to_numpy()
                    tot_d = demand[:, BURN_IN:].sum()
                    r = simulate(demand, tgt.to_numpy(), args.shelf_life, 0)
                    for b in BGRID:
                        cost = r["waste"].sum() + b * r["lost"].sum() + HOLD * r["hold"].sum()
                        rows.append({"seed": seed, "run": name, "q": q, "b": b,
                                     "target": tgt_name,
                                     "prespecified": bool(abs(PRESPEC[b] - q) < 1e-9),
                                     "cost_per_unit_demand": float(cost / tot_d),
                                     "fill_rate": float(r["sales"].sum() / tot_d),
                                     "waste_rate": float(r["waste"].sum()
                                                         / max(r["ordered"].sum(), 1e-9))})
        print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"synthetic_censoring_{ds}.csv", index=False)
    accdf = pd.DataFrame(acc_rows)
    accdf.to_csv(RESULTS / f"synthetic_censoring_{ds}_accuracy.csv", index=False)

    # 요약: 사전지정 셀에서 참/검열 비용, 순위, 순위상관
    pre = df[df.prespecified]
    g = (pre.groupby(["b", "q", "run", "target"])["cost_per_unit_demand"].mean()
         .unstack("target").reset_index())
    g["delta_pct"] = (g["censored"] / g["true"] - 1) * 100
    g["rank_true"] = g.groupby("b")["true"].rank()
    g["rank_censored"] = g.groupby("b")["censored"].rank()
    summ_rows = []
    for b, s in g.groupby("b"):
        rho = spearmanr(s["true"], s["censored"]).correlation
        summ_rows.append({"b": b, "q": PRESPEC[b], "spearman_true_vs_censored": rho,
                          "best_true": s.sort_values("true").run.iloc[0],
                          "best_censored": s.sort_values("censored").run.iloc[0],
                          "mean_delta_pct": s.delta_pct.mean(),
                          "max_abs_rank_shift": (s.rank_true - s.rank_censored).abs().max()})
    summ = pd.DataFrame(summ_rows)
    meta = pd.DataFrame([{"dataset": ds, "n_clean_series": len(true_wide),
                          "censored_day_share": p_c, "terminal_stockout_share": terminal_share,
                          "seeds": args.seeds}])
    g.to_csv(RESULTS / f"synthetic_censoring_{ds}_summary.csv", index=False)
    meta.to_csv(RESULTS / f"synthetic_censoring_{ds}_meta.csv", index=False)
    print("\n=== 사전지정 셀: 참수요 vs 검열판매 ===")
    print(summ.round(3).to_string(index=False))
    print("\n=== b=3, q=0.9 기법별 ===")
    print(g[np.isclose(g.b, 3.0)].sort_values("true")
          [["run", "true", "censored", "delta_pct", "rank_true", "rank_censored"]]
          .round(4).to_string(index=False))
    a = accdf.groupby("run").mean(numeric_only=True)
    print("\n=== 정확도: 참수요(all) vs 검열(all) vs 검열(비검열일 프로토콜) ===")
    print(a[["wape_true_all", "wape_censored_all", "wape_censored_uncensored_days",
             "wpe_true_all", "wpe_censored_all", "wpe_censored_uncensored_days"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
