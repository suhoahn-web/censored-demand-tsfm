"""E9 (JORS R1, Minor 10): 편향 정의·달성 서비스·MASE 가중 진단.

(1) 중앙값 부호오차 vs 평균 편향: WPE는 q0.5 기준이다. 우왜도 수요에서는 검열이
    없어도 중앙값 < 평균이므로 음의 WPE 일부는 검열이 아니라 분포 비대칭이다.
    분위수 격자에서 사다리꼴로 근사한 예측평균(grid mean)과 LightGBM Tweedie 평균
    으로 '평균 기반 WPE'를 병기한다.
(2) 명목 발주 분위수 vs 달성 서비스: 분위수 q마다 fill rate(수량 기준)와
    cycle service level(품절 없는 시계열-일 비율)을 기법별로 보고한다.
(3) MASE: 시계열별 비검열 평가일 수가 1~7로 다르므로, 비검열일 수로 가중한 MASE와
    등가중 MASE를 병기한다. MASE 분모(계절차분 평균) 0 건수와 비검열일 0 시계열 수를
    기록한다.

산출: results/bias_mean_vs_median_{ds}.csv, results/achieved_service_{ds}.csv,
      results/mase_weighting_{ds}.csv, results/eval_day_counts_{ds}.csv

사용: python -m src.eval.forecast_diagnostics --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import (BURN_IN, HORIZON, collect_predictions,
                                    order_at_cr, simulate)
from src.eval.metrics import mase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
QGRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def grid_mean(pred: pd.DataFrame, qcols: list[tuple[float, str]]) -> np.ndarray:
    """분위수 격자 → 예측분포 평균 근사 (사다리꼴; 양 끝은 최외곽 분위수로 연장)."""
    qs = np.array([q for q, _ in qcols])
    Q = np.stack([pred[c].to_numpy() for _, c in qcols], axis=1)
    # 외곽: [0, q_min]은 Q_min, [q_max, 1]은 Q_max로 상수 연장 (보수적)
    mean = Q[:, 0] * qs[0] + Q[:, -1] * (1 - qs[-1])
    mean += ((Q[:, 1:] + Q[:, :-1]) / 2 * np.diff(qs)).sum(axis=1)
    return mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    args = ap.parse_args()
    ds = args.dataset

    truth = pd.read_parquet(PROCESSED / f"{ds}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    scored = truth[truth.stock_hour6_22_cnt == 0][KEY + ["h", "sale_amount"]]
    insample = pd.read_parquet(PROCESSED / f"{ds}_daily_train.parquet",
                               columns=KEY + ["dt", "sale_amount"])
    insample = insample.merge(truth[KEY].drop_duplicates(), on=KEY)
    ins_g = dict(list(insample.sort_values(KEY + ["dt"]).groupby(KEY)["sale_amount"]))

    # (3a) 평가일 수·MASE 분모 진단
    n_unc = scored.groupby(KEY).size().rename("n_uncensored_days")
    all_keys = truth[KEY].drop_duplicates().set_index(KEY).index
    n_unc = n_unc.reindex(all_keys).fillna(0).astype(int)
    scale = pd.Series({k: np.abs(v.to_numpy()[7:] - v.to_numpy()[:-7]).mean()
                       if len(v) > 7 else np.nan for k, v in ins_g.items()})
    scale.index.names = KEY
    diag = pd.DataFrame({"n_uncensored_days": n_unc,
                         "mase_scale": scale.reindex(all_keys)})
    diag.to_csv(RESULTS / f"eval_day_counts_{ds}.csv")
    print(f"series: {len(diag)} | zero uncensored days: {(diag.n_uncensored_days == 0).sum()}"
          f" | MASE scale NaN: {diag.mase_scale.isna().sum()} | scale==0: "
          f"{(diag.mase_scale == 0).sum()}", flush=True)
    print("uncensored-day distribution:",
          diag.n_uncensored_days.value_counts().sort_index().to_dict(), flush=True)

    dwide = truth.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    demand = dwide.to_numpy()
    tot_d = demand[:, BURN_IN:].sum()
    n_sd = demand.shape[0] * (demand.shape[1] - BURN_IN)

    bias_rows, serv_rows, mase_rows = [], [], []
    for name, pred in collect_predictions(ds):
        qcols = sorted((float(c[1:]), c) for c in pred.columns if c.startswith("q"))
        m = pred.merge(scored, on=KEY + ["h"], how="inner")
        y = m["sale_amount"].to_numpy()
        med = m["q0.5"].to_numpy()
        rec = {"run": name, "wpe_median": float((med - y).sum() / y.sum()),
               "n_levels": len(qcols)}
        if len(qcols) >= 3:
            gm = grid_mean(m, qcols)
            rec["wpe_grid_mean"] = float((gm - y).sum() / y.sum())
            rec["median_minus_mean_share"] = float((med - gm).sum() / y.sum())
        if "tweedie_point" in m.columns:
            tp = m["tweedie_point"].to_numpy()
            rec["wpe_tweedie_mean"] = float((tp - y).sum() / y.sum())
        bias_rows.append(rec)

        # MASE 등가중 vs 비검열일 수 가중
        vals, wts = [], []
        for k, g in m.groupby(KEY):
            if k in ins_g:
                v = mase(g["sale_amount"].to_numpy(), g["q0.5"].to_numpy(),
                         ins_g[k].to_numpy())
                vals.append(v)
                wts.append(len(g))
        vals, wts = np.array(vals), np.array(wts, dtype=float)
        ok = ~np.isnan(vals)
        mase_rows.append({"run": name, "mase_equal": float(vals[ok].mean()),
                          "mase_day_weighted": float((vals[ok] * wts[ok]).sum() / wts[ok].sum()),
                          "n_series": int(ok.sum()), "n_nan": int((~ok).sum())})

        # 달성 서비스
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
            # cycle service level: 품절(lost>0) 없는 시계열-일 비율 — simulate는
            # 합계만 돌려주므로 일 단위로 다시 계산한다
            r = simulate(demand, tgt.to_numpy(), 2, 0)
            # 일 단위 품절 여부: 간단히 재시뮬 (vectorised, 작은 비용)
            n, T = demand.shape
            inv = np.zeros((n, 2)); stockout_days = 0; counted = 0
            for t in range(T):
                pos = inv.sum(axis=1)
                order = np.maximum(tgt.to_numpy()[:, t] - pos, 0.0)
                inv[:, -1] += order
                rem = demand[:, t].copy()
                for a in range(2):
                    take = np.minimum(inv[:, a], rem); inv[:, a] -= take; rem -= take
                if t >= BURN_IN:
                    stockout_days += (rem > 1e-9).sum(); counted += n
                inv[:, :-1] = inv[:, 1:]; inv[:, -1] = 0.0
            serv_rows.append({"run": name, "q": q,
                              "fill_rate": float(r["sales"].sum() / tot_d),
                              "cycle_service_level": float(1 - stockout_days / counted),
                              "waste_rate": float(r["waste"].sum() / max(r["ordered"].sum(), 1e-9)),
                              "avg_inventory": float(r["hold"].sum() / n_sd)})
        print(f"  {name}", flush=True)

    pd.DataFrame(bias_rows).to_csv(RESULTS / f"bias_mean_vs_median_{ds}.csv", index=False)
    pd.DataFrame(serv_rows).to_csv(RESULTS / f"achieved_service_{ds}.csv", index=False)
    pd.DataFrame(mase_rows).to_csv(RESULTS / f"mase_weighting_{ds}.csv", index=False)
    print("\n=== 편향: 중앙값 vs 격자평균 ===")
    print(pd.DataFrame(bias_rows).round(4).to_string(index=False))
    print("\n=== 달성 서비스 (q=0.9) ===")
    s = pd.DataFrame(serv_rows)
    print(s[np.isclose(s.q, 0.9)].sort_values("fill_rate").round(4).to_string(index=False))
    print("\n=== MASE 가중 ===")
    print(pd.DataFrame(mase_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
