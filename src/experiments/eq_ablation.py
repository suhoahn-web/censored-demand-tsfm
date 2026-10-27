"""P0-3: 경험적 분위수(EQ) 처리의 분해 — 재중심화 효과와 분산 부여 효과를 분리한다.

문제: 현재 EQ 절차는 q_pred = point + scale x Q(q,h)로, Q(0.5,h)가 0이 아니므로
**중앙값을 옮기는 동시에 분포를 부여한다**. 따라서 "파운데이션 모델 이점의 대부분은
예측분포를 갖는 데서 온다"는 주장을 이 비교만으로는 식별할 수 없다.

네 가지 변형을 비교한다.
  A point   : 원 점예측 (변형 없음)
  B recenter: 모든 분위수 = point + scale x Q(0.5,h)  — 중심만 이동, 분산 없음
  C spread  : q_j = point + scale x (Q(q_j,h) - Q(0.5,h)) — 원 점예측을 중앙값으로
              유지한 채 분산만 부여
  D full EQ : 현행 절차 (B + C 동시)

산출:
  results/eqabl/{model}_{variant}_{dataset}.parquet
  results/eq_ablation_{dataset}.csv

사용: python -m src.experiments.eq_ablation --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import BURN_IN, HORIZON, order_at_cr, simulate
from src.eval.metrics import crps_from_quantiles, wape, wpe, wql
from src.experiments.add_empirical_quantiles import (MODEL_COLS, QUANTILES,
                                                     validation_errors)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
QGRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BGRID = [0.3, 1.0, 3.0]
# 사전지정 발주 분위수 (본 논문 §4.5의 고정 분위수 설계 — 테스트셋 선택 아님)
PRESPECIFIED_Q = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
HOLD = 0.02


def build_variants(short: str, col: str, val: pd.DataFrame, scale: pd.Series,
                   dataset: str) -> dict[str, pd.DataFrame]:
    """A/B/C/D 네 변형의 예측 테이블을 만든다."""
    src = RESULTS / "zeroshot" / f"{short}_{dataset}_raw.parquet"
    if not src.exists():
        return {}
    pred = pd.read_parquet(src).join(scale, on=KEY)
    point = pred["q0.5"].to_numpy()
    sc = pred["scale"].to_numpy()
    hs = pred["h"].to_numpy()

    e = ((val["sale_amount"] - val[col]) / val["scale"]).to_numpy()
    vh = val["h"].to_numpy()
    qtab = {h: np.nanquantile(e[vh == h], QUANTILES) for h in range(1, HORIZON + 1)}
    med_j = QUANTILES.index(0.5)
    med = np.array([qtab[h][med_j] for h in hs])

    base = pred[KEY + ["h"]].copy()
    out = {}

    a = base.copy()
    a["q0.5"] = np.clip(point, 0, None)
    out["A_point"] = a

    b = base.copy()
    b["q0.5"] = np.clip(point + sc * med, 0, None)
    out["B_recenter"] = b

    for tag, centred in [("C_spread", True), ("D_fullEQ", False)]:
        d = base.copy()
        for j, q in enumerate(QUANTILES):
            adj = np.array([qtab[h][j] for h in hs])
            if centred:
                adj = adj - med
            d[f"q{q}"] = np.clip(point + sc * adj, 0, None)
        qc = [f"q{q}" for q in QUANTILES]
        d[qc] = np.sort(d[qc].to_numpy(), axis=1)   # 분위수 교차 방지
        out[tag] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    ap.add_argument("--shelf-life", type=int, default=2)
    args = ap.parse_args()

    truth = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    scored = truth[truth.stock_hour6_22_cnt == 0][KEY + ["h", "sale_amount"]]
    dwide = truth.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    demand = dwide.to_numpy()
    dem_tot = demand[:, BURN_IN:].sum()

    val, scale = validation_errors(args.dataset)
    val = val.join(scale, on=KEY)
    print(f"validation rows: {len(val):,}", flush=True)

    outdir = RESULTS / "eqabl"
    outdir.mkdir(exist_ok=True)
    rows = []
    for short, col in MODEL_COLS.items():
        variants = build_variants(short, col, val, scale, args.dataset)
        if not variants:
            print(f"skip {short} (no point predictions)")
            continue
        for tag, pred in variants.items():
            pred.to_parquet(outdir / f"{short}_{tag}_{args.dataset}.parquet",
                            index=False)
            m = pred.merge(scored, on=KEY + ["h"], how="inner")
            y = m["sale_amount"].to_numpy()
            qc = {float(c[1:]): m[c].to_numpy() for c in m.columns if c.startswith("q")}
            rec = {"model": short, "variant": tag,
                   "wape": wape(y, qc[0.5]), "wpe": wpe(y, qc[0.5]),
                   "wql": wql(y, qc), "crps": crps_from_quantiles(y, qc),
                   "n_quantiles": len(qc)}
            for lvl in (0.7, 0.9):
                rec[f"cov{lvl}"] = (float((y <= qc[lvl]).mean())
                                    if lvl in qc else np.nan)
            # 재고 시뮬레이션
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
                n_days = demand.shape[0] * (demand.shape[1] - BURN_IN)
                for b in BGRID:
                    cost = r["waste"].sum() + b * r["lost"].sum() + HOLD * r["hold"].sum()
                    rows.append({**rec, "q": q, "b": b,
                                 "cost_per_unit_demand": float(cost / dem_tot),
                                 "fill_rate": float(r["sales"].sum() / dem_tot),
                                 "lost_rate": float(r["lost"].sum() / dem_tot),
                                 "waste_rate": float(r["waste"].sum()
                                                    / max(r["ordered"].sum(), 1e-9)),
                                 "avg_inventory": float(r["hold"].sum() / n_days)})
            print(f"  {short} {tag}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / f"eq_ablation_{args.dataset}.csv"
    df.to_csv(out, index=False)
    print("\nsaved:", out.name)

    print("\n=== 예측 지표 (변형별, 모델 평균) ===")
    acc = (df.drop_duplicates(["model", "variant"])
             .groupby("variant")[["wape", "wpe", "wql", "cov0.9"]].mean())
    print(acc.round(4).to_string())

    # 주의: 변형별 최소비용 q(idxmin) 요약은 테스트셋 q 선택이라 본문 근거로 쓰지
    # 않는다 (eq_ablation_q_selection_audit.md 참조). 본문 수치는 사전지정 분위수
    # PRESPECIFIED_Q에서 계산한다. A/B는 q=0.5(중앙값) 단일 운영점만 갖는다.
    print("\n=== 재고비용 (사전지정 분위수: b->q =", PRESPECIFIED_Q, ") ===")
    for b, q in PRESPECIFIED_Q.items():
        s = df[np.isclose(df.b, b)]
        cd = s[np.isclose(s.q, q)]
        ab = s[np.isclose(s.q, 0.5) & s.variant.isin(["A_point", "B_recenter"])]
        tab = (pd.concat([ab, cd[cd.variant.isin(["C_spread", "D_fullEQ"])]])
               .pivot(index="model", columns="variant",
                      values="cost_per_unit_demand"))
        print(f"\nb={b} (A/B at q=0.5; C/D at q={q})")
        print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
