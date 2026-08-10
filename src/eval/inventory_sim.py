"""실험 4-B: 신선식품 다기간 재고 시뮬레이션 (7일 평가 구간).

프로토콜 근거 (설계서 v2.0 6절, notes/theodorou2025-ejor.md, notes/syntetos-boylan2006-ijpe.md):
  - 주기적 리뷰(R=1일) order-up-to, 리드타임 L, **lost sales**(소매 방어적 가정)
  - 발주점 S_t = 분위수 예측 F̂⁻¹(CR) (보호구간 R+L 합), 발주량 = max(0, S_t − 재고포지션)
  - 신선식품 확장(우리 기여): 유통기한 m일 + FIFO 소진, 만료분은 폐기비용
  - 비용: 폐기(구매원가 1) + 품절(기회손실 b) + 보유(h/단위·일) + 발주 고정비(k)
  - 시나리오 3종은 마진 b로 구분: 저마진(폐기 우세) / 중간 / 고마진(품절 우세)
  - 트레이드오프 곡선: CR 스윕 → (달성 서비스수준, 평균재고) (Syntetos & Boylan 2006 방식)

주의(논문 한계로 명시): 평가 구간 수요는 관측 판매량이라 품절일에는 검열되어 있다.
따라서 (a) 전체 시계열 (b) 평가 주간에 품절이 전혀 없는 시계열만 — 두 조건으로 보고한다.

사용: python -m src.eval.inventory_sim --dataset lt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
HORIZON = 7
TRUTH_OF = {"frn50krec": "frn50k"}

# 비용 시나리오: 구매원가 1 정규화. b=단위 품절 기회손실(마진), h=보유, k=발주 고정비
# cr_star = 뉴스벤더 임계비 b/(1+b) — 이론적 발주 분위수.
# 다기간·유통기한 이월이 있으면 잔여재고가 내일 팔릴 수 있어 유효 과잉비용이
# 1보다 작고, 따라서 실제 최적 분위수는 cr_star보다 높다. 주 결과는 모든 모델을
# **동일한 고정 CR**에서 비교한다 (Huber et al. 2019의 목표 서비스수준 고정 방식).
SCENARIOS = {
    "low_margin_waste_dominant": dict(b=0.3, h=0.02, k=0.0, cr_star=0.231, cr_fixed=0.5),
    "medium_margin": dict(b=1.0, h=0.02, k=0.0, cr_star=0.500, cr_fixed=0.7),
    "high_margin_stockout_dominant": dict(b=3.0, h=0.02, k=0.0, cr_star=0.750, cr_fixed=0.9),
}
CRS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SHELF_LIFE = 2  # 신선식품 기본 유통기한(일). 민감도는 --shelf-life로
BURN_IN = 2     # 초기 재고 0에서 출발하는 과도구간 — 비용 집계에서 제외


def order_at_cr(pred: pd.DataFrame, cr: float) -> np.ndarray | None:
    """CR 분위수 (보유 분위수에서 필요시 선형보간). 없으면 None."""
    levels = sorted(float(c[1:]) for c in pred.columns if c.startswith("q"))
    if not levels:
        return None
    if cr in levels:
        return np.clip(pred[f"q{cr}"].to_numpy(), 0, None)
    lo = [l for l in levels if l < cr]
    hi = [l for l in levels if l > cr]
    if not lo or not hi:
        return None
    l0, l1 = max(lo), min(hi)
    w = (cr - l0) / (l1 - l0)
    return np.clip((1 - w) * pred[f"q{l0}"].to_numpy() + w * pred[f"q{l1}"].to_numpy(), 0, None)


def simulate(demand: np.ndarray, target: np.ndarray, shelf_life: int,
             lead_time: int, burn_in: int = BURN_IN) -> dict:
    """(n_series, 7) 수요·발주점 → 다기간 lost-sales + 유통기한 시뮬레이션.

    재고는 (잔여수명별) 벡터로 관리하고 FIFO로 소진, 수명 만료분은 폐기.
    """
    n, t_max = demand.shape
    # inv[:, a] = 잔여수명 a+1일인 재고량
    inv = np.zeros((n, shelf_life))
    pipeline = np.zeros((n, max(lead_time, 1)))  # 미도착 발주
    tot_sales = np.zeros(n)
    tot_lost = np.zeros(n)
    tot_waste = np.zeros(n)
    tot_hold = np.zeros(n)
    tot_orders = np.zeros(n)
    n_order_events = np.zeros(n)

    for t in range(t_max):
        # 1) 입고 (리드타임 경과분) — 신규 입고는 최대 수명
        if lead_time == 0:
            arriving = np.zeros(n)
        else:
            arriving = pipeline[:, 0].copy()
            pipeline[:, :-1] = pipeline[:, 1:]
            pipeline[:, -1] = 0.0
        inv[:, -1] += arriving

        # 2) 발주 결정 (리뷰 시점) — 재고포지션 = 보유 + 미도착
        pos = inv.sum(axis=1) + pipeline.sum(axis=1)
        order = np.maximum(target[:, t] - pos, 0.0)
        if t >= burn_in:
            tot_orders += order
            n_order_events += (order > 1e-9)
        if lead_time == 0:
            inv[:, -1] += order  # 즉시 입고
        else:
            pipeline[:, lead_time - 1] += order

        # 3) 수요 충족 (FIFO: 잔여수명 짧은 것부터)
        remaining = demand[:, t].copy()
        for a in range(shelf_life):
            take = np.minimum(inv[:, a], remaining)
            inv[:, a] -= take
            remaining -= take
        count = t >= burn_in   # 초기 재고 0 과도구간은 비용 집계 제외
        if count:
            tot_sales += demand[:, t] - remaining
            tot_lost += remaining

        # 4) 마감: 보유비용 + 노화(수명 만료분 폐기)
        if count:
            tot_hold += inv.sum(axis=1)
            tot_waste += inv[:, 0]
        inv[:, :-1] = inv[:, 1:]
        inv[:, -1] = 0.0

    return {"sales": tot_sales, "lost": tot_lost, "waste": tot_waste,
            "hold": tot_hold, "ordered": tot_orders, "n_orders": n_order_events}


def collect_predictions(dataset: str):
    items = []
    zs = RESULTS / "zeroshot"
    for meta_path in sorted(zs.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("cutoff_date") or meta["dataset"] != dataset:
            continue
        name = meta_path.name.replace(".parquet.meta.json", "")
        if name.startswith("ctxabl_"):
            continue
        if re.search(r"_s\d+_raw$", name):
            continue  # 시드 복제본은 Table 7 집계 전용  # 맥락길이 ablation은 본 분석 제외
        items.append((name, pd.read_parquet(str(meta_path).replace(".meta.json", ""))))
    ft = RESULTS / "finetune_v2"
    if ft.exists():
        for run_dir in sorted(ft.iterdir()):
            mf, pf = run_dir / "meta.json", run_dir / "predictions.parquet"
            if not (mf.exists() and pf.exists()):
                continue
            meta = json.loads(mf.read_text())
            if meta["dataset"] != dataset or meta.get("seed") != 0:
                continue
            items.append((f"ft_{run_dir.name}", pd.read_parquet(pf)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--shelf-life", type=int, default=SHELF_LIFE)
    ap.add_argument("--lead-time", type=int, default=0)
    ap.add_argument("--clean-only", action="store_true",
                    help="평가 주간에 품절이 전혀 없는 시계열만 (검열 없는 부분표본)")
    args = ap.parse_args()

    truth = pd.read_parquet(PROCESSED / f"{TRUTH_OF.get(args.dataset, args.dataset)}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    if args.clean_only:
        clean = (truth.groupby(KEY)["stock_hour6_22_cnt"].max() == 0)
        clean = clean[clean].index
        truth = truth.set_index(KEY).loc[clean].reset_index()
        print(f"clean subsample: {len(clean)} series", flush=True)

    dwide = truth.pivot_table(index=KEY, columns="h", values="sale_amount").sort_index()
    demand = dwide.to_numpy()

    rows = []
    for name, pred in collect_predictions(args.dataset):
        p = pred.set_index(KEY + ["h"]).sort_index()
        for cr in CRS:
            flat = order_at_cr(pred.sort_values(KEY + ["h"]), cr)
            if flat is None:
                continue
            keys_sorted = pred.sort_values(KEY + ["h"])[KEY + ["h"]]
            tgt = (keys_sorted.assign(v=flat)
                   .pivot_table(index=KEY, columns="h", values="v")
                   .reindex(dwide.index))
            if tgt.isna().any().any():
                continue
            r = simulate(demand, tgt.to_numpy(), args.shelf_life, args.lead_time)
            tot_d = demand[:, BURN_IN:].sum()
            for sc, c in SCENARIOS.items():
                cost = (r["waste"].sum() * 1.0 + r["lost"].sum() * c["b"]
                        + r["hold"].sum() * c["h"] + r["n_orders"].sum() * c["k"])
                rows.append({
                    "run": name, "cr": cr, "scenario": sc,
                    "is_fixed_cr": bool(abs(cr - c["cr_fixed"]) < 1e-9),
                    "cr_star": c["cr_star"],
                    "total_cost": float(cost),
                    "cost_per_unit_demand": float(cost / tot_d),
                    "fill_rate": float(r["sales"].sum() / tot_d),
                    "waste_rate": float(r["waste"].sum() / max(r["ordered"].sum(), 1e-9)),
                    "avg_inventory": float(r["hold"].sum() / (len(demand) * (HORIZON - BURN_IN))),
                    "lost_rate": float(r["lost"].sum() / tot_d),
                })
        print(f"  simulated: {name}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.dataset}{'_clean' if args.clean_only else ''}_sl{args.shelf_life}_L{args.lead_time}"
    out = RESULTS / f"inventory_{tag}.csv"
    df.to_csv(out, index=False)

    # 시나리오별 최적 CR에서의 모델 순위
    print("\n=== 시나리오별 최소 비용 (모델이 CR을 최적 선택했을 때) ===")
    best = (df.loc[df.groupby(["run", "scenario"])["total_cost"].idxmin()]
              [["run", "scenario", "cr", "cost_per_unit_demand", "fill_rate", "waste_rate"]])
    for sc in SCENARIOS:
        s = best[best.scenario == sc].sort_values("cost_per_unit_demand")
        print(f"\n[{sc}]")
        print(s[["run", "cr", "cost_per_unit_demand", "fill_rate", "waste_rate"]]
              .round(4).to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
