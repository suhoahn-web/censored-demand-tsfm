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


def quantile_matrix(pred_sorted: pd.DataFrame, index: pd.MultiIndex
                    ) -> tuple[np.ndarray, np.ndarray]:
    """예측 테이블 → (levels, Q[n, T, K]) 분위수 격자 (series × horizon × level)."""
    levels = np.array(sorted(float(c[1:]) for c in pred_sorted.columns
                             if c.startswith("q")))
    mats = []
    for q in levels:
        w = (pred_sorted[KEY + ["h", f"q{q}"]]
             .pivot_table(index=KEY, columns="h", values=f"q{q}")
             .reindex(index))
        mats.append(np.clip(w.to_numpy(), 0, None))
    Q = np.stack(mats, axis=-1)
    Q = np.sort(Q, axis=-1)  # 분위수 교차 방지
    return levels, Q


def protection_levels(levels: np.ndarray, Q: np.ndarray, q: float, lead_time: int,
                      method: str = "independent", n_draws: int = 400,
                      seed: int = 0) -> np.ndarray:
    """L>0에서의 order-up-to 수준: 보호구간 t..t+L 누적수요의 q-분위수 근사 (n, T).

    일별 한계분위수만 있으므로 합의 분위수는 종속구조 가정이 필요하다.
      "comonotone"  — 일별 q-분위수의 합 (완전 양의 종속; 합 분위수의 상한, 버퍼 과다)
      "independent" — 일별 한계분포(분위수 격자의 선형보간 역CDF)에서 독립 추출한
                      합의 경험적 q-분위수 (Monte Carlo, 격자 밖은 최외곽 분위수로 절단)
    구간이 평가창을 넘으면 평가창 안의 날만 포함한다.
    """
    n, T, K = Q.shape
    out = np.zeros((n, T))
    if method == "comonotone":
        Qq = np.stack([_interp_level(levels, Q[:, t, :], q) for t in range(T)], axis=1)
        for t in range(T):
            out[:, t] = Qq[:, t:min(t + lead_time, T - 1) + 1].sum(axis=1)
        return out
    if method != "independent":
        raise ValueError(method)
    rng = np.random.default_rng(seed)
    rows = np.arange(n)[:, None]
    samples = np.zeros((n, T, n_draws))
    for t in range(T):
        u = rng.random((n, n_draws))
        k = np.clip(np.searchsorted(levels, u) - 1, 0, K - 2)
        w = (u - levels[k]) / (levels[k + 1] - levels[k])
        v = Q[rows, t, k] * (1 - w) + Q[rows, t, k + 1] * w
        v = np.where(u < levels[0], Q[:, t, 0][:, None], v)
        v = np.where(u > levels[-1], Q[:, t, -1][:, None], v)
        samples[:, t, :] = v
    for t in range(T):
        s = samples[:, t:min(t + lead_time, T - 1) + 1, :].sum(axis=1)
        out[:, t] = np.quantile(s, q, axis=1)
    return np.clip(out, 0, None)


def _interp_level(levels: np.ndarray, Qt: np.ndarray, q: float) -> np.ndarray:
    """(n, K) 격자에서 수준 q의 분위수를 선형보간 (격자 밖은 최외곽)."""
    if q <= levels[0]:
        return Qt[:, 0]
    if q >= levels[-1]:
        return Qt[:, -1]
    k = np.searchsorted(levels, q) - 1
    w = (q - levels[k]) / (levels[k + 1] - levels[k])
    return Qt[:, k] * (1 - w) + Qt[:, k + 1] * w


def simulate(demand: np.ndarray, target: np.ndarray, shelf_life: int,
             lead_time: int, burn_in: int = BURN_IN,
             lead_target: str = "protection", terminal: str = "ignore") -> dict:
    """(n_series, 7) 수요·발주점 → 다기간 lost-sales + 유통기한 시뮬레이션.

    재고는 (잔여수명별) 벡터로 관리하고 FIFO로 소진, 수명 만료분은 폐기.

    일별 순서: (1) 리드타임 경과 발주분 입고 → (2) 재고포지션 검토·발주 →
    (3) 수요 충족(FIFO) → (4) 보유비용·노화·만료 폐기. 예측은 origin에서 1회
    생성되며(h=1..7) 일별 재예측은 없다.

    lead_target: L>0일 때 발주목표(order-up-to level)의 정의.
      "protection" — target을 이미 보호구간 t..t+L 수준으로 계산해 넘긴 경우
                     (`protection_levels` 참조; 코모노톤 합 또는 독립 합 근사).
                     그대로 target[:, t]를 쓴다.
      "same-day"   — 당일 t의 분위수 하나 (2026-08 민감도 실행의 레거시 설정:
                     보호구간을 L일만큼 과소평가한다; 재현용으로만 유지).
    terminal: 평가창 종료 시 잔여 재고 처리. "ignore"(무정산, 기존) 또는
      "waste"(전량 폐기로 계상).
    """
    n, t_max = demand.shape
    if lead_target not in ("protection", "same-day"):
        raise ValueError(lead_target)
    if terminal not in ("ignore", "waste"):
        raise ValueError(terminal)
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

    terminal_inv = inv.sum(axis=1) + pipeline.sum(axis=1)
    if terminal == "waste":
        tot_waste += terminal_inv
    return {"sales": tot_sales, "lost": tot_lost, "waste": tot_waste,
            "hold": tot_hold, "ordered": tot_orders, "n_orders": n_order_events,
            "terminal_inv": terminal_inv}


def collect_predictions(dataset: str, seeds=(0,)):
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
        if re.search(r"equ_", name):
            continue  # EQ 비검열-잔차 변형(JORS R1 E4)은 별도 비교 스크립트에서만 사용
        items.append((name, pd.read_parquet(str(meta_path).replace(".meta.json", ""))))
    ft = RESULTS / "finetune_v2"
    if ft.exists():
        for run_dir in sorted(ft.iterdir()):
            mf, pf = run_dir / "meta.json", run_dir / "predictions.parquet"
            if not (mf.exists() and pf.exists()):
                continue
            meta = json.loads(mf.read_text())
            if meta["dataset"] != dataset or meta.get("seed") not in seeds:
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
    ap.add_argument("--lead-target", default="independent",
                    choices=["independent", "comonotone", "same-day"],
                    help="L>0 발주목표: 보호구간 합의 분위수(독립/코모노톤 근사) "
                         "또는 당일 분위수(legacy)")
    ap.add_argument("--terminal", default="ignore", choices=["ignore", "waste"],
                    help="평가창 종료 시 잔여 재고: 무정산 / 전량 폐기 계상")
    ap.add_argument("--seeds", default="0",
                    help="(R1) 포함할 파인튜닝 시드, 쉼표 구분 (기본 0; 예: 0,1,2 → 시드별 재고비용)")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))

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
    for name, pred in collect_predictions(args.dataset, seeds):
        ps = pred.sort_values(KEY + ["h"])
        levels = Q = None
        if args.lead_time > 0 and args.lead_target != "same-day":
            levels, Q = quantile_matrix(ps, dwide.index)
            if np.isnan(Q).any():
                print(f"  skip (incomplete): {name}", flush=True)
                continue
        for cr in CRS:
            flat = order_at_cr(ps, cr)
            if flat is None:
                continue
            if Q is not None:
                # 점예측 전용(격자 1개)은 보호구간 분위수를 만들 수 없다 → 중앙값만
                if len(levels) < 3 and abs(cr - 0.5) > 1e-9:
                    continue
                if len(levels) < 3:
                    tgt_arr = np.stack([Q[:, t:min(t + args.lead_time, HORIZON - 1) + 1, 0]
                                        .sum(axis=1) for t in range(HORIZON)], axis=1)
                else:
                    tgt_arr = protection_levels(levels, Q, cr, args.lead_time,
                                                method=args.lead_target)
                sim_target = "protection"
            else:
                keys_sorted = ps[KEY + ["h"]]
                tgt = (keys_sorted.assign(v=flat)
                       .pivot_table(index=KEY, columns="h", values="v")
                       .reindex(dwide.index))
                if tgt.isna().any().any():
                    continue
                tgt_arr = tgt.to_numpy()
                sim_target = "same-day"
            r = simulate(demand, tgt_arr, args.shelf_life, args.lead_time,
                         lead_target=sim_target, terminal=args.terminal)
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
                    "terminal_inv_per_unit_demand": float(r["terminal_inv"].sum() / tot_d),
                })
        print(f"  simulated: {name}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.dataset}{'_clean' if args.clean_only else ''}_sl{args.shelf_life}_L{args.lead_time}"
    if args.lead_time > 0:
        tag += {"same-day": "_sameday", "comonotone": "_comon",
                "independent": "_indep"}[args.lead_target]
    if args.terminal == "waste":
        tag += "_termwaste"
    if seeds != (0,):
        tag += "_seeds" + "".join(str(x) for x in seeds)
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
