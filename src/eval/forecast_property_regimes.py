"""핵심 기여 (RQ1): 어떤 예측 속성이 재고비용을 예측하는가 — 서비스 체제별.

기존 분석 대비 세 가지를 고친다.
  1) 비용구조 b와 발주분위수 q를 **분리**한다 (기존 시나리오는 둘을 함께 바꿨다).
  2) "상단 꼬리 캘리브레이션"을 WQL로 대리하지 않고, **명목 q에서의 경험적 커버리지
     오차**라는 직접 측정치를 넣는다.
  3) 순위상관 tau의 불확실성을 **매장 군집 부트스트랩**으로 준다 (시계열 독립 가정 폐기).

산출: results/property_regimes_{dataset}[_{path}].csv

사용: python -m src.eval.forecast_property_regimes --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
PROPS = ["wape", "abs_wpe", "wql", "abs_cov_err"]
HOLD = 0.02


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--demand-path", default="observed",
                    choices=["observed", "uplift", "clean"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--b", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    args = ap.parse_args()

    suffix = "" if args.demand_path == "observed" else f"_{args.demand_path}"
    diag = pd.read_parquet(RESULTS / f"per_series_diag_{args.dataset}.parquet")
    out = pd.read_parquet(
        RESULTS / f"cost_q_outcomes_{args.dataset}{suffix}_sl2_L0.parquet")

    runs = sorted(set(diag.run) & set(out.run))
    qs = sorted(out.q.unique())
    diag = diag[diag.run.isin(runs)]
    out = out[out.run.isin(runs)]

    # ---------- 매장 단위 집계 (부트스트랩 재표집 단위) ----------
    acc_cols = ["abs_err_sum", "err_sum", "y_sum", "pinball_sum", "n_days"]
    hit_cols = [c for c in diag.columns if c.startswith("hits_")]
    A = (diag.groupby(["store_id", "run"])[acc_cols + hit_cols].sum()
         .unstack("run").reindex(columns=runs, level=1).fillna(0.0))
    stores = A.index.to_numpy()
    ns = len(stores)

    print(f"stores: {ns} | runs: {len(runs)} | q: {qs} | boot: {args.n_boot}",
          flush=True)

    def mat(df, field, run_order):
        return df.loc[:, [(field, r) for r in run_order]].to_numpy()

    M = {f: mat(A, f, runs) for f in acc_cols}
    H = {c: mat(A, c, runs) for c in hit_cols}

    # 점예측 모델은 중앙값 위로 발주할 수 없어 q!=0.5 격자에 값이 없다.
    # q마다 해당 격자에서 실제로 발주 가능한 기법만 사용한다.
    W, L, HD, D, RUNS_Q = {}, {}, {}, {}, {}
    for q in qs:
        sub = out[np.isclose(out.q, q)]
        rq = [r for r in runs if r in set(sub.run)]
        piv = (sub.groupby(["store_id", "run"])[["waste", "lost", "hold", "demand"]]
               .sum().unstack("run").reindex(index=stores).fillna(0.0))
        RUNS_Q[q] = rq
        W[q], L[q] = mat(piv, "waste", rq), mat(piv, "lost", rq)
        HD[q], D[q] = mat(piv, "hold", rq), mat(piv, "demand", rq)

    rng = np.random.default_rng(0)
    boot_idx = [np.arange(ns)] + [rng.integers(0, ns, ns)
                                  for _ in range(args.n_boot)]

    rows = []
    for bi, idx in enumerate(boot_idx):
        y = M["y_sum"][idx].sum(axis=0)
        wape = M["abs_err_sum"][idx].sum(axis=0) / y
        wpe = M["err_sum"][idx].sum(axis=0) / y
        wql = M["pinball_sum"][idx].sum(axis=0) / y
        nd = M["n_days"][idx].sum(axis=0)
        cov = {float(c.split("_")[1]): H[c][idx].sum(axis=0) / nd for c in hit_cols}
        for q in qs:
            dem = D[q][idx].sum(axis=0)
            sel = [runs.index(r) for r in RUNS_Q[q]]
            base = dict(wape=wape[sel], abs_wpe=np.abs(wpe[sel]), wql=wql[sel])
            if q in cov:
                base["abs_cov_err"] = np.abs(cov[q][sel] - q)
            else:
                base["abs_cov_err"] = np.full(len(sel), np.nan)
            for b in args.b:
                cost = (W[q][idx].sum(axis=0) + b * L[q][idx].sum(axis=0)
                        + HOLD * HD[q][idx].sum(axis=0)) / dem
                rec = {"boot": bi, "q": q, "b": b}
                for p in PROPS:
                    v = base[p]
                    ok = np.isfinite(v) & np.isfinite(cost)
                    if ok.sum() >= 8 and np.ptp(v[ok]) > 0:
                        t, pv = kendalltau(v[ok], cost[ok])
                        rec[f"tau_{p}"] = t
                        if bi == 0:
                            rec[f"p_{p}"] = pv
                            rec[f"n_{p}"] = int(ok.sum())
                    else:
                        rec[f"tau_{p}"] = np.nan
                rows.append(rec)
        if bi % 200 == 0:
            print(f"  boot {bi}/{args.n_boot}", flush=True)

    df = pd.DataFrame(rows)
    point = df[df.boot == 0].drop(columns="boot")
    bs = df[df.boot > 0]
    ci = (bs.groupby(["q", "b"])[[f"tau_{p}" for p in PROPS]]
          .quantile([0.025, 0.975]).unstack())
    ci.columns = [f"{a}_{'lo' if b == 0.025 else 'hi'}" for a, b in ci.columns]
    res = point.merge(ci.reset_index(), on=["q", "b"])
    res.insert(0, "demand_path", args.demand_path)
    dst = RESULTS / f"property_regimes_{args.dataset}{suffix}.csv"
    res.to_csv(dst, index=False)
    print("\nsaved:", dst.name)

    for b in args.b:
        s = res[np.isclose(res.b, b)].sort_values("q")
        print(f"\n=== b={b}: 발주분위수별 tau(속성, 비용) [95% 매장 군집 부트스트랩] ===")
        for r in s.itertuples():
            def f(p):
                t = getattr(r, f"tau_{p}")
                lo = getattr(r, f"tau_{p}_lo")
                hi = getattr(r, f"tau_{p}_hi")
                return f"{t:+.2f} [{lo:+.2f},{hi:+.2f}]" if np.isfinite(t) else "  --  "
            print(f"  q={r.q:<5} |WPE| {f('abs_wpe')}   WQL {f('wql')}   "
                  f"|cov err| {f('abs_cov_err')}   WAPE {f('wape')}")


if __name__ == "__main__":
    main()
