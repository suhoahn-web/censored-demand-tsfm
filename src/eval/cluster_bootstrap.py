"""P0-7: 매장(및 상품) 군집 부트스트랩 신뢰구간.

문제: 매장-상품 시계열 1만 개는 독립 반복이 아니다. 같은 매장의 상품들은 같은 수요
충격·품절 이벤트를 공유하고, 같은 상품은 여러 매장에 반복된다. Friedman-Nemenyi와
MCB는 시계열 독립을 가정하므로 주 추론 근거로 쓸 수 없다.

여기서는 **매장을 군집으로 재표집**해 지표 차이의 신뢰구간을 만든다. 지표가 모두
합의 비(比)이므로 매장 단위 합을 미리 만들어 두면 재표집이 행렬 합산 한 번으로 끝난다.

산출: results/cluster_bootstrap_{dataset}.csv

사용: python -m src.eval.cluster_bootstrap --dataset lt --reference etseq_lt_raw
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
B_DEFAULT = 2000


def _ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, n_runs) 부트스트랩 비율. num/den은 (n_clusters, n_runs)/(n_clusters,)."""
    return num[idx].sum(axis=1) / den[idx].sum(axis=1)[:, None]


def bootstrap_metric(pivot_num: pd.DataFrame, den: pd.Series, idx: np.ndarray,
                     runs: list[str]) -> pd.DataFrame:
    """군집 합 행렬에서 지표의 부트스트랩 분포를 만든다."""
    num = pivot_num[runs].to_numpy()
    boot = _ratio_ci(num, den.to_numpy(), idx)
    point = num.sum(axis=0) / den.sum()
    return pd.DataFrame({"run": runs, "point": point,
                         "lo": np.percentile(boot, 2.5, axis=0),
                         "hi": np.percentile(boot, 97.5, axis=0)}), boot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--cluster", default="store_id",
                    choices=["store_id", "product_id"])
    ap.add_argument("--reference", default="etseq_lt_raw",
                    help="차이를 계산할 기준 기법")
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    ap.add_argument("--b", type=float, default=1.0, help="재고비용 시나리오 b")
    ap.add_argument("--q", type=float, default=0.7, help="발주 분위수")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cl = args.cluster

    # ---------- 예측 정확도 ----------
    diag = pd.read_parquet(RESULTS / f"per_series_diag_{args.dataset}.parquet")
    runs = sorted(diag.run.unique())
    g = diag.groupby([cl, "run"]).agg(
        abs_err=("abs_err_sum", "sum"), err=("err_sum", "sum"),
        y=("y_sum", "sum"), pin=("pinball_sum", "sum")).reset_index()
    den = g[g.run == runs[0]].set_index(cl)["y"].sort_index()
    clusters = den.index.to_numpy()
    idx = rng.integers(0, len(clusters), size=(args.n_boot, len(clusters)))
    print(f"clusters ({cl}): {len(clusters)} | runs: {len(runs)} | "
          f"boot: {args.n_boot}", flush=True)

    def piv(col):
        return (g.pivot(index=cl, columns="run", values=col)
                .sort_index().reindex(columns=runs))

    rows = []
    for metric, col in [("WAPE", "abs_err"), ("WPE", "err"), ("WQL", "pin")]:
        tab, boot = bootstrap_metric(piv(col), den, idx, runs)
        ref_j = runs.index(args.reference)
        dif = boot - boot[:, [ref_j]]
        tab["diff_vs_ref"] = tab.point - tab.point[ref_j]
        tab["diff_lo"] = np.percentile(dif, 2.5, axis=0)
        tab["diff_hi"] = np.percentile(dif, 97.5, axis=0)
        tab["metric"] = metric
        rows.append(tab)

    # ---------- 재고비용 ----------
    out = pd.read_parquet(RESULTS / f"cost_q_outcomes_{args.dataset}_sl2_L0.parquet")
    sel = out[np.isclose(out.q, args.q)]
    sel = sel.assign(cost=sel.waste + args.b * sel.lost + 0.02 * sel.hold)
    gc = sel.groupby([cl, "run"]).agg(cost=("cost", "sum"), dem=("demand", "sum"),
                                      lost=("lost", "sum"), sales=("sales", "sum")
                                      ).reset_index()
    cruns = sorted(gc.run.unique())
    cden = gc[gc.run == cruns[0]].set_index(cl)["dem"].sort_index()
    cidx = rng.integers(0, len(cden), size=(args.n_boot, len(cden)))
    for metric, col in [("cost_per_unit_demand", "cost"), ("fill_rate", "sales")]:
        pv = gc.pivot(index=cl, columns="run", values=col).sort_index().reindex(columns=cruns)
        tab, boot = bootstrap_metric(pv, cden, cidx, cruns)
        ref = args.reference if args.reference in cruns else cruns[0]
        rj = cruns.index(ref)
        dif = boot - boot[:, [rj]]
        tab["diff_vs_ref"] = tab.point - tab.point[rj]
        tab["diff_lo"] = np.percentile(dif, 2.5, axis=0)
        tab["diff_hi"] = np.percentile(dif, 97.5, axis=0)
        tab["metric"] = f"{metric} (b={args.b}, q={args.q})"
        rows.append(tab)

    res = pd.concat(rows, ignore_index=True)
    res["cluster"] = cl
    res["reference"] = args.reference
    out_path = RESULTS / f"cluster_bootstrap_{args.dataset}_{cl}.csv"
    res.to_csv(out_path, index=False)
    print("saved:", out_path.name)

    for m in res.metric.unique():
        s = res[res.metric == m].sort_values("point")
        print(f"\n=== {m} (군집={cl}, 기준={args.reference}) ===")
        print(s[["run", "point", "lo", "hi", "diff_vs_ref", "diff_lo", "diff_hi"]]
              .head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
