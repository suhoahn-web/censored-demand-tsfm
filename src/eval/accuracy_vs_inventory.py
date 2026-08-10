"""실험 4-C: 정확도 순위 vs 재고성과 순위의 괴리 분석 (RQ4 핵심).

설계 근거: Theodorou et al. (2025, EJOR) — 정확도-재고성과 관계는 약하고 조건부.
        Kourentzes et al. (2020, IJPE) — 재고성과의 동인은 정확도가 아니라 편향.
분석:
  1) 지표(WAPE/WQL/|WPE|) 순위 vs 시나리오별 재고비용 순위의 Kendall τ
  2) 순위 역전 사례 추출
  3) 편향(|WPE|)이 정확도(WAPE)보다 비용을 잘 설명하는지 비교

사용: python -m src.eval.accuracy_vs_inventory --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"


def load_accuracy(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[(df.eval_filter == "nostock") & (df.dataset == dataset)]
    zs = df[df.adapt == "zero-shot"][["run", "wape", "wpe", "wql", "crps"]]
    ft = df[(df.adapt != "zero-shot") & (df.get("seed").eq(0))].copy()
    if len(ft):
        ft["run"] = "ft_" + ft["run"].str.split("/").str[-1]
        ft = ft[["run", "wape", "wpe", "wql", "crps"]]
        zs = pd.concat([zs, ft], ignore_index=True)
    zs["abs_wpe"] = zs["wpe"].abs()
    return zs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--inventory-file", default=None)
    args = ap.parse_args()

    inv_file = args.inventory_file or f"inventory_{args.dataset}_sl2_L0.csv"
    inv = pd.read_csv(RESULTS / inv_file)
    best = inv.loc[inv.groupby(["run", "scenario"])["total_cost"].idxmin()]
    acc = load_accuracy(args.dataset)

    merged = best.merge(acc, on="run", how="inner")
    print(f"matched models: {merged['run'].nunique()}")

    print("\n=== Kendall τ: 예측 지표 순위 vs 재고비용 순위 (양수=일치) ===")
    out_rows = []
    for sc, g in merged.groupby("scenario"):
        row = {"scenario": sc, "n": len(g)}
        for metric in ["wape", "wql", "crps", "abs_wpe"]:
            tau, p = kendalltau(g[metric], g["cost_per_unit_demand"])
            row[f"tau_{metric}"] = round(tau, 3)
            row[f"p_{metric}"] = round(p, 4)
        out_rows.append(row)
    tau_df = pd.DataFrame(out_rows)
    print(tau_df.to_string(index=False))

    print("\n=== 시나리오 간 비용 순위 역전 (상위 모델) ===")
    piv = best.pivot_table(index="run", columns="scenario", values="cost_per_unit_demand")
    ranks = piv.rank()
    ranks["rank_swing"] = ranks.max(axis=1) - ranks.min(axis=1)
    print(ranks.sort_values("rank_swing", ascending=False).head(8).round(1).to_string())

    print("\n=== 분포예측 보유 여부별 비용 (분위수 개수) ===")
    qcount = {}
    import json
    for meta in (RESULTS / "zeroshot").glob("*.meta.json"):
        m = json.loads(meta.read_text())
        qcount[meta.name.replace(".parquet.meta.json", "")] = len(m.get("quantiles", []))
    merged["n_quantiles"] = merged["run"].map(qcount).fillna(11).astype(int)
    merged["prob_capable"] = merged["n_quantiles"] > 1
    summ = (merged.groupby(["scenario", "prob_capable"])["cost_per_unit_demand"]
            .agg(["mean", "min", "count"]).round(4))
    print(summ.to_string())

    out = RESULTS / f"accuracy_vs_inventory_{args.dataset}.csv"
    merged.to_csv(out, index=False)
    tau_df.to_csv(RESULTS / f"kendall_tau_{args.dataset}.csv", index=False)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
