"""편향-서비스수준 교차 분석 (RQ4 핵심 기여).

발주 분위수(CR)를 낮은 값에서 높은 값으로 옮기면, 예측 편향과 재고비용의 순위상관이
양(+)에서 음(−)으로 뒤집힌다. 해석:
  - 낮은 CR: 중앙값 근처에서 발주 → 하향 편향이 곧 과소발주 → 품절 → 비용. 편향이 직접 비용.
  - 높은 CR: 의도적으로 중앙값 위에서 발주 → 버퍼가 하향 편향을 흡수하고, 오히려
    편향이 큰 모델이 과잉발주·폐기를 덜 함 → 상관이 음으로 역전.
  - CR이 오를수록 결정적인 것은 상위 분위수의 보정 여부, 즉 분포 형태 품질(WQL).

이 결과는 Kourentzes et al.(2020)의 "편향이 재고성과의 동인" 주장과 우리의
"편향은 무의미하고 WQL이 지배" 관찰을 하나의 메커니즘으로 통합한다.

사용: python -m src.eval.bias_service_crossover --dataset lt
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
METRICS = ["abs_wpe", "wql", "wape"]
MIN_N = 12  # 분위수 보유 모델이 적으면 상관 추정 불안정


def load_accuracy(dataset: str) -> pd.DataFrame:
    acc = pd.read_csv(RESULTS / "summary_metrics.csv")
    acc = acc[(acc.eval_filter == "nostock") & (acc.dataset == dataset)]
    zs = acc[acc.adapt == "zero-shot"][["run", "wape", "wql", "wpe"]]
    ft = acc[(acc.adapt != "zero-shot") & (acc["seed"] == 0)].copy()
    if len(ft):
        ft["run"] = "ft_" + ft["run"].str.split("/").str[-1]
        zs = pd.concat([zs, ft[["run", "wape", "wql", "wpe"]]], ignore_index=True)
    zs = zs.drop_duplicates("run")
    zs["abs_wpe"] = zs["wpe"].abs()
    return zs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--scenario", default="medium_margin")
    args = ap.parse_args()

    inv = pd.read_csv(RESULTS / f"inventory_{args.dataset}_sl2_L0.csv")
    inv = inv[inv.scenario == args.scenario]
    acc = load_accuracy(args.dataset)
    merged = inv.merge(acc, on="run")

    rows = []
    for cr, g in merged.groupby("cr"):
        if len(g) < MIN_N:
            continue
        row = {"cr": cr, "n_models": len(g)}
        for m in METRICS:
            tau, p = kendalltau(g[m], g["cost_per_unit_demand"])
            row[f"tau_{m}"], row[f"p_{m}"] = round(tau, 3), round(p, 4)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("cr")
    out.to_csv(RESULTS / f"bias_crossover_{args.dataset}.csv", index=False)

    print(f"발주 분위수별 순위상관 ({args.scenario}, {args.dataset})")
    print(out.to_string(index=False))
    sig_pos = out[(out.tau_abs_wpe > 0) & (out.p_abs_wpe < 0.05)]["cr"].tolist()
    sig_neg = out[(out.tau_abs_wpe < 0) & (out.p_abs_wpe < 0.05)]["cr"].tolist()
    print(f"\n편향이 비용을 유의하게 예측: CR {sig_pos}")
    print(f"부호가 역전(유의): CR {sig_neg}")
    print("saved:", RESULTS / f"bias_crossover_{args.dataset}.csv")


if __name__ == "__main__":
    main()
