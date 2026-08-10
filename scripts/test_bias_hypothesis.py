"""중심 가설 검증: 편향(WPE)이 재고비용을 설명하는가?

앞선 분석은 각 모델이 CR을 자유 선택(스윕 최소값)하도록 허용했다 → 분위수 선택이
수준 편향을 보정해 |WPE| 효과가 상쇄됐을 가능성. 여기서는 **이론적 최적 CR로 고정**해
(모델별 튜닝 불가) 편향 가설(Kourentzes et al. 2020)을 공정하게 재검정한다.

뉴스벤더 CR = cu/(cu+co) = b/(b+1):  medium(b=1)→0.5,  high(b=3)→0.75
"""

from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIXED_CR = {"medium_margin": 0.5, "high_margin_stockout_dominant": 0.8}  # 0.75 미보유 → 0.8

inv = pd.read_csv(RESULTS / "inventory_lt_sl2_L0.csv")
acc = pd.read_csv(RESULTS / "summary_metrics.csv")
acc = acc[(acc.eval_filter == "nostock") & (acc.dataset == "lt")]
zs = acc[acc.adapt == "zero-shot"][["run", "wape", "wpe", "wql"]]
ft = acc[(acc.adapt != "zero-shot") & (acc["seed"] == 0)].copy()
ft["run"] = "ft_" + ft["run"].str.split("/").str[-1]
acc = pd.concat([zs, ft[["run", "wape", "wpe", "wql"]]], ignore_index=True)
acc["abs_wpe"] = acc["wpe"].abs()

print("=== 고정 CR 조건에서의 Kendall τ (모델별 CR 튜닝 불가) ===")
for sc, cr in FIXED_CR.items():
    g = inv[(inv.scenario == sc) & (inv.cr == cr)].merge(acc, on="run")
    print(f"\n[{sc}] CR 고정={cr}, n={len(g)}")
    for metric in ["wape", "wql", "abs_wpe", "wpe"]:
        tau, p = kendalltau(g[metric], g["cost_per_unit_demand"])
        flag = "*" if p < 0.05 else " "
        print(f"  tau({metric:8s}) = {tau:+.3f}  p={p:.4f} {flag}")

print("\n=== 비교: 모델별 CR 자유 선택 시 (앞선 분석) ===")
best = inv.loc[inv.groupby(["run", "scenario"])["total_cost"].idxmin()]
for sc in FIXED_CR:
    g = best[best.scenario == sc].merge(acc, on="run")
    tau_b, p_b = kendalltau(g["abs_wpe"], g["cost_per_unit_demand"])
    tau_q, p_q = kendalltau(g["wql"], g["cost_per_unit_demand"])
    print(f"[{sc}] tau(abs_wpe)={tau_b:+.3f} p={p_b:.4f} | tau(wql)={tau_q:+.3f} p={p_q:.4f}")
