"""OR Spectrum 개정판 신규 표 (markdown).

  table_regimes       발주분위수별 tau(예측속성, 비용) + 부트스트랩 구간
  table_coverage      명목 분위수별 경험적 커버리지
  table_eqabl         EQ 분해 (A/B/C/D)
  table_bootstrap     군집 부트스트랩 재고비용 구간
  table_demandpath    평가 수요 경로 민감도

사용: python -m src.viz.make_tables_orsp
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.style import short_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"


def _write(name: str, head: str, body: str, foot: str = "") -> None:
    txt = f"**{head}**\n\n{body}\n"
    if foot:
        txt += f"\n*{foot}*\n"
    (TABLES / name).write_text(txt, encoding="utf-8")
    print("  ", name)


def _md(df: pd.DataFrame, align=None) -> str:
    cols = list(df.columns)
    a = align or (["---"] * len(cols))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(a) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def table_regimes() -> None:
    d = pd.read_csv(RESULTS / "property_regimes_lt.csv")
    rows = []
    for b in [0.3, 1.0, 3.0]:
        s = d[np.isclose(d.b, b)].sort_values("q")
        for r in s.itertuples():
            def cell(p):
                t = getattr(r, f"tau_{p}")
                if not np.isfinite(t):
                    return "–"
                lo, hi = getattr(r, f"tau_{p}_lo"), getattr(r, f"tau_{p}_hi")
                return f"{t:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            rows.append({"b": f"{b:g}", "q": f"{r.q:g}",
                         "abs. bias \\|WPE\\|": cell("abs_wpe"),
                         "WQL": cell("wql"),
                         "\\|coverage error\\| at q": cell("abs_cov_err"),
                         "WAPE": cell("wape")})
    _write("orsp_regimes.md",
           "Kendall rank correlation across methods between each forecast "
           "property and inventory cost per unit of demand, by lost margin $b$ "
           "and ordering quantile $q$.",
           _md(pd.DataFrame(rows)),
           "Brackets give 95% percentile intervals from 1,000 store-level "
           "cluster bootstrap replicates. Coverage error is only defined at "
           "quantiles where the predictive distribution is available for enough "
           "methods. A positive value means that a larger value of the property "
           "goes with higher cost.")


def table_coverage() -> None:
    c = pd.read_csv(RESULTS / "quantile_calibration_lt.csv")
    sel = ["ft_lora_r16_lr1e-3_s0", "ft_lora_r16_mask_s0", "chronos2_lt_raw",
           "chronos2_lt_mask", "tirex_lt_raw", "moirai2_lt_raw",
           "timesfm_lt_raw", "lightgbm_lt_raw", "etseq_lt_raw",
           "crostoneq_lt_raw", "sbaeq_lt_raw", "tsbeq_lt_raw"]
    lv = [0.5, 0.6, 0.7, 0.8, 0.9]
    p = c[c.run.isin(sel)].pivot(index="run", columns="nominal",
                                 values="coverage").reindex(sel)[lv]
    out = pd.DataFrame({"Method": [short_label(m) for m in p.index]})
    for q in lv:
        out[f"{q:g}"] = p[q].round(3).values
    out["mean \\|error\\|"] = (p[lv].sub(lv, axis=1).abs().mean(axis=1)
                               .round(4).values)
    _write("orsp_coverage.md",
           "Empirical coverage at the nominal quantiles the replenishment "
           "policy orders from, on uncensored evaluation days.",
           _md(out),
           "Each cell is the share of scored observations at or below the "
           "forecast quantile; perfect calibration equals the nominal level. "
           "The last column is the mean absolute deviation from nominal over "
           "the five levels.")


def table_eqabl() -> None:
    d = pd.read_csv(RESULTS / "eq_ablation_lt.csv")
    # 사전지정 분위수 (§4.5 고정 분위수 설계): A/B는 유일한 운영점(중앙값),
    # C/D는 해당 시나리오의 사전지정 q. 테스트셋 q 선택 금지
    # (results/eq_ablation_q_selection_audit.md).
    PRE_Q = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
    order = ["A_point", "B_recenter", "C_spread", "D_fullEQ"]
    nice = {"A_point": "A. point forecast", "B_recenter": "B. recentre only",
            "C_spread": "C. spread only (median preserved)",
            "D_fullEQ": "D. full empirical quantiles"}
    acc = (d.drop_duplicates(["model", "variant"])
             .groupby("variant")[["wape", "wpe", "wql", "cov0.9"]].mean()
             .reindex(order))
    rows = []
    for v in order:
        rec = {"Variant": nice[v],
               "WAPE": f"{acc.loc[v, 'wape'] * 100:.2f}",
               "WPE": f"{acc.loc[v, 'wpe'] * 100:+.2f}",
               "WQL": f"{acc.loc[v, 'wql']:.4f}",
               "cov. at 0.9": ("–" if not np.isfinite(acc.loc[v, "cov0.9"])
                               else f"{acc.loc[v, 'cov0.9']:.3f}")}
        for b, q in PRE_Q.items():
            vq = 0.5 if v in ("A_point", "B_recenter") else q
            s = d[np.isclose(d.b, b) & (d.variant == v) & np.isclose(d.q, vq)]
            rec[f"cost, b={b:g} (q={q:g})"] = f"{s.cost_per_unit_demand.mean():.4f}"
        rows.append(rec)
    tab = pd.DataFrame(rows)
    _write("orsp_eqabl.md",
           "Decomposing the empirical-quantile procedure: forecast metrics and "
           "inventory cost for the four variants of Section 4.3, averaged over "
           "the five point forecasters, at the prespecified ordering quantile "
           "of each cost regime.",
           _md(tab),
           "WAPE and WPE in per cent. Variants A and B carry no quantiles above "
           "the median and are therefore evaluated at their only operating "
           "point, q = 0.5; variants C and D are evaluated at the prespecified "
           "quantile of each regime. At q = 0.5 the orders of C equal those of "
           "A and the orders of D equal those of B exactly, so the b = 0.3 "
           "column shows the structural identity. Variant C leaves the point "
           "forecast unchanged, so its WAPE and WPE equal those of variant A "
           "by construction. No quantile is selected on the test data.")


def table_bootstrap() -> None:
    d = pd.read_csv(RESULTS / "cluster_bootstrap_lt_store_id.csv")
    m = [x for x in d.metric.unique() if x.startswith("cost_per_unit")][0]
    s = d[d.metric == m].nsmallest(10, "point")
    fr = d[d.metric.str.startswith("fill_rate")].set_index("run")["point"]
    out = pd.DataFrame({
        "Method": [short_label(r) for r in s.run],
        "cost / unit demand": [f"{v:.4f} [{lo:.4f}, {hi:.4f}]"
                               for v, lo, hi in zip(s.point, s.lo, s.hi)],
        "difference vs ETS+EQ": [f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                                 for v, lo, hi in zip(s.diff_vs_ref, s.diff_lo,
                                                      s.diff_hi)],
        "achieved fill rate": [f"{fr.get(r, float('nan')):.3f}" for r in s.run],
    })
    _write("orsp_bootstrap.md",
           f"Inventory cost with store-level cluster bootstrap intervals in the "
           f"shortage-dominated cell ($b = 3.0$, $q = 0.9$); ten cheapest "
           f"methods.",
           _md(out),
           "Brackets give 95% percentile intervals from 2,000 store-level "
           "cluster bootstrap replicates. A difference whose interval excludes "
           "zero is distinguishable from the reference; the achieved fill rate "
           "differs across methods even though all order at the same quantile.")


def table_demandpath() -> None:
    d = pd.read_csv(RESULTS / "demand_target_sensitivity_lt.csv")
    rows = []
    for q in [0.5, 0.7, 0.9]:
        for b in [0.3, 1.0, 3.0]:
            s = d[np.isclose(d.q, q) & np.isclose(d.b, b)]
            pv = s.pivot(index="run", columns="demand_path",
                         values="cost_per_unit_demand").dropna()
            c = pv.rank().corr(method="spearman")
            best = {p: short_label(pv[p].idxmin()) for p in pv.columns}
            rows.append({"q": f"{q:g}", "b": f"{b:g}",
                         "obs. vs uplift": f"{c.loc['observed', 'uplift']:.2f}",
                         "obs. vs clean": f"{c.loc['observed', 'clean']:.2f}",
                         "cheapest (observed)": best["observed"],
                         "cheapest (uplift)": best["uplift"]})
    _write("orsp_demandpath.md",
           "Sensitivity of the inventory comparison to the assumed evaluation "
           "demand path.",
           _md(pd.DataFrame(rows)),
           "Entries are Spearman rank correlations of cost across the 28 "
           "methods between demand paths, and the cheapest method under each "
           "path. The observed path uses recorded sales, the uplift path "
           "applies the hour-proportional correction of Section 3.3, and the "
           "clean path restricts to the 340 series with no stockout in the "
           "evaluation week.")


if __name__ == "__main__":
    TABLES.mkdir(parents=True, exist_ok=True)
    table_regimes()
    table_coverage()
    table_eqabl()
    table_bootstrap()
    table_demandpath()
    print("done")
