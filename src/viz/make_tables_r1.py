"""JORS R1: 신규 부록 표를 markdown으로 생성.

  C4  분위수 수준별 pinball loss·커버리지 (quantile_losses_lt_wide.csv)
  C5  중앙값 부호오차 vs 격자평균 편향 (bias_mean_vs_median_lt.csv)
  C6  명목 발주 분위수별 달성 서비스 (achieved_service_lt.csv)
  C7  비검열일수 가중 MASE (mase_weighting_lt.csv)
  B8  EQ 비검열 잔차 변형 비교 (eq_uncensored_compare_lt.csv)
  B7  합성 검열 요약 (synthetic_censoring_{lt,frn50k}_summary.csv)

산출: results/tables/r1_*.md
사용: python -m src.viz.make_tables_r1
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "tables"
OUT.mkdir(exist_ok=True)

LABEL = {
    "ft_lora_r16_lr1e-3_s0": "C2 LoRA r16 (lr 1e-3)", "ft_lora_r16_lr1e-5_s0": "C2 LoRA r16 (lr 1e-5)",
    "ft_full_s0": "C2 full-FT", "ft_lora_r16_mask_s0": "C2 LoRA r16 + mask",
    "ft_lora_r16_s0": "C2 LoRA r16", "ft_lora_r32_s0": "C2 LoRA r32", "ft_lora_r8_s0": "C2 LoRA r8",
    "chronos2_lt_raw": "Chronos-2", "chronos2_lt_mask": "Chronos-2 + mask",
    "timesfm_lt_raw": "TimesFM", "timesfm_lt_mask": "TimesFM + mask",
    "moirai2_lt_raw": "Moirai-2", "moirai2_lt_mask": "Moirai-2 + mask",
    "tirex_lt_raw": "TiRex", "tirex_lt_mask": "TiRex + mask",
    "ttm_lt_raw": "TTM", "ttm_lt_mask": "TTM + mask", "ttmeq_lt_raw": "TTM+EQ",
    "lightgbm_lt_raw": "LightGBM",
    "croston_lt_raw": "Croston", "crostoneq_lt_raw": "Croston+EQ", "sba_lt_raw": "SBA",
    "sbaeq_lt_raw": "SBA+EQ", "tsb_lt_raw": "TSB", "tsbeq_lt_raw": "TSB+EQ",
    "ets_lt_raw": "ETS", "etseq_lt_raw": "ETS+EQ", "snaive_lt_raw": "sNaive",
    "snaiveeq_lt_raw": "sNaive+EQ",
    # 50K release (Table B5 lower panel)
    "lightgbm_frn50k_raw": "LightGBM", "etseq_frn50k_raw": "ETS+EQ", "crostoneq_frn50k_raw": "Croston+EQ",
    "tsbeq_frn50k_raw": "TSB+EQ", "sbaeq_frn50k_raw": "SBA+EQ", "snaiveeq_frn50k_raw": "sNaive+EQ",
    "ttmeq_frn50k_raw": "TTM+EQ", "ft_c2_50k_lora_r16_s0": "C2 LoRA r16",
    "tirex_50k_raw": "TiRex", "tirex_frn50k_mask": "TiRex + mask", "timesfm_50k_raw": "TimesFM",
    "timesfm_50k_mask": "TimesFM + mask", "moirai2_50k_raw": "Moirai-2", "moirai2_frn50k_mask": "Moirai-2 + mask",
    "chronos2_50k_raw": "Chronos-2", "chronos2_frn50k_mask": "Chronos-2 + mask",
}


def lab(r: str) -> str:
    return LABEL.get(r, r)


def md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("–")
                else:
                    cells.append(fmt.get(c, "{:.4f}").format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── C4: quantile-level pinball & coverage ──
w = pd.read_csv(R / "quantile_losses_lt_wide.csv").rename(columns={"run": "Model"})
w = w[~w.Model.str.contains("equ_")]
keep_q = ["0.1", "0.3", "0.5", "0.7", "0.9"]
c4 = w[["Model"] + [f"pin_{q}" for q in keep_q] + [f"cov_{q}" for q in keep_q]].copy()
c4["Model"] = c4.Model.map(lab)
c4 = c4.sort_values("pin_0.9")
c4.columns = ["Model"] + [f"PL {q}" for q in keep_q] + [f"Cov {q}" for q in keep_q]
(OUT / "r1_C4_quantile_losses.md").write_text(
    "**Table C4.** Quantile-level normalised pinball loss (PL, 2·mean pinball / mean outturn) "
    "and empirical coverage (Cov) on FreshRetailNet-LT, uncensored evaluation days, for the "
    "quantile-capable methods; sorted by PL at the 0.9 level.\n\n" + md(c4, {c: "{:.3f}" for c in c4.columns if c != "Model"}),
    encoding="utf-8")

# ── C5: median vs grid-mean signed error ──
b = pd.read_csv(R / "bias_mean_vs_median_lt.csv")
b = b[~b.run.str.contains("equ_")]
c5 = pd.DataFrame({"Model": b.run.map(lab),
                   "Signed median error (WPE, %)": b.wpe_median * 100,
                   "Signed mean error (grid mean, %)": b.wpe_grid_mean * 100,
                   "Median − mean component (pp)": b.median_minus_mean_share * 100,
                   "Tweedie mean error (%)": b.wpe_tweedie_mean * 100})
c5 = c5.sort_values("Signed median error (WPE, %)")
(OUT / "r1_C5_bias_definitions.md").write_text(
    "**Table C5.** Signed error of the median forecast (WPE, as in the main text) against the signed "
    "error of the mean of the predictive distribution approximated from the quantile grid "
    "(trapezoid over the reported levels, outer tails held at the extreme quantiles); LightGBM also "
    "with its Tweedie point forecast. FreshRetailNet-LT, uncensored days; adapted Chronos-2 configurations are seed 0, whereas Tables C1 and C3 average three seeds. Point-only methods have no grid mean.\n\n"
    + md(c5, {c: "{:+.2f}" for c in c5.columns if c != "Model"}), encoding="utf-8")

# ── C6: achieved service ──
s = pd.read_csv(R / "achieved_service_lt.csv")
s = s[~s.run.str.contains("equ_")]
rows = []
for run, g in s.groupby("run"):
    rec = {"Model": lab(run)}
    for q in [0.5, 0.7, 0.9]:
        gg = g[np.isclose(g.q, q)]
        if len(gg):
            rec[f"Fill {q}"] = float(gg.fill_rate.iloc[0])
            rec[f"CSL {q}"] = float(gg.cycle_service_level.iloc[0])
        else:
            rec[f"Fill {q}"] = np.nan
            rec[f"CSL {q}"] = np.nan
    rows.append(rec)
c6 = pd.DataFrame(rows).sort_values("Fill 0.9", ascending=False)
(OUT / "r1_C6_achieved_service.md").write_text(
    "**Table C6.** Achieved service at the nominal ordering quantiles 0.5, 0.7 and 0.9 "
    "(FreshRetailNet-LT, shelf life 2, lead time 0, observed-demand path): fill rate (units sold / "
    "units demanded) and cycle service level (CSL, share of series-days with no lost sales). "
    "Point-only methods order at the median only.\n\n" + md(c6, {c: "{:.3f}" for c in c6.columns if c != "Model"}),
    encoding="utf-8")

# ── C7: MASE weighting ──
m = pd.read_csv(R / "mase_weighting_lt.csv")
m = m[~m.run.str.contains("equ_")]
c7 = pd.DataFrame({"Model": m.run.map(lab), "MASE (equal weights)": m.mase_equal,
                   "MASE (weighted by scored days)": m.mase_day_weighted})
c7["Rank (equal)"] = c7["MASE (equal weights)"].rank().astype(int)
c7["Rank (weighted)"] = c7["MASE (weighted by scored days)"].rank().astype(int)
c7 = c7.sort_values("MASE (equal weights)")
(OUT / "r1_C7_mase_weighting.md").write_text(
    "**Table C7.** MASE with every series weighted equally (main text) and weighted by its number "
    "of uncensored evaluation days (1–7). No series has a zero MASE scale or zero scored days; adapted Chronos-2 configurations are seed 0.\n\n"
    + md(c7, {"MASE (equal weights)": "{:.3f}", "MASE (weighted by scored days)": "{:.3f}",
              "Rank (equal)": "{:.0f}", "Rank (weighted)": "{:.0f}"}), encoding="utf-8")

# ── B8: EQ uncensored ──
e = pd.read_csv(R / "eq_uncensored_compare_lt.csv")
e["Model"] = e.model.str.capitalize().replace({"Sba": "SBA", "Tsb": "TSB", "Ets": "ETS", "Snaive": "sNaive"})
e["Variant"] = e.variant.map({"eq": "all validation days", "equ": "uncensored days only"})
b8 = e[["Model", "Variant", "wape", "wpe", "wql", "cov0.7", "cov0.9", "cost_b1.0_q0.7", "cost_b3.0_q0.9"]].copy()
b8["wape"] *= 100; b8["wpe"] *= 100
b8.columns = ["Model", "Residuals from", "WAPE %", "WPE %", "WQL", "Cov 0.7", "Cov 0.9",
              "Cost b=1, q=0.7", "Cost b=3, q=0.9"]
(OUT / "r1_B8_eq_uncensored.md").write_text(
    "**Table B6.** Empirical quantiles estimated from all validation days (main specification) "
    "against uncensored validation days only (35,472 of 70,000 series-days). FreshRetailNet-LT, "
    "uncensored evaluation days; cost per unit of demand under the observed-demand path.\n\n"
    + md(b8, {"WAPE %": "{:.2f}", "WPE %": "{:+.2f}", "WQL": "{:.4f}", "Cov 0.7": "{:.3f}",
              "Cov 0.9": "{:.3f}", "Cost b=1, q=0.7": "{:.4f}", "Cost b=3, q=0.9": "{:.4f}"}),
    encoding="utf-8")

# ── B7: synthetic censoring ──
parts = []
for ds, name in [("lt", "LT (340 series)"), ("frn50k", "50K (5,874 series)")]:
    g = pd.read_csv(R / f"synthetic_censoring_{ds}_summary.csv")
    g3 = g[np.isclose(g.b, 3.0)].sort_values("true")
    tab = pd.DataFrame({"Model": g3.run.map(lab), "Cost vs true demand": g3["true"],
                        "Cost vs censored target": g3["censored"], "Inflation %": g3.delta_pct,
                        "Rank (true)": g3.rank_true, "Rank (censored)": g3.rank_censored})
    parts.append(f"*{name}, b = 3.0, q = 0.9, mean of five censoring draws.*\n\n"
                 + md(tab, {"Cost vs true demand": "{:.4f}", "Cost vs censored target": "{:.4f}",
                            "Inflation %": "{:.0f}", "Rank (true)": "{:.0f}", "Rank (censored)": "{:.0f}"}))
(OUT / "r1_B7_synthetic_censoring.md").write_text(
    "**Table B5.** Controlled censoring of the evaluation target on fully observed series: inventory "
    "cost per unit of demand when the same forecasts are scored against true demand and against "
    "synthetically censored sales.\n\n" + "\n\n".join(parts), encoding="utf-8")

# ── B9: demand-path sensitivity at the prespecified cells ──
d = pd.read_csv(R / "demand_target_sensitivity_lt.csv")
d = d[~d.run.str.contains("equ_")]
cells = [(0.3, 0.5), (1.0, 0.7), (3.0, 0.9)]
rows = []
for run, g in d.groupby("run"):
    rec = {"Model": lab(run)}
    for b_, q_ in cells:
        for path in ["observed", "profile", "uplift", "clean"]:
            gg = g[np.isclose(g.b, b_) & np.isclose(g.q, q_) & (g.demand_path == path)]
            rec[f"b={b_}, q={q_}: {path}"] = float(gg.cost_per_unit_demand.iloc[0]) if len(gg) else np.nan
    rows.append(rec)
b9 = pd.DataFrame(rows).sort_values("b=3.0, q=0.9: observed")
(OUT / "r1_B9_demand_paths.md").write_text(
    "**Table B7.** Inventory cost per unit of demand under the four evaluation demand paths "
    "(observed sales; hour-profile correction; hour-proportional uplift; the 340 stockout-free "
    "series) at the three prespecified cells, FreshRetailNet-LT, shelf life 2, lead time 0. Point-only methods appear "
    "at the median cell only. Sorted by observed-path cost at b = 3, q = 0.9.\n\n"
    + md(b9, {c: "{:.4f}" for c in b9.columns if c != "Model"}), encoding="utf-8")

# ── C8: coverage at the ordering levels (from per-series diagnostics) ──
qc = pd.read_csv(R / "quantile_calibration_lt.csv")
qc = qc[~qc.run.str.contains("equ_")]
piv = qc[qc.nominal.isin([0.5, 0.6, 0.7, 0.8, 0.9])].pivot(index="run", columns="nominal", values="coverage")
piv.insert(0, "Model", [lab(r) for r in piv.index])
piv.columns = ["Model"] + [f"Cov {c}" for c in piv.columns[1:]]
piv = piv.sort_values("Cov 0.9")
(OUT / "r1_C8_coverage.md").write_text(
    "**Table C8.** Empirical coverage of the nominal quantiles the replenishment policy orders from "
    "(FreshRetailNet-LT, uncensored evaluation days).\n\n"
    + md(piv.reset_index(drop=True), {c: "{:.3f}" for c in piv.columns if c != "Model"}), encoding="utf-8")

print("written:", sorted(p.name for p in OUT.glob("r1_*.md")))
