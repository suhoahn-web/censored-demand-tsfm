# -*- coding: utf-8 -*-
"""JORS R1 (second reviewer pass): per-origin table (B3), 50K inventory table (B4),
cheapest method on the b x q grid (C10). Writes markdown into Revision-1/tables/."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT.parent / "Target Journal" / "Journal of the Operational Research Society" / "Revision-1" / "tables"
LAB = {"c2lora_lt_raw": "Chronos-2 adapted at each origin", "chronos2_lt_raw": "Chronos-2 zero-shot",
       "tirex_lt_raw": "TiRex", "lightgbm_lt_raw": "LightGBM (retrained)", "etseq_lt_raw": "ETS + EQ",
       "moirai2_lt_raw": "Moirai-2", "chronos2_lt_mask": "Chronos-2 + mask", "tsbeq_lt_raw": "TSB + EQ",
       "sbaeq_lt_raw": "SBA + EQ", "crostoneq_lt_raw": "Croston + EQ", "snaiveeq_lt_raw": "sNaive + EQ",
       "ets_lt_raw": "ETS", "ttm_lt_raw": "TTM"}
def md(df):
    cols = list(df.columns)
    s = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    return s

# ---- B3: per-origin accuracy, correlations, cheapest ----
acc = pd.read_csv(R / "multi_origin_lt.csv")
tau = pd.read_csv(R / "multi_origin_tau_lt.csv")
tau = tau[(tau.demand_path == "observed") & np.isclose(tau.b, 1.0)]
inv = pd.read_csv(R / "multi_origin_inventory_lt.csv")
rows = []
for o in sorted(acc.origin.unique()):
    a = acc[acc.origin == o].set_index("method")
    t = tau[tau.origin == o].set_index("q")
    i3 = inv[(inv.origin == o) & np.isclose(inv.b, 3.0) & np.isclose(inv.q, 0.9)]
    obs = i3[i3.demand_path == "observed"].sort_values("cost_per_unit_demand")
    piv = i3.pivot(index="method", columns="demand_path", values="cost_per_unit_demand").dropna()
    sp = spearmanr(piv["observed"], piv["uplift"]).correlation
    spp = spearmanr(piv["observed"], piv["profile"]).correlation
    rows.append({"Origin": o, "Series": f"{int(obs.n_series.iloc[0]):,}",
                 "Adapted C2": f"{100*a.loc['c2lora_lt_raw','wape']:.1f}",
                 "Zero-shot C2": f"{100*a.loc['chronos2_lt_raw','wape']:.1f}",
                 "TiRex": f"{100*a.loc['tirex_lt_raw','wape']:.1f}",
                 "LightGBM": f"{100*a.loc['lightgbm_lt_raw','wape']:.1f}",
                 "ETS+EQ": f"{100*a.loc['etseq_lt_raw','wape']:.1f}",
                 "τ(WPE), q=0.2": f"{t.loc[0.2,'tau_wpe']:+.2f}",
                 "τ(WPE), q=0.9": f"{t.loc[0.9,'tau_wpe']:+.2f}",
                 "τ(WQL), q=0.9": f"{t.loc[0.9,'tau_wql']:+.2f}",
                 "Cheapest, b=3, q=0.9": LAB.get(obs.method.iloc[0], obs.method.iloc[0]),
                 "ρ obs–profile": f"{spp:+.2f}", "ρ obs–uplift": f"{sp:+.2f}"})
b3 = pd.DataFrame(rows)
cap = ("**Table B3.** Eight forecast origins on FreshRetailNet-LT: WAPE (%) of five methods on the series "
       "common to the origin, Kendall correlation of inventory cost with the signed median error (WPE) "
       "and with the weighted quantile loss (WQL) at $b = 1.0$ and the quantile shown, the cheapest "
       "method at $b = 3.0$, $q = 0.9$ (observed-demand path), and the Spearman correlation ρ between "
       "the observed-path cost ranking and the profile- and uplift-path rankings at that cell. Adapted Chronos-2 is LoRA r16, "
       "learning rate $10^{-3}$, seed 0, retrained at each origin; LightGBM is retrained at each origin "
       "with the official-origin recipe.\n\n")
(OUT / "table_B3_origins.md").write_text(cap + md(b3), encoding="utf-8")

# ---- B4: 50K inventory at the three cells ----
lab50 = {"lightgbm_frn50k_raw": "LightGBM", "ft_c2_50k_lora_r16_s0": "C2 LoRA r16", "etseq_frn50k_raw": "ETS+EQ",
         "tsbeq_frn50k_raw": "TSB+EQ", "crostoneq_frn50k_raw": "Croston+EQ", "sbaeq_frn50k_raw": "SBA+EQ",
         "ttmeq_frn50k_raw": "TTM+EQ", "tirex_50k_raw": "TiRex", "chronos2_50k_raw": "Chronos-2",
         "snaiveeq_frn50k_raw": "sNaive+EQ", "timesfm_50k_raw": "TimesFM", "moirai2_50k_raw": "Moirai-2",
         "chronos2_frn50k_mask": "Chronos-2 + mask", "tirex_frn50k_mask": "TiRex + mask",
         "moirai2_frn50k_mask": "Moirai-2 + mask", "timesfm_50k_mask": "TimesFM + mask",
         "ft_c2_rec_lora_r16_s0": "C2 LoRA r16, recovered training target",
         "chronos2_50krec_raw": "Chronos-2, recovered context", "moirai2_50krec_raw": "Moirai-2, recovered context"}
order = ["low_margin_waste_dominant", "medium_margin", "high_margin_stockout_dominant"]
parts = []
for ds in ["frn50k", "frn50krec"]:
    x = pd.read_csv(R / f"inventory_{ds}_sl2_L0.csv")
    x = x[x.is_fixed_cr & ~x.run.str.contains("equ_")]
    p = x.pivot(index="run", columns="scenario", values="cost_per_unit_demand")[order].dropna()
    p = p.sort_values(order[-1])
    d = pd.DataFrame({"Model": [lab50.get(r, r) for r in p.index],
                      "b=0.3, q=0.5": p[order[0]].map("{:.4f}".format).values,
                      "b=1.0, q=0.7": p[order[1]].map("{:.4f}".format).values,
                      "b=3.0, q=0.9": p[order[2]].map("{:.4f}".format).values})
    parts.append(d)
cap = ("**Table B4.** Inventory cost per unit of demand on FreshRetailNet-50K at the three prespecified "
       "cells (observed-demand path, shelf life 2, lead time 0, quantile-capable methods). Upper panel: "
       "models trained or run on censored sales; lower panel: the same model run on the recovered "
       "demand of Section 4.2.\n\n*Censored sales.*\n\n" + md(parts[0]) +
       "\n*Recovered demand.*\n\n" + md(parts[1]))
(OUT / "table_B4_50k_inventory.md").write_text(cap, encoding="utf-8")

# ---- C10: cheapest on the grid ----
g = pd.read_csv(R / "cost_q_grid_lt_sl2_L0.csv"); g = g[~g.run.str.contains("equ_")]
labg = {"ft_lora_r16_lr1e-3_s0": "C2 LoRA r16 (lr 1e-3)", "ft_lora_r16_lr1e-5_s0": "C2 LoRA r16 (lr 1e-5)",
        "ft_full_s0": "C2 full-FT", "lightgbm_lt_raw": "LightGBM", "etseq_lt_raw": "ETS+EQ", "tsbeq_lt_raw": "TSB+EQ",
        "timesfm_lt_mask": "TimesFM + mask", "tirex_lt_raw": "TiRex", "chronos2_lt_raw": "Chronos-2"}
rows = []
for b in sorted(g.b.unique()):
    rec = {"b": f"{b:g}"}
    for q in [0.5, 0.7, 0.9]:
        x = g[np.isclose(g.b, b) & np.isclose(g.q, q)].sort_values("cost_per_unit_demand").reset_index(drop=True)
        rk = lambda run: int(x.index[x.run == run][0]) + 1
        rec[f"q = {q}"] = (f"{labg.get(x.run[0], x.run[0])} ({x.cost_per_unit_demand[0]:.4f}); "
                           f"LightGBM {rk('lightgbm_lt_raw')}, adapted C2 {rk('ft_lora_r16_lr1e-3_s0')}")
    rows.append(rec)
c10 = pd.DataFrame(rows)
cap = (r"**Table C10.** Cheapest method on the $b \times q$ grid (FreshRetailNet-LT, observed-demand path, "
       "shelf life 2, lead time 0): for each lost margin $b$ and ordering quantile $q$, the method with the "
       "lowest simulated inventory cost per unit of demand (in parentheses) and the rank positions of "
       "LightGBM and of the best adapted Chronos-2 configuration (LoRA r16, lr $10^{-3}$) among the 29 methods "
       "at $q = 0.5$ and the 22 quantile-capable methods above it. Reading down a column varies the cost "
       "ratio alone; reading across a row varies the ordering quantile alone.\n\n")
(OUT / "table_C10_grid.md").write_text(cap + md(c10), encoding="utf-8")
print(b3.to_string()); print(c10.to_string()); print(parts[0].head(4).to_string()); print(parts[1].to_string())
