"""JORS R1: 답변서·수정 원고에 적은 핵심 수치를 results/ 산출물에서 재계산해 대조한다.

사용: python scripts/verify_r1_numbers.py
출력: 항목별 OK/MISMATCH, 마지막에 요약. 불일치는 수정 원고·답변서를 고쳐야 한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
checks = []


def chk(name, got, want, tol=0.0):
    ok = abs(float(got) - float(want)) <= tol if isinstance(want, (int, float)) else got == want
    checks.append((name, got, want, ok))
    print(("OK  " if ok else "MISMATCH") + f"  {name}: got {got}, paper {want}")


# ── Minor 10: evaluation-day counts, MASE scale ──
d = pd.read_csv(R / "eval_day_counts_lt.csv")
chk("LT series with 0 uncensored days", int((d.n_uncensored_days == 0).sum()), 0)
chk("LT series with 1 uncensored day", int((d.n_uncensored_days == 1).sum()), 1084)
chk("LT series with 7 uncensored days", int((d.n_uncensored_days == 7).sum()), 340)
chk("LT zero MASE scale", int((d.mase_scale == 0).sum()), 0)

# ── Minor 8: TTM+EQ ──
s = pd.read_csv(R / "summary_metrics.csv")
s = s[(s.dataset == "lt") & (s.eval_filter == "nostock")].set_index("run")
chk("TTM WAPE %", round(100 * s.loc["ttm_lt_raw", "wape"], 2), 28.08)
chk("TTM+EQ WAPE %", round(100 * s.loc["ttmeq_lt_raw", "wape"], 2), 26.33)
chk("TTM+EQ WQL", round(s.loc["ttmeq_lt_raw", "wql"], 3), 0.215)
inv = pd.read_csv(R / "inventory_lt_sl2_L0.csv")
inv = inv[inv.is_fixed_cr]
c = inv[(inv.run == "ttmeq_lt_raw")].set_index("scenario").cost_per_unit_demand
chk("TTM+EQ cost b=0.3", round(c["low_margin_waste_dominant"], 4), 0.0615)
chk("TTM+EQ cost b=1", round(c["medium_margin"], 4), 0.1056)
chk("TTM+EQ cost b=3", round(c["high_margin_stockout_dominant"], 4), 0.1547)

# ── §5.1 MCB (K=29) ──
m = pd.read_csv(R / "mcb_lt.csv", index_col=0)
chk("K methods", len(m), 29)
chk("full-FT mean rank", round(m.loc["ft_full_s0", "mean_rank"], 2), 12.57)
chk("LightGBM mean rank", round(m.loc["lightgbm_lt_raw", "mean_rank"], 2), 12.94)
chk("zero-shot C2 mean rank", round(m.loc["chronos2_lt_raw", "mean_rank"], 2), 13.62)
chk("TimesFM mean rank", round(m.loc["timesfm_lt_raw", "mean_rank"], 2), 13.79)
lora = [m.loc[r, "mean_rank"] for r in ["ft_lora_r8_s0", "ft_lora_r16_s0", "ft_lora_r32_s0"]]
chk("LoRA ranks min", round(min(lora), 2), 12.74); chk("LoRA ranks max", round(max(lora), 2), 12.81)
chk("critical distance", round(float(m.upper.iloc[0] - m.lower.iloc[0]), 2), 0.45)

# ── §5.5 property regimes (b=1, observed) ──
p = pd.read_csv(R / "property_regimes_lt.csv"); p = p[np.isclose(p.b, 1.0)].set_index("q")
chk("tau |WPE| q=0.2", round(p.loc[0.2, "tau_abs_wpe"], 2), 0.54, 0.011)
chk("tau |WPE| q=0.9", round(p.loc[0.9, "tau_abs_wpe"], 2), -0.39, 0.011)
chk("tau WQL q=0.2", round(p.loc[0.2, "tau_wql"], 2), 0.06, 0.011)
chk("tau WQL q=0.9", round(p.loc[0.9, "tau_wql"], 2), 0.51, 0.011)
chk("tau cov-err q=0.5", round(p.loc[0.5, "tau_abs_cov_err"], 2), 0.49, 0.011)
chk("tau cov-err q=0.9", round(p.loc[0.9, "tau_abs_cov_err"], 2), -0.12, 0.011)
pu = pd.read_csv(R / "property_regimes_lt_uplift.csv"); pu = pu[np.isclose(pu.b, 1.0)].set_index("q")
chk("uplift tau |WPE| q=0.2", round(pu.loc[0.2, "tau_abs_wpe"], 2), 0.66, 0.011)
chk("uplift tau |WPE| q=0.9", round(pu.loc[0.9, "tau_abs_wpe"], 2), -0.22, 0.011)
chk("uplift tau WQL q=0.9", round(pu.loc[0.9, "tau_wql"], 2), 0.47, 0.011)

# ── §5.8 demand paths ──
dt = pd.read_csv(R / "demand_target_sensitivity_lt.csv")
def sp(b, q, a="observed", bb="uplift"):
    x = dt[np.isclose(dt.b, b) & np.isclose(dt.q, q)].pivot(index="run", columns="demand_path", values="cost_per_unit_demand").dropna()
    return round(spearmanr(x[a], x[bb]).correlation, 2)
chk("Spearman obs-uplift b=3 q=0.9", sp(3.0, 0.9), 0.27, 0.011)
chk("Spearman obs-uplift b=1 q=0.7", sp(1.0, 0.7), 0.45, 0.011)
chk("Spearman obs-clean b=3 q=0.9", sp(3.0, 0.9, bb="clean"), 0.64, 0.011)
best_up = dt[np.isclose(dt.b, 3.0) & np.isclose(dt.q, 0.9) & (dt.demand_path == "uplift")].nsmallest(1, "cost_per_unit_demand").run.iloc[0]
chk("cheapest under uplift b=3 q=0.9", best_up, "tirex_lt_mask")

# ── B.7 synthetic censoring ──
g = pd.read_csv(R / "synthetic_censoring_lt_summary.csv"); g3 = g[np.isclose(g.b, 3.0)]
chk("LT masked C2 rank true", int(g3[g3.run == "chronos2_lt_mask"].rank_true.iloc[0]), 1)
chk("LT masked C2 rank censored", int(g3[g3.run == "chronos2_lt_mask"].rank_censored.iloc[0]), 11)
chk("LT LightGBM rank true", int(g3[g3.run == "lightgbm_lt_raw"].rank_true.iloc[0]), 15)
chk("LT LightGBM rank censored", int(g3[g3.run == "lightgbm_lt_raw"].rank_censored.iloc[0]), 2)
chk("LT Spearman true-censored", round(spearmanr(g3["true"], g3["censored"]).correlation, 2), 0.75, 0.011)
chk("LT inflation min %", int(round(g3.delta_pct.min())), 25, 1); chk("LT inflation max %", int(round(g3.delta_pct.max())), 107, 1)
g5 = pd.read_csv(R / "synthetic_censoring_frn50k_summary.csv"); g53 = g5[np.isclose(g5.b, 3.0)]
chk("50K Spearman", round(spearmanr(g53["true"], g53["censored"]).correlation, 2), 0.76, 0.011)
chk("50K ETS+EQ rank censored", int(g53[g53.run == "etseq_frn50k_raw"].rank_censored.iloc[0]), 9)
chk("50K LoRA rank censored", int(g53[g53.run == "ft_c2_50k_lora_r16_s0"].rank_censored.iloc[0]), 11)
meta = pd.read_csv(R / "synthetic_censoring_lt_meta.csv").iloc[0]
chk("LT terminal stockout share", round(meta.terminal_stockout_share, 2), 0.86, 0.011)
chk("LT censored-day share", round(meta.censored_day_share, 2), 0.49, 0.011)

# ── B.8 EQ uncensored ──
e = pd.read_csv(R / "eq_uncensored_compare_lt.csv").set_index(["model", "variant"])
chk("validation rows uncensored", int(e.loc[("ets", "equ"), "validation_rows"]), 35472)
d_b1 = [e.loc[(mm, "equ"), "cost_b1.0_q0.7"] - e.loc[(mm, "eq"), "cost_b1.0_q0.7"] for mm in ["croston", "sba", "tsb", "ets"]]
chk("EQ-uncensored cost delta b=1 (max)", round(max(d_b1), 3), -0.004, 0.0006)
cov = [e.loc[(mm, "equ"), "cov0.9"] for mm in ["croston", "sba", "tsb", "ets"]]
chk("EQ-uncensored coverage 0.9 (min)", round(min(cov), 2), 0.93, 0.006)

# ── C6 achieved service ──
a = pd.read_csv(R / "achieved_service_lt.csv"); a9 = a[np.isclose(a.q, 0.9)]
chk("fill q=0.9 min", round(a9.fill_rate.min(), 3), 0.957, 0.001); chk("fill q=0.9 max", round(a9.fill_rate.max(), 3), 0.985, 0.001)
chk("CSL q=0.9 min", round(a9.cycle_service_level.min(), 2), 0.83, 0.006); chk("CSL q=0.9 max", round(a9.cycle_service_level.max(), 2), 0.92, 0.006)

# ── B.2 / Table B2 multi-origin inventory ──
rk = pd.read_csv(R / "multi_origin_inventory_rank_lt.csv")
r3 = rk[np.isclose(rk.b, 3.0)].set_index("method")
chk("adapted C2 mean rank b=3", round(r3.loc["c2lora_lt_raw", "rank_mean"], 1), 2.1, 0.06)
chk("adapted C2 top1 freq b=3", round(r3.loc["c2lora_lt_raw", "top1_freq"], 3), 0.625, 0.001)
chk("TiRex mean rank b=3", round(r3.loc["tirex_lt_raw", "rank_mean"], 1), 2.4, 0.06)
chk("TiRex top1 freq b=3", round(r3.loc["tirex_lt_raw", "top1_freq"], 3), 0.25, 0.001)
chk("Chronos-2 mean rank b=3", round(r3.loc["chronos2_lt_raw", "rank_mean"], 1), 3.5, 0.06)
chk("LightGBM mean rank b=3", round(r3.loc["lightgbm_lt_raw", "rank_mean"], 1), 3.6, 0.06)
ms = pd.read_csv(R / "multi_origin_summary_lt.csv").set_index("method")
chk("adapted C2 WQL rank mean", round(ms.loc["c2lora_lt_raw", "rank_wql_mean"], 2), 1.0)
chk("adapted C2 WAPE rank mean", round(ms.loc["c2lora_lt_raw", "rank_wape_mean"], 1), 1.5, 0.06)
chk("LightGBM origin WAPE max %", round(100 * ms.loc["lightgbm_lt_raw", "wape_max"], 1), 36.5, 0.06)
r1 = rk[np.isclose(rk.b, 1.0)].set_index("method")
chk("ETS+EQ top1 freq b=1", round(r1.loc["etseq_lt_raw", "top1_freq"], 3), 0.625, 0.001)
ts = pd.read_csv(R / "multi_origin_tau_signed_lt.csv"); t1 = ts[(ts.path == "observed") & np.isclose(ts.b, 1.0)]
tt = pd.read_csv(R / "multi_origin_tau_lt.csv"); t1 = tt[(tt.demand_path == "observed") & np.isclose(tt.b, 1.0)]
chk("origins with tau(WQL,q=0.9)>0", int((t1[np.isclose(t1.q, 0.9)].tau_wql > 0).sum()), 8)
chk("origins with tau(WPE,q=0.2)<0", int((t1[np.isclose(t1.q, 0.2)].tau_wpe < 0).sum()), 7)
chk("origins with tau(WPE,q=0.9)>=0", int((t1[np.isclose(t1.q, 0.9)].tau_wpe >= 0).sum()), 6)

# ── B.5 lead time / terminal ──
def fixed(tag):
    x = pd.read_csv(R / f"inventory_lt_sl2_{tag}.csv"); x = x[x.is_fixed_cr]
    return x.pivot(index="run", columns="scenario", values="cost_per_unit_demand")
hi, lo = "high_margin_stockout_dominant", "low_margin_waste_dominant"
li = fixed("L1_indep"); lc = fixed("L1_comon"); tw = fixed("L0_termwaste")
chk("L1 indep LightGBM b=3", round(li.loc["lightgbm_lt_raw", hi], 4), 0.3475)
chk("L1 indep best FT b=3", round(li.loc[[i for i in li.index if i.startswith("ft_")], hi].min(), 3), 0.378, 0.0006)
chk("L1 comon LightGBM b=3", lc.loc["lightgbm_lt_raw", hi], 0.370, 0.0006)
chk("terminal-waste LightGBM b=3", round(tw.loc["lightgbm_lt_raw", hi], 4), 0.2092)
chk("terminal-waste ZS C2 b=3", round(tw.loc["chronos2_lt_raw", hi], 4), 0.2100)
chk("terminal-waste best low", tw[lo].idxmin(), "chronos2_lt_raw")

# ── C9 CPU timing ──
cc = pd.read_csv(R / "compute_cost_cpu_lt.csv").set_index("pipeline").wall_seconds
chk("classical s", int(cc["classical_5models_fit_forecast"]), 205); chk("EQ s", int(cc["empirical_quantile_calibration_5models"]), 261)
chk("LightGBM s", int(cc["lightgbm_tweedie_plus_9quantiles"]), 743)

mn = pd.read_csv(R / "recovery_mnar_validation_rates.csv").set_index("missing_rate")
chk("MNAR 0.1 WAPE %", round(100 * mn.loc[0.1, "wape"], 1), 25.5); chk("MNAR 0.5 WAPE %", round(100 * mn.loc[0.5, "wape"], 1), 29.8)
chk("MNAR 0.1 WPE %", round(100 * mn.loc[0.1, "wpe"], 1), 2.7); chk("MNAR 0.5 WPE %", round(100 * mn.loc[0.5, "wpe"], 1), 2.9)
chk("MNAR 0.5 hours (M)", round(mn.loc[0.5, "masked_hours"] / 1e6, 1), 16.9)


# ── second pass: grid axes (Table C10, §5.4) ──
gg = pd.read_csv(R / "cost_q_grid_lt_sl2_L0.csv"); gg = gg[~gg.run.str.contains("equ_")]
def cheapest(b, q):
    x = gg[np.isclose(gg.b, b) & np.isclose(gg.q, q)].sort_values("cost_per_unit_demand")
    return x.run.iloc[0]
chk("grid q=0.7 b=0.75 cheapest", cheapest(0.75, 0.7), "lightgbm_lt_raw")
chk("grid q=0.7 b=1.0 cheapest", cheapest(1.0, 0.7), "ft_lora_r16_lr1e-3_s0")
chk("grid b=1 q=0.5 cheapest", cheapest(1.0, 0.5), "etseq_lt_raw")
chk("grid b=1 q=0.4 cheapest", cheapest(1.0, 0.4), "tsbeq_lt_raw")
chk("grid b=1 q=0.6 cheapest", cheapest(1.0, 0.6), "lightgbm_lt_raw")
chk("grid b=1 q=0.8 cheapest", cheapest(1.0, 0.8), "ft_lora_r16_lr1e-3_s0")
chk("grid b=1 q=0.9 cheapest", cheapest(1.0, 0.9), "lightgbm_lt_raw")
x = gg[np.isclose(gg.b, 0.3) & np.isclose(gg.q, 0.5)].sort_values("cost_per_unit_demand").reset_index(drop=True)
chk("TiRex rank b=0.3 q=0.5", int(x.index[x.run == "tirex_lt_raw"][0]) + 1, 16)
x = gg[np.isclose(gg.b, 3.0) & np.isclose(gg.q, 0.9)].sort_values("cost_per_unit_demand").reset_index(drop=True)
chk("TiRex rank b=3 q=0.9", int(x.index[x.run == "tirex_lt_raw"][0]) + 1, 4)

# ── second pass: clean-path regimes (§5.8) ──
pc = pd.read_csv(R / "property_regimes_lt_clean.csv"); pc = pc[np.isclose(pc.b, 1.0)].set_index("q")
chk("clean tau |WPE| q=0.2", round(pc.loc[0.2, "tau_abs_wpe"], 2), 0.71, 0.011)
chk("clean tau |WPE| q=0.9", round(pc.loc[0.9, "tau_abs_wpe"], 2), -0.36, 0.011)
chk("clean tau WQL q=0.2", round(pc.loc[0.2, "tau_wql"], 2), -0.26, 0.011)
chk("clean tau WQL q=0.9", round(pc.loc[0.9, "tau_wql"], 2), 0.52, 0.011)

# ── second pass: seed SD of inventory cost (B.1) ──
sx = pd.read_csv(R / "inventory_lt_sl2_L0_seeds012.csv"); sx = sx[sx.is_fixed_cr & sx.run.str.startswith("ft_")]
sx["cfg"] = sx.run.str.replace(r"_s\d$", "", regex=True)
sd = sx.groupby(["cfg", "scenario"]).cost_per_unit_demand.std().dropna()
chk("seed SD min", round(sd.min(), 4), 0.0001, 0.00005); chk("seed SD max", round(sd.max(), 4), 0.0008, 0.00005)

# ── second pass: origin LightGBM co-movement (B.2) ──
mo = pd.read_csv(R / "multi_origin_lt.csv"); w = mo.pivot_table(index="method", columns="origin", values="wape")
chk("LightGBM-ETS+EQ origin corr", round(np.corrcoef(w.loc["lightgbm_lt_raw"], w.loc["etseq_lt_raw"])[0, 1], 2), 0.89, 0.011)
chk("LightGBM-C2 origin corr", round(np.corrcoef(w.loc["lightgbm_lt_raw"], w.loc["chronos2_lt_raw"])[0, 1], 2), 0.38, 0.011)
chk("ETS+EQ origin WAPE max %", round(100 * w.loc["etseq_lt_raw"].max(), 1), 38.1, 0.06)

# ── second pass: 50K recovery effect on cost (§5.3.2, Table B4) ──
def cells(ds, run):
    x = pd.read_csv(R / f"inventory_{ds}_sl2_L0.csv"); x = x[x.is_fixed_cr & (x.run == run)]
    return x.set_index("scenario").cost_per_unit_demand
lo, rec = cells("frn50k", "ft_c2_50k_lora_r16_s0"), cells("frn50krec", "ft_c2_rec_lora_r16_s0")
d = (100 * (lo - rec) / lo); chk("50K LoRA recovery gain min %", round(d.min(), 1), 0.8, 0.06); chk("50K LoRA recovery gain max %", round(d.max(), 1), 11.7, 0.06)
zs = [(cells("frn50k", a), cells("frn50krec", b)) for a, b in [("chronos2_50k_raw", "chronos2_50krec_raw"), ("moirai2_50k_raw", "moirai2_50krec_raw")]]
dz = pd.concat([100 * (a - b) / a for a, b in zs]); chk("50K zero-shot recovery gain min %", round(dz.min(), 1), 11.5, 0.06); chk("50K zero-shot recovery gain max %", round(dz.max(), 1), 19.6, 0.06)

# ── second pass: B.8 numbers ──
e2 = pd.read_csv(R / "eq_uncensored_compare_lt.csv").set_index(["model", "variant"])
chk("B8 WPE uncensored min %", round(100 * min(e2.loc[(m, "equ"), "wpe"] for m in ["croston", "sba", "tsb", "ets", "snaive"]), 1), 1.5, 0.06)
chk("B8 WPE uncensored max %", round(100 * max(e2.loc[(m, "equ"), "wpe"] for m in ["croston", "sba", "tsb", "ets", "snaive"]), 1), 3.8, 0.06)
chk("B8 sNaive cost delta b=3", round(e2.loc[("snaive", "equ"), "cost_b3.0_q0.9"] - e2.loc[("snaive", "eq"), "cost_b3.0_q0.9"], 3), 0.024, 0.0006)


# ── third pass: hour-profile demand path (§3.3, §5.8, Table B7) ──
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.demand_target_sensitivity import build_paths
_paths = build_paths("lt")
_ratio = _paths["profile"].to_numpy().sum() / _paths["observed"].to_numpy().sum()
chk("profile path demand ratio (20%)", round(100 * (_ratio - 1)), 20)
chk("uplift path demand ratio (31%)", round(100 * (_paths["uplift"].to_numpy().sum() / _paths["observed"].to_numpy().sum() - 1)), 31)
pp = pd.read_csv(R / "property_regimes_lt_profile.csv"); pp = pp[np.isclose(pp.b, 1.0)].set_index("q")
chk("profile tau |WPE| q=0.2", round(pp.loc[0.2, "tau_abs_wpe"], 2), 0.66, 0.011)
chk("profile tau |WPE| q=0.9", round(pp.loc[0.9, "tau_abs_wpe"], 2), -0.36, 0.011)
chk("profile tau WQL q=0.2", round(pp.loc[0.2, "tau_wql"], 2), -0.03, 0.011)
chk("profile tau WQL q=0.9", round(pp.loc[0.9, "tau_wql"], 2), 0.52, 0.011)
chk("Spearman obs-profile b=3 q=0.9", sp(3.0, 0.9, bb="profile"), 0.82, 0.011)
_med = [sp(b, 0.5, bb="profile") for b in [0.3, 1.0, 3.0]]
chk("Spearman obs-profile median min", min(_med), 0.78, 0.011); chk("Spearman obs-profile median max", max(_med), 0.92, 0.011)
_q7 = [sp(b, 0.7, bb="profile") for b in [0.3, 1.0, 3.0]]
chk("Spearman obs-profile q=0.7 min", min(_q7), 0.59, 0.011); chk("Spearman obs-profile q=0.7 max", max(_q7), 0.66, 0.011)
_pr = dt[np.isclose(dt.b, 3.0) & np.isclose(dt.q, 0.9) & (dt.demand_path == "profile")].nsmallest(3, "cost_per_unit_demand").run.tolist()
chk("cheapest under profile b=3 q=0.9", _pr[0], "ft_lora_r16_lr1e-5_s0")
chk("second under profile b=3 q=0.9", _pr[1], "tirex_lt_raw")
chk("third under profile b=3 q=0.9", _pr[2], "ft_lora_r16_lr1e-3_s0")

# ── third pass: profile path across origins (B.2, Table B3) ──
_mo = pd.read_csv(R / "multi_origin_inventory_lt.csv")
_i3 = _mo[np.isclose(_mo.b, 3.0) & np.isclose(_mo.q, 0.9)]
_rho, _best = [], []
for _o in sorted(_i3.origin.unique()):
    _x = _i3[_i3.origin == _o].pivot(index="method", columns="demand_path", values="cost_per_unit_demand").dropna()
    _rho.append(round(spearmanr(_x["observed"], _x["profile"]).correlation, 2)); _best.append(_x["profile"].idxmin())
chk("origin rho obs-profile min", min(_rho), -0.21, 0.011); chk("origin rho obs-profile max", max(_rho), 0.99, 0.011)
chk("origin cheapest under profile in {TiRex, ETS+EQ}", all(b in ("tirex_lt_raw", "etseq_lt_raw") for b in _best), True)
_mt = pd.read_csv(R / "multi_origin_tau_lt.csv")
_mt = _mt[(_mt.demand_path == "profile") & np.isclose(_mt.b, 1.0) & np.isclose(_mt.q, 0.9)]
chk("origins with positive WQL tau q=0.9 under profile", int((_mt.tau_wql > 0).sum()), 6)

# ── third pass: shelf life 3, lead time 1 (B.5) ──
def _ranks(tag):
    x = pd.read_csv(R / f"inventory_{tag}.csv"); x = x[x.is_fixed_cr & ~x.run.str.contains("equ_")]
    out = {}
    for sc in ["low_margin_waste_dominant", "medium_margin", "high_margin_stockout_dominant"]:
        s = x[x.scenario == sc].sort_values("cost_per_unit_demand").reset_index(drop=True)
        out[sc] = (int(s.index[s.run == "lightgbm_lt_raw"][0]) + 1, float(s.set_index("run").cost_per_unit_demand["lightgbm_lt_raw"]),
                   float(s.set_index("run").cost_per_unit_demand["ft_lora_r16_lr1e-3_s0"]))
    return out
_ind, _com = _ranks("lt_sl3_L1_indep"), _ranks("lt_sl3_L1_comon")
chk("sl3 L1 indep LightGBM ranks", tuple(v[0] for v in _ind.values()), (1, 10, 8))
chk("sl3 L1 indep LightGBM waste cost", round(_ind["low_margin_waste_dominant"][1], 4), 0.0554, 0.00006)
chk("sl3 L1 indep LightGBM medium/shortage", (round(_ind["medium_margin"][1], 4), round(_ind["high_margin_stockout_dominant"][1], 4)), (0.1087, 0.1667))
chk("sl3 L1 indep adapted C2 medium/shortage", (round(_ind["medium_margin"][2], 4), round(_ind["high_margin_stockout_dominant"][2], 4)), (0.1016, 0.1410))
chk("sl3 L1 comon LightGBM ranks", tuple(v[0] for v in _com.values()), (1, 2, 1))
chk("sl3 L1 comon medium LightGBM/adapted", (round(_com["medium_margin"][1], 4), round(_com["medium_margin"][2], 4)), (0.0924, 0.0922))
chk("sl3 L1 comon shortage LightGBM/adapted", (round(_com["high_margin_stockout_dominant"][1], 4), round(_com["high_margin_stockout_dominant"][2], 4)), (0.1234, 0.1302))

n_bad = sum(1 for c in checks if not c[3])
print(f"\n{len(checks)} checks, {n_bad} mismatches")
