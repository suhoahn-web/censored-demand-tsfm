"""OR Spectrum 개정본에서 새로 추가한 수치를 산출물과 대조한다.

기존 `verify_draft_numbers.py`(46건)는 그대로 두고, 이 스크립트가 신규 분석
(다중 origin, 비용×분위수 격자, 캘리브레이션, EQ 분해, 군집 부트스트랩,
수요 경로 민감도, 데이터셋 중복)의 본문 수치를 검증한다.

사용: python scripts/verify_orsp_numbers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = ROOT / "data" / "processed"
KEY = ["store_id", "product_id"]

rows = []


def chk(label, draft, actual, tol=5e-3):
    ok = actual is not None and abs(float(draft) - float(actual)) <= tol
    rows.append((ok, label, draft, actual))


def head(t):
    rows.append((None, t, None, None))


# ---------- 5.5 / Table 9: 체제별 예측 속성 ----------
head("5.5 forecast-property regimes (property_regimes_lt.csv)")
pr = pd.read_csv(R / "property_regimes_lt.csv")


def tau(b, q, prop, field="tau_"):
    s = pr[np.isclose(pr.b, b) & np.isclose(pr.q, q)]
    return float(s[f"{field}{prop}"].iloc[0]) if len(s) else None


for b, q, prop, val in [
        (1.0, 0.2, "abs_wpe", 0.53), (1.0, 0.9, "abs_wpe", -0.40),
        (1.0, 0.2, "wql", 0.06), (1.0, 0.9, "wql", 0.48),
        (0.3, 0.2, "abs_wpe", 0.48), (0.3, 0.9, "abs_wpe", -0.35),
        (0.3, 0.2, "wql", 0.08), (0.3, 0.9, "wql", 0.37),
        (3.0, 0.2, "abs_wpe", 0.56), (3.0, 0.9, "abs_wpe", -0.36),
        (3.0, 0.2, "wql", 0.03), (3.0, 0.9, "wql", 0.63),
        (1.0, 0.5, "abs_cov_err", 0.49), (1.0, 0.9, "abs_cov_err", -0.10)]:
    chk(f"tau({prop}) b={b} q={q}", val, tau(b, q, prop))
for b, q, prop, lo, hi in [
        (1.0, 0.2, "abs_wpe", 0.48, 0.57), (1.0, 0.9, "abs_wpe", -0.42, -0.32),
        (1.0, 0.2, "wql", 0.00, 0.09), (1.0, 0.9, "wql", 0.42, 0.54),
        (1.0, 0.5, "abs_cov_err", 0.38, 0.59), (1.0, 0.9, "abs_cov_err", -0.22, 0.01)]:
    chk(f"CI lo tau({prop}) b={b} q={q}", lo, tau(b, q, prop, "tau_") and
        float(pr[np.isclose(pr.b, b) & np.isclose(pr.q, q)][f"tau_{prop}_lo"].iloc[0]))
    chk(f"CI hi tau({prop}) b={b} q={q}", hi,
        float(pr[np.isclose(pr.b, b) & np.isclose(pr.q, q)][f"tau_{prop}_hi"].iloc[0]))
w = pr[np.isclose(pr.b, 1.0)]["tau_wape"]
chk("WAPE tau min (b=1)", 0.29, float(w.min()))
chk("WAPE tau max (b=1)", 0.62, float(w.max()), tol=0.02)

# ---------- 5.6 / Table 11: EQ 분해 (사전지정 분위수 기준) ----------
head("5.6 EQ decomposition at prespecified quantiles (eq_ablation_lt.csv)")
eq = pd.read_csv(R / "eq_ablation_lt.csv")
PRE_Q = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}


def eqcost(variant, b):
    """A/B는 유일한 운영점 q=0.5, C/D는 사전지정 q — 테스트셋 q 선택 없음."""
    q = 0.5 if variant in ("A_point", "B_recenter") else PRE_Q[b]
    s = eq[(eq.variant == variant) & np.isclose(eq.b, b) & np.isclose(eq.q, q)]
    return float(s.cost_per_unit_demand.mean())


c1 = {v: eqcost(v, 1.0) for v in ["A_point", "B_recenter", "C_spread", "D_fullEQ"]}
c3 = {v: eqcost(v, 3.0) for v in ["A_point", "B_recenter", "C_spread", "D_fullEQ"]}
for v, val in zip(c1, [0.1953, 0.1566, 0.1255, 0.1063]):
    chk(f"cost b=1 {v}", val, c1[v], tol=5e-4)
for v, val in zip(c3, [0.5531, 0.4321, 0.1701, 0.1619]):
    chk(f"cost b=3 {v}", val, c3[v], tol=5e-4)
chk("spread share of gain, b=1 (79%)", 0.79,
    (c1["A_point"] - c1["C_spread"]) / (c1["A_point"] - c1["D_fullEQ"]), tol=0.01)
chk("spread share of gain, b=3 (98%)", 0.98,
    (c3["A_point"] - c3["C_spread"]) / (c3["A_point"] - c3["D_fullEQ"]), tol=0.01)
chk("recentre share of gain, b=1 (43%)", 0.435,
    (c1["A_point"] - c1["B_recenter"]) / (c1["A_point"] - c1["D_fullEQ"]), tol=0.01)
chk("recentre share of gain, b=3 (31%)", 0.31,
    (c3["A_point"] - c3["B_recenter"]) / (c3["A_point"] - c3["D_fullEQ"]), tol=0.01)
chk("A->C cost reduction b=1 (36%)", 0.357,
    (c1["A_point"] - c1["C_spread"]) / c1["A_point"], tol=0.005)
chk("A->C cost reduction b=3 (69%)", 0.692,
    (c3["A_point"] - c3["C_spread"]) / c3["A_point"], tol=0.005)
# 구조적 항등성: q=0.5에서 A==C, B==D
at5 = eq[np.isclose(eq.q, 0.5)].pivot_table(index=["model", "b"],
                                            columns="variant",
                                            values="cost_per_unit_demand")
chk("identity max|A-C| at q=0.5", 0.0,
    float(np.max(np.abs(at5["A_point"] - at5["C_spread"]))), tol=1e-12)
chk("identity max|B-D| at q=0.5", 0.0,
    float(np.max(np.abs(at5["B_recenter"] - at5["D_fullEQ"]))), tol=1e-12)
acc = eq.drop_duplicates(["model", "variant"]).groupby("variant")[
    ["wape", "wpe", "cov0.9"]].mean()
chk("variant A WAPE", 0.2786, acc.loc["A_point", "wape"], tol=5e-4)
chk("variant C WAPE (= A)", 0.2786, acc.loc["C_spread", "wape"], tol=5e-4)
chk("variant A WPE", -0.0798, acc.loc["A_point", "wpe"], tol=5e-4)
chk("variant C WPE (= A)", -0.0798, acc.loc["C_spread", "wpe"], tol=5e-4)
chk("variant B WAPE", 0.2691, acc.loc["B_recenter", "wape"], tol=5e-4)
chk("variant B WPE", -0.0193, acc.loc["B_recenter", "wpe"], tol=5e-4)
chk("variant C coverage @0.9", 0.882, acc.loc["C_spread", "cov0.9"], tol=1e-3)
chk("variant D coverage @0.9", 0.901, acc.loc["D_fullEQ", "cov0.9"], tol=1e-3)

# ---------- Table 10: 커버리지 ----------
head("5.5 quantile coverage (quantile_calibration_lt.csv)")
cal = pd.read_csv(R / "quantile_calibration_lt.csv")


def cov(run, q):
    s = cal[(cal.run == run) & np.isclose(cal.nominal, q)]
    return float(s.coverage.iloc[0]) if len(s) else None


for run, val in [("etseq_lt_raw", 0.904), ("crostoneq_lt_raw", 0.903),
                 ("sbaeq_lt_raw", 0.901), ("lightgbm_lt_raw", 0.876),
                 ("tirex_lt_raw", 0.922)]:
    chk(f"coverage @0.9 {run}", val, cov(run, 0.9), tol=1e-3)

# ---------- Table 12: 군집 부트스트랩 ----------
head("5.7 cluster bootstrap (cluster_bootstrap_lt_store_id.csv)")
cb = pd.read_csv(R / "cluster_bootstrap_lt_store_id.csv")
cost = cb[cb.metric.str.startswith("cost_per_unit")].set_index("run")
chk("b=3 q=0.9 adapted C2 cost", 0.1094, cost.loc["ft_lora_r16_lr1e-3_s0", "point"],
    tol=5e-4)
chk("b=3 q=0.9 adapted C2 lo", 0.1053, cost.loc["ft_lora_r16_lr1e-3_s0", "lo"], tol=1e-3)
chk("b=3 q=0.9 adapted C2 hi", 0.1139, cost.loc["ft_lora_r16_lr1e-3_s0", "hi"], tol=1e-3)
chk("b=3 q=0.9 diff vs ETS+EQ", -0.0335,
    cost.loc["ft_lora_r16_lr1e-3_s0", "diff_vs_ref"], tol=5e-4)
chk("b=3 q=0.9 ETS+EQ cost", 0.1429, cost.loc["etseq_lt_raw", "point"], tol=5e-4)
chk("b=3 q=0.9 TiRex cost", 0.1129, cost.loc["tirex_lt_raw", "point"], tol=5e-4)
fr = cb[cb.metric.str.startswith("fill_rate")].set_index("run")["point"]
chk("q=0.9 min fill rate", 0.957, float(fr.min()), tol=1e-3)
chk("q=0.9 max fill rate", 0.985, float(fr.max()), tol=1e-3)

# ---------- Table 6: 다중 origin ----------
head("5.1 multiple origins (multi_origin_summary_lt.csv)")
mo = pd.read_csv(R / "multi_origin_summary_lt.csv").set_index("method")
chk("Chronos-2 WAPE mean", 0.2674, mo.loc["chronos2_lt_raw", "wape_mean"], tol=5e-4)
chk("Chronos-2 WAPE sd", 0.0155, mo.loc["chronos2_lt_raw", "wape_sd"], tol=5e-4)
chk("Chronos-2 WAPE min", 0.2455, mo.loc["chronos2_lt_raw", "wape_min"], tol=5e-4)
chk("Chronos-2 WAPE max", 0.2980, mo.loc["chronos2_lt_raw", "wape_max"], tol=5e-4)
chk("Chronos-2 mean rank", 3.25, mo.loc["chronos2_lt_raw", "rank_wape_mean"], tol=1e-2)
chk("Chronos-2 rank sd", 2.25, mo.loc["chronos2_lt_raw", "rank_wape_sd"], tol=1e-2)
chk("Chronos-2 WQL rank (1st everywhere)", 1.0,
    mo.loc["chronos2_lt_raw", "rank_wql_mean"], tol=1e-6)
chk("ETS+EQ WAPE sd", 0.0424, mo.loc["etseq_lt_raw", "wape_sd"], tol=5e-4)
chk("Croston WAPE sd", 0.0219, mo.loc["croston_lt_raw", "wape_sd"], tol=5e-4)
chk("Croston+EQ WAPE sd", 0.0379, mo.loc["crostoneq_lt_raw", "wape_sd"], tol=5e-4)
chk("Croston WAPE mean", 0.2961, mo.loc["croston_lt_raw", "wape_mean"], tol=5e-4)
chk("Croston+EQ WAPE mean", 0.2896, mo.loc["crostoneq_lt_raw", "wape_mean"], tol=5e-4)
det = pd.read_csv(R / "multi_origin_lt.csv")
piv = det.pivot(index="origin", columns="method", values="wape")
sds = piv.std()
zs = ["chronos2_lt_raw", "moirai2_lt_raw", "tirex_lt_raw"]
eqs = ["etseq_lt_raw", "crostoneq_lt_raw", "sbaeq_lt_raw", "tsbeq_lt_raw"]
chk("zero-shot WAPE sd min (1.5pp)", 0.015, float(sds[zs].min()), tol=1e-3)
chk("zero-shot WAPE sd max (1.7pp)", 0.017, float(sds[zs].max()), tol=1e-3)
chk("EQ WAPE sd min (3.7pp)", 0.037, float(sds[eqs].min()), tol=1e-3)
chk("EQ WAPE sd max (4.2pp)", 0.042, float(sds[eqs].max()), tol=1e-3)
chk("n_series min per origin", 5500, float(det.groupby("origin").n_series.max().min()),
    tol=1)
chk("n_series max per origin", 6849, float(det.groupby("origin").n_series.max().max()),
    tol=1)

# ---------- Table 13 / 5.8: 수요 경로 ----------
head("5.8 demand-path sensitivity (demand_target_sensitivity_lt.csv)")
dp = pd.read_csv(R / "demand_target_sensitivity_lt.csv")
s = dp[np.isclose(dp.q, 0.9) & np.isclose(dp.b, 3.0)]
pv = s.pivot(index="run", columns="demand_path", values="cost_per_unit_demand").dropna()
chk("Spearman observed-uplift, b=3 q=0.9", 0.29,
    float(pv.rank().corr(method="spearman").loc["observed", "uplift"]), tol=0.02)
chk("clean subsample series", 340, float(dp[dp.demand_path == "clean"].n_series.iloc[0]),
    tol=0)
up = pd.read_csv(R / "property_regimes_lt_uplift.csv")


def utau(q, prop, f="tau_"):
    r = up[np.isclose(up.b, 1.0) & np.isclose(up.q, q)]
    return float(r[f"{f}{prop}"].iloc[0])


chk("uplift tau(|WPE|) q=0.2", 0.65, utau(0.2, "abs_wpe"))
chk("uplift tau(|WPE|) q=0.9", -0.22, utau(0.9, "abs_wpe"))
chk("uplift tau(WQL) q=0.2", -0.02, utau(0.2, "wql"))
chk("uplift tau(WQL) q=0.9", 0.45, utau(0.9, "wql"))

# ---------- Table 4 / 3.2: 데이터셋 중복 ----------
head("3.2 release overlap and censoring intensity")
lt_tr = pd.read_parquet(P / "lt_daily_train.parquet",
                        columns=KEY + ["dt", "stock_hour6_22_cnt"])
lt_ev = pd.read_parquet(P / "lt_daily_eval.parquet", columns=KEY + ["dt"])
k5 = pd.concat([pd.read_parquet(P / "frn50k_daily_train.parquet", columns=KEY + ["dt"]),
                pd.read_parquet(P / "frn50k_daily_eval.parquet", columns=KEY + ["dt"])])
lt = pd.concat([lt_tr[KEY + ["dt"]], lt_ev])
chk("common stores", 898, len(set(lt.store_id) & set(k5.store_id)), tol=0)
chk("common products", 576, len(set(lt.product_id) & set(k5.product_id)), tol=0)
lt_sp = set(map(tuple, lt[KEY].drop_duplicates().to_numpy()))
k5_sp = set(map(tuple, k5[KEY].drop_duplicates().to_numpy()))
chk("common store-item series", 1686, len(lt_sp & k5_sp), tol=0)
chk("common dates", 97, len(set(lt.dt.unique()) & set(k5.dt.unique())), tol=0)
ev_keys = lt_ev[KEY].drop_duplicates()
k5_ev = pd.read_parquet(P / "frn50k_daily_eval.parquet", columns=KEY).drop_duplicates()
chk("common evaluation series", 679,
    len(set(map(tuple, ev_keys.to_numpy())) & set(map(tuple, k5_ev.to_numpy()))), tol=0)
inter = lt_sp & k5_sp
a = set(map(tuple, lt[lt.set_index(KEY).index.isin(inter)][KEY + ["dt"]].to_numpy()))
b_ = set(map(tuple, k5[k5.set_index(KEY).index.isin(inter)][KEY + ["dt"]].to_numpy()))
chk("common store-item-day observations", 57551, len(a & b_), tol=0)
tr_ev = lt_tr.merge(ev_keys, on=KEY)
rate = tr_ev.groupby(KEY)["stock_hour6_22_cnt"].apply(lambda s: (s > 0).mean())
chk("median share of stockout days", 0.46, float(rate.median()), tol=5e-3)
hrs = tr_ev.loc[tr_ev.stock_hour6_22_cnt > 0, "stock_hour6_22_cnt"]
chk("median stockout hours on stockout days", 6, float(hrs.median()), tol=0)
chk("share of all-day stockouts", 0.085, float((hrs >= 16).mean()), tol=1e-3)

# ---------- B.7: 복원 파이프라인 MNAR 검증 ----------
head("B.7 recovery MNAR validation (recovery_mnar_validation.csv)")
mn = pd.read_csv(R / "recovery_mnar_validation.csv").set_index("check")["value"]
chk("MNAR masked hours (millions)", 9.16, float(mn["masked_hours"]) / 1e6, tol=5e-3)
chk("MNAR recovery WAPE", 0.276, float(mn["wape"]), tol=1e-3)
chk("MNAR recovery WPE", 0.014, float(mn["wpe"]), tol=1e-3)

# ---------- B.9: 패밀리 균형 강건성 ----------
head("B.9 family robustness (rq1_family_representative / _leave_one_family_out)")
rep = pd.read_csv(R / "rq1_family_representative.csv")
r1 = rep[np.isclose(rep.b, 1.0)]
chk("rep tau bias q=0.2", 0.64,
    float(r1[np.isclose(r1.q, 0.2)].tau_abs_wpe.iloc[0]), tol=5e-3)
chk("rep tau bias q=0.9", -0.29,
    float(r1[np.isclose(r1.q, 0.9)].tau_abs_wpe.iloc[0]), tol=5e-3)
chk("rep tau wql q=0.2", 0.00,
    float(r1[np.isclose(r1.q, 0.2)].tau_wql.iloc[0]), tol=5e-3)
chk("rep tau wql q=0.9", 0.36,
    float(r1[np.isclose(r1.q, 0.9)].tau_wql.iloc[0]), tol=5e-3)
lofo = pd.read_csv(R / "rq1_leave_one_family_out.csv")
l1 = lofo[np.isclose(lofo.b, 1.0)]
lo = l1[l1.q <= 0.3].groupby("excluded_family").tau_abs_bias.mean()
hi = l1[l1.q >= 0.9].groupby("excluded_family").tau_abs_bias.mean()
chk("LOFO low-q bias min (+0.38)", 0.38, float(lo.min()), tol=0.01)
chk("LOFO low-q bias max (+0.69)", 0.69, float(lo.max()), tol=0.01)
chk("LOFO high-q bias all negative", 1.0, float((hi < 0).mean()), tol=0)
chk("LOFO families count", 9, float(lofo.excluded_family.nunique()), tol=0)

# ---------- B.10: 공통 패널 ----------
head("B.10 common panel (multi_origin_common_panel_*)")
ids = pd.read_csv(R / "multi_origin_common_panel_ids.csv")
chk("common panel series", 3650, float(len(ids)), tol=0)
chk("common panel stores", 815, float(ids.store_id.nunique()), tol=0)
chk("common panel products", 146, float(ids.product_id.nunique()), tol=0)
cp = pd.read_csv(R / "multi_origin_common_panel_summary.csv").set_index("method")
chk("CP Chronos-2 WAPE min", 0.2464, cp.loc["chronos2_lt_raw", "wape_min"], tol=5e-4)
chk("CP Chronos-2 WAPE max", 0.2878, cp.loc["chronos2_lt_raw", "wape_max"], tol=5e-4)
chk("CP Chronos-2 WAPE sd (1.4pp)", 0.0143, cp.loc["chronos2_lt_raw", "wape_sd"],
    tol=5e-4)
chk("CP Chronos-2 WQL rank 1.0", 1.0, cp.loc["chronos2_lt_raw", "rank_wql_mean"],
    tol=1e-9)
zs = ["chronos2_lt_raw", "moirai2_lt_raw", "tirex_lt_raw"]
eqs = ["etseq_lt_raw", "crostoneq_lt_raw", "sbaeq_lt_raw", "tsbeq_lt_raw"]
chk("CP zero-shot WAPE sd mean (1.4pp)", 0.0139, float(cp.loc[zs, "wape_sd"].mean()),
    tol=5e-4)
chk("CP classical EQ WAPE sd mean (3.8pp)", 0.0383,
    float(cp.loc[eqs, "wape_sd"].mean()), tol=5e-4)

# ---------- B.11: 상품 군집 부트스트랩 ----------
head("B.11 product-cluster bootstrap")
pb3 = pd.read_csv(R / "cluster_bootstrap_lt_product_id_b3q09.csv")
c3p = pb3[pb3.metric.str.startswith("cost_per_unit")].set_index("run")
chk("prod b=3 diff", -0.0335, c3p.loc["ft_lora_r16_lr1e-3_s0", "diff_vs_ref"],
    tol=5e-4)
chk("prod b=3 diff lo", -0.0614, c3p.loc["ft_lora_r16_lr1e-3_s0", "diff_lo"],
    tol=1e-3)
chk("prod b=3 diff hi", -0.0128, c3p.loc["ft_lora_r16_lr1e-3_s0", "diff_hi"],
    tol=1e-3)
pb1 = pd.read_csv(R / "cluster_bootstrap_lt_product_id_b1q07.csv")
c1p = pb1[pb1.metric.str.startswith("cost_per_unit")].set_index("run")
chk("prod b=1 diff", -0.0034, c1p.loc["ft_lora_r16_lr1e-3_s0", "diff_vs_ref"],
    tol=5e-4)
chk("prod b=1 diff hi (>0)", 0.0044, c1p.loc["ft_lora_r16_lr1e-3_s0", "diff_hi"],
    tol=1e-3)

# ---------- Task 1: origin 날짜 ----------
head("multi-origin exact dates (multi_origin_exact_dates.csv)")
od = pd.read_csv(R / "multi_origin_exact_dates.csv")
chk("origin count", 8, float(len(od)), tol=0)
gaps = od.days_since_previous.dropna().astype(int).tolist()
chk("gaps 56-day count", 5, float(sum(g == 56 for g in gaps)), tol=0)
chk("gaps 28-day count", 2, float(sum(g == 28 for g in gaps)), tol=0)
chk("first origin 2024-07-07", 1.0,
    float(od.forecast_origin.iloc[0] == "2024-07-07"), tol=0)
chk("last origin 2025-06-08", 1.0,
    float(od.forecast_origin.iloc[-1] == "2025-06-08"), tol=0)

# ---------- 보고 ----------
bad = 0
for ok, label, draft, actual in rows:
    if ok is None:
        print(f"\n=== {label} ===")
        continue
    if not ok:
        bad += 1
    fa = "None" if actual is None else f"{float(actual):.4f}"
    print(f"[{'OK  ' if ok else 'FAIL'}] {label:<48s} draft={float(draft):<10.4f} "
          f"results={fa}")
n = sum(1 for r in rows if r[0] is not None)
print("\n" + "=" * 70)
print(f"신규 {n}건 중 불일치 {bad}건")
sys.exit(1 if bad else 0)
