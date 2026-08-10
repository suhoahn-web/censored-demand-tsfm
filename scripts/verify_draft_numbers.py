"""초안의 수치를 결과 파일과 기계적으로 대조 (마지막 검산).

논문에 인용된 값들을 결과 CSV/parquet에서 다시 계산해 일치 여부를 출력한다.
불일치가 있으면 FAIL로 표시 — 초안을 고치거나 근거를 다시 확인할 것.

사용: python scripts/verify_draft_numbers.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TOL = 0.0006  # 소수 넷째 자리 반올림 허용


def check(label, claimed, actual, tol=TOL):
    ok = actual is not None and abs(claimed - actual) <= tol
    mark = "OK  " if ok else "FAIL"
    got = "n/a" if actual is None else f"{actual:.4f}"
    print(f"[{mark}] {label:<52} draft={claimed:<9.4f} results={got}")
    return ok


def acc(run, col, dataset="lt", eval_filter="nostock"):
    d = pd.read_csv(R / "summary_metrics.csv")
    d = d[(d.eval_filter == eval_filter) & (d.dataset == dataset) & (d.run == run)]
    return None if d.empty else float(d[col].mean())


def ft(cfg, col, dataset="lt"):
    d = pd.read_csv(R / "summary_metrics.csv")
    d = d[(d.eval_filter == "nostock") & (d.dataset == dataset) & (d.ver == "v2")]
    d = d[d.run.str.split("/").str[-1].str.replace(r"_s\d+$", "", regex=True) == cfg]
    return None if d.empty else float(d[col].mean())


def inv(run, scenario, dataset="lt", clean=False):
    f = R / f"inventory_{dataset}{'_clean' if clean else ''}_sl2_L0.csv"
    d = pd.read_csv(f)
    d = d[(d.is_fixed_cr) & (d.run == run) & (d.scenario == scenario)]
    return None if d.empty else float(d.cost_per_unit_demand.iloc[0])


def mcb(run, col="mean_rank"):
    d = pd.read_csv(R / "mcb_lt.csv", index_col=0)
    return None if run not in d.index else float(d.loc[run, col])


print("=== 5.1 정확도 ===")
ok = []
ok.append(check("LoRA r16 lr1e-3 WAPE", 0.2444, ft("lora_r16_lr1e-3", "wape")))
ok.append(check("full-FT WAPE", 0.2445, ft("full", "wape")))
ok.append(check("LoRA r16 + mask WAPE", 0.2446, ft("lora_r16_mask", "wape")))
ok.append(check("LightGBM WAPE", 0.2482, acc("lightgbm_lt_raw", "wape")))
ok.append(check("Croston+EQ WAPE", 0.2520, acc("crostoneq_lt_raw", "wape")))
ok.append(check("Chronos-2 ZS WAPE", 0.2534, acc("chronos2_lt_raw", "wape")))
ok.append(check("ETS+EQ WAPE", 0.2538, acc("etseq_lt_raw", "wape")))
ok.append(check("LightGBM MASE", 0.841, acc("lightgbm_lt_raw", "mase_mean"), 0.001))
ok.append(check("full-FT MASE", 0.850, ft("full", "mase_mean"), 0.001))
ok.append(check("MCB full-FT mean rank", 12.16, mcb("ft_full_s0"), 0.006))
ok.append(check("MCB LightGBM mean rank", 12.54, mcb("lightgbm_lt_raw"), 0.006))
ok.append(check("MCB Chronos-2 mean rank", 13.19, mcb("chronos2_lt_raw"), 0.006))

print("\n=== 5.2-5.3 편향과 검열 처리 ===")
ok.append(check("Chronos-2 ZS WPE (nostock)", -0.0617, acc("chronos2_lt_raw", "wpe")))
ok.append(check("Chronos-2 ZS WPE (all days)", -0.0121,
                acc("chronos2_lt_raw", "wpe", eval_filter="all")))
ok.append(check("Chronos-2 mask WPE", -0.0192, acc("chronos2_lt_mask", "wpe")))
ok.append(check("TiRex ZS WPE", -0.0996, acc("tirex_lt_raw", "wpe")))
ok.append(check("LoRA+mask WPE", -0.0233, ft("lora_r16_mask", "wpe")))
ok.append(check("LoRA+mask WQL", 0.1688, ft("lora_r16_mask", "wql")))
ok.append(check("50K Chronos-2 raw WAPE", 0.3183, acc("chronos2_50k_raw", "wape", "frn50k")))
ok.append(check("50K Chronos-2 rec WAPE", 0.3042, acc("chronos2_50krec_raw", "wape", "frn50krec")))
ok.append(check("50K TFT rec WAPE", 0.2842, acc("tft_50krec_raw", "wape", "frn50k")))
ok.append(check("50K DLinear rec WAPE", 0.3055, acc("dlinear_50krec_raw", "wape", "frn50k")))

print("\n=== 5.4 재고 (고정 CR) ===")
LO, HI = "low_margin_waste_dominant", "high_margin_stockout_dominant"
ok.append(check("LightGBM 폐기우세 비용", 0.0536, inv("lightgbm_lt_raw", LO)))
ok.append(check("LoRA lr1e-3 폐기우세 비용", 0.0560, inv("ft_lora_r16_lr1e-3_s0", LO)))
ok.append(check("ETS+EQ 폐기우세 비용", 0.0562, inv("etseq_lt_raw", LO)))
ok.append(check("LoRA lr1e-3 품절우세 비용", 0.1094, inv("ft_lora_r16_lr1e-3_s0", HI)))
ok.append(check("LightGBM 품절우세 비용", 0.1197, inv("lightgbm_lt_raw", HI)))

print("\n=== 5.5 편향-서비스수준 교차 ===")
cx = pd.read_csv(R / "bias_crossover_lt.csv").set_index("cr")
ok.append(check("tau(|WPE|) at CR 0.2", 0.533, float(cx.loc[0.2, "tau_abs_wpe"]), 0.001))
ok.append(check("tau(|WPE|) at CR 0.9", -0.400, float(cx.loc[0.9, "tau_abs_wpe"]), 0.001))
ok.append(check("tau(WQL) at CR 0.2", 0.057, float(cx.loc[0.2, "tau_wql"]), 0.001))
ok.append(check("tau(WQL) at CR 0.9", 0.476, float(cx.loc[0.9, "tau_wql"]), 0.001))

print("\n=== 5.7 강건성 ===")
ro = pd.read_csv(R / "rolling_origin_lt.csv")
ro = ro[ro.eval_filter == "nostock"].set_index("run")
ok.append(check("origin o1 (06-29) WAPE", 0.2722, float(ro.loc["chronos2_lt_o1_raw", "wape"])))
ok.append(check("origin o2 (07-06) WAPE", 0.2598, float(ro.loc["chronos2_lt_o2_raw", "wape"])))
ok.append(check("50K LightGBM 폐기우세", 0.0731, inv("lightgbm_frn50k_raw", LO, "frn50k")))
ok.append(check("50K LightGBM 품절우세", 0.1850, inv("lightgbm_frn50k_raw", HI, "frn50k")))
ok.append(check("50K LoRA 품절우세", 0.1785, inv("ft_c2_50k_lora_r16_s0", HI, "frn50k")))
ok.append(check("청정표본 Chronos-2+mask 품절우세", 0.0774,
                inv("chronos2_lt_mask", HI, clean=True)))

mat = pd.read_parquet(R / "per_series_mase_lt.parquet")
r = mat.rank(axis=1, method="min")
ties = float(r.apply(lambda x: x.duplicated().any(), axis=1).mean())
cells = float(np.mean([x.duplicated(keep=False).sum() / len(x) for _, x in r.iterrows()]))
ok.append(check("동점 시계열 비율", 0.758, ties, 0.001))
ok.append(check("동점 셀 비율", 0.100, cells, 0.001))
ok.append(check("모델 수", 28, float(mat.shape[1]), 0))


print("=== TFT 시드 통계 (통일 집계) ===")
tft_c = ["tft_50k_s2025_raw", "tft_50k_s1_raw", "tft_50k_s2_raw"]
tft_r = ["tft_50krec_raw", "tft_50krec_s1_raw", "tft_50krec_s2_raw"]
sm = pd.read_csv(R / "summary_metrics.csv")
sm = sm[(sm.eval_filter == "nostock")]
def seed_stat(runs, col):
    v = sm[sm.run.isin(runs)][col]
    return float(v.mean()), float(v.std(ddof=1))
mc, sc_ = seed_stat(tft_c, "wape"); ok.append(check("TFT censored WAPE mean", 0.3170, mc, 0.001))
ok.append(check("TFT censored WAPE sd", 0.0083, sc_, 0.0006))
mr, sr = seed_stat(tft_r, "wape"); ok.append(check("TFT recovered WAPE mean", 0.2932, mr, 0.001))
mpc, spc = seed_stat(tft_c, "wpe"); ok.append(check("TFT censored WPE mean", -0.1002, mpc, 0.001))
mpr, spr = seed_stat(tft_r, "wpe"); ok.append(check("TFT recovered WPE mean", 0.0442, mpr, 0.001))
ok.append(check("TFT recovered WPE sd", 0.0449, spr, 0.001))

n_fail = sum(1 for x in ok if not x)
print(f"\n{'='*70}\n총 {len(ok)}건 중 불일치 {n_fail}건")
