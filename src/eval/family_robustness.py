"""P1 Task 4: RQ1 상관이 모델 패밀리 구성에 좌우되는지 검증.

28개 실행은 독립 기법의 균등 표본이 아니다 — Chronos-2 계열만 9개다. 두 가지로
견고성을 본다:
  (a) 패밀리당 대표 1개 (test 비용을 쓰지 않는 사전 규칙으로 선정)
  (b) leave-one-family-out — 패밀리를 하나씩 빼고 tau 곡선 재계산

대표 선정 규칙 (사전지정, test-set 미사용):
  각 패밀리의 '기본 공개 구성(default published configuration)' — 입력은 raw,
  적응 패밀리는 표준 적응인 full fine-tuning, 점예측 전용 고전 기법은 분위수가
  필요한 분석이므로 경험적 분위수 버전(그 패밀리의 원형 기법 기준).

산출:
  results/model_family_map.csv
  results/rq1_family_representative.csv
  results/rq1_leave_one_family_out.csv

사용: python -m src.eval.family_robustness
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
HOLD = 0.02
MIN_N = 6   # 대표 분석은 n이 작으므로 최소 기법 수를 낮춘다 (질적 방향만 본다)

FAMILY = {
    "chronos2_lt_raw": "chronos2_zeroshot", "chronos2_lt_mask": "chronos2_zeroshot",
    "ft_full_s0": "chronos2_adapted", "ft_lora_r8_s0": "chronos2_adapted",
    "ft_lora_r16_s0": "chronos2_adapted", "ft_lora_r32_s0": "chronos2_adapted",
    "ft_lora_r16_lr1e-3_s0": "chronos2_adapted",
    "ft_lora_r16_lr1e-5_s0": "chronos2_adapted",
    "ft_lora_r16_mask_s0": "chronos2_adapted",
    "moirai2_lt_raw": "moirai2", "moirai2_lt_mask": "moirai2",
    "timesfm_lt_raw": "timesfm", "timesfm_lt_mask": "timesfm",
    "tirex_lt_raw": "tirex", "tirex_lt_mask": "tirex",
    "ttm_lt_raw": "ttm", "ttm_lt_mask": "ttm",
    "ttmeq_lt_raw": "ttm",   # JORS R1 E3: TTM + empirical quantiles
    "lightgbm_lt_raw": "lightgbm",
    "croston_lt_raw": "intermittent_classical",
    "crostoneq_lt_raw": "intermittent_classical",
    "sba_lt_raw": "intermittent_classical", "sbaeq_lt_raw": "intermittent_classical",
    "tsb_lt_raw": "intermittent_classical", "tsbeq_lt_raw": "intermittent_classical",
    "ets_lt_raw": "smoothing_classical", "etseq_lt_raw": "smoothing_classical",
    "snaive_lt_raw": "smoothing_classical", "snaiveeq_lt_raw": "smoothing_classical",
}
REPRESENTATIVE = {
    "chronos2_zeroshot": "chronos2_lt_raw",
    "chronos2_adapted": "ft_full_s0",          # 표준 적응 = full fine-tuning
    "moirai2": "moirai2_lt_raw",
    "timesfm": "timesfm_lt_raw",
    "tirex": "tirex_lt_raw",
    "ttm": "ttmeq_lt_raw",                      # R1: TTM+EQ (TTM 점예측은 q>0.5 발주 불가, 고전 패밀리와 같은 규칙)
    "lightgbm": "lightgbm_lt_raw",
    "intermittent_classical": "crostoneq_lt_raw",  # 원형(Croston)+EQ
    "smoothing_classical": "etseq_lt_raw",
}


def load_metrics_and_costs():
    """실행별 예측 속성(집계)과 (q,b)별 비용을 만든다."""
    diag = pd.read_parquet(RESULTS / "per_series_diag_lt.parquet")
    agg = diag.groupby("run").agg(y=("y_sum", "sum"),
                                  ae=("abs_err_sum", "sum"),
                                  e=("err_sum", "sum"),
                                  pin=("pinball_sum", "sum"),
                                  nd=("n_days", "sum"))
    props = pd.DataFrame({
        "wape": agg.ae / agg.y, "abs_wpe": (agg.e / agg.y).abs(),
        "wql": agg.pin / agg.y})
    cov_cols = [c for c in diag.columns if c.startswith("hits_")]
    hits = diag.groupby("run")[cov_cols].sum()
    for c in cov_cols:
        lvl = float(c.split("_")[1])
        props[f"cov_{lvl}"] = hits[c] / agg.nd

    out = pd.read_parquet(RESULTS / "cost_q_outcomes_lt_sl2_L0.parquet")
    phys = out.groupby(["run", "q"])[["waste", "lost", "hold", "demand"]].sum()
    return props, phys


def tau_curves(runs, props, phys, bs=(0.3, 1.0, 3.0)):
    rows = []
    for (run, q), r in phys.iterrows():
        pass
    qs = sorted(phys.index.get_level_values("q").unique())
    for b in bs:
        for q in qs:
            sub = phys.xs(q, level="q")
            sub = sub[sub.index.isin(runs)]
            cost = (sub.waste + b * sub.lost + HOLD * sub.hold) / sub.demand
            pr = props.loc[cost.index]
            rec = {"b": b, "q": q, "n_methods": len(cost)}
            for prop in ["abs_wpe", "wql", "wape"]:
                v = pr[prop]
                if len(v) >= MIN_N and np.ptp(v) > 0:
                    rec[f"tau_{prop}"] = kendalltau(v, cost)[0]
                else:
                    rec[f"tau_{prop}"] = np.nan
            ck = f"cov_{q}"
            if ck in pr.columns and pr[ck].notna().sum() >= MIN_N:
                v = (pr[ck] - q).abs()
                ok = v.notna()
                rec["tau_coverage"] = (kendalltau(v[ok], cost[ok])[0]
                                       if ok.sum() >= MIN_N and np.ptp(v[ok]) > 0
                                       else np.nan)
            else:
                rec["tau_coverage"] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    props, phys = load_metrics_and_costs()
    all_runs = sorted(set(props.index) & set(phys.index.get_level_values("run")))
    fam = pd.DataFrame({"run": all_runs,
                        "family": [FAMILY[r] for r in all_runs]})
    fam["is_representative"] = fam.run.isin(REPRESENTATIVE.values())
    fam.to_csv(RESULTS / "model_family_map.csv", index=False)
    print(f"runs {len(all_runs)} | families {fam.family.nunique()}")

    # (a) 패밀리당 대표 1개
    reps = [r for r in REPRESENTATIVE.values() if r in all_runs]
    rep = tau_curves(reps, props, phys)
    rep.insert(0, "analysis", "one_representative_per_family")
    rep.to_csv(RESULTS / "rq1_family_representative.csv", index=False)
    print("\n=== 대표 1개/패밀리 (b=1.0) ===")
    s = rep[np.isclose(rep.b, 1.0)]
    print(s[["q", "n_methods", "tau_abs_wpe", "tau_wql", "tau_coverage",
             "tau_wape"]].round(3).to_string(index=False))

    # (b) leave-one-family-out
    lofo = []
    for f in sorted(fam.family.unique()):
        keep = [r for r in all_runs if FAMILY[r] != f]
        t = tau_curves(keep, props, phys)
        t.insert(0, "excluded_family", f)
        lofo.append(t)
    lofo = pd.concat(lofo, ignore_index=True)
    lofo = lofo.rename(columns={"tau_abs_wpe": "tau_abs_bias"})
    lofo.to_csv(RESULTS / "rq1_leave_one_family_out.csv", index=False)

    # 질적 교차가 유지되는가: b별·제외 패밀리별로
    #   tau_bias(q<=0.3 평균) > 0 > tau_bias(q>=0.9 평균),
    #   tau_wql(q>=0.9) - tau_wql(q<=0.3) > 0
    print("\n=== leave-one-family-out: 교차 유지 여부 (b=1.0) ===")
    ok_all = True
    for f in sorted(fam.family.unique()):
        s = lofo[(lofo.excluded_family == f) & np.isclose(lofo.b, 1.0)]
        lo_b = s[s.q <= 0.3].tau_abs_bias.mean()
        hi_b = s[s.q >= 0.9].tau_abs_bias.mean()
        lo_w = s[s.q <= 0.3].tau_wql.mean()
        hi_w = s[s.q >= 0.9].tau_wql.mean()
        keep = (lo_b > 0) and (hi_b < lo_b) and (hi_w > lo_w)
        sign_flip = hi_b < 0
        ok_all &= keep
        print(f"  -{f:24s} bias {lo_b:+.2f}->{hi_b:+.2f} "
              f"(sign flip {'Y' if sign_flip else 'N'}) | "
              f"wql {lo_w:+.2f}->{hi_w:+.2f} | crossover "
              f"{'OK' if keep else 'WEAK'}")
    print("\nqualitative crossover preserved in all LOFO samples:",
          "YES" if ok_all else "NO")


if __name__ == "__main__":
    main()
