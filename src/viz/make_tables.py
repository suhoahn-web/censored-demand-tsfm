"""논문 표 자동 생성 (results/tables/): LaTeX(booktabs) + 검토용 Markdown.

본문 번호: Table 1~3은 수동 작성(갭 표·데이터셋·모델), 아래는 자동 생성분.
Table 4  주 정확도 결과 (모델 × 지표), 데이터셋별
Table 5  적응 전략 비용-성능 (시드 3회 평균±표준편차, 학습시간, 학습 파라미터)
Table 6  재고 성과 (비용 시나리오 × 모델, 고정 CR)
Table 7  검열 처리 팩토리얼 (검열/복원 × 모델)

수치는 results/*.csv에서 읽으므로 실험이 갱신되면 재실행만 하면 된다.
사용: python -m src.viz.make_tables
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.style import family_of, is_ablation, short_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
TABDIR = RESULTS / "tables"

FAMILY_ORDER = ["Statistical", "Tree/ML", "Local DL", "TSFM (zero-shot)", "TSFM (fine-tuned)"]


def write(name: str, df: pd.DataFrame, caption: str, note: str = "") -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    (TABDIR / f"{name}.md").write_text(
        f"**{caption}**\n\n{df.to_markdown(index=False)}\n\n{note}\n", encoding="utf-8")
    latex = df.to_latex(index=False, escape=False, column_format="l" + "r" * (df.shape[1] - 1))
    latex = latex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")
    body = ("\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{name}}}\n{latex}")
    if note:
        body += f"\\begin{{tablenotes}}\\footnotesize\n\\item {note}\n\\end{{tablenotes}}\n"
    body += "\\end{table}\n"
    (TABDIR / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"  {name}: {df.shape[0]} rows")


def _acc(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[~df.run.str.contains("equ_")]  # R1: 비검열-잔차 EQ 변형은 부록 B.8 전용
    df = df[(df.eval_filter == "nostock") & (df.dataset == dataset)]
    zs = df[df.adapt == "zero-shot"].copy()
    zs = zs[[not is_ablation(r) for r in zs["run"]]]
    ft = df[(df.adapt != "zero-shot") & (df["ver"] == "v2")].copy()
    if len(ft):
        # 시드 접미사(_s0/_s1/_s2)를 떼고 설정 단위로 집계 — 시드별 행 분리 방지
        ft["run"] = ("ft_" + ft["run"].str.split("/").str[-1]
                     .str.replace(r"_s\d+$", "", regex=True))
        g = (ft.groupby("run").agg(wape=("wape", "mean"), wape_sd=("wape", "std"),
                                   wpe=("wpe", "mean"), mase_mean=("mase_mean", "mean"),
                                   wql=("wql", "mean"), crps=("crps", "mean"),
                                   n_seeds=("seed", "nunique"))
             .reset_index())
        zs = pd.concat([zs, g], ignore_index=True)
    zs["Model"] = zs["run"].map(short_label)
    zs["Family"] = zs["run"].map(family_of)
    return zs


def table2_accuracy() -> None:
    for ds, tag in [("lt", "LT"), ("frn50k", "50K")]:
        d = _acc(ds)
        if d.empty:
            continue
        # 본 표는 대표 설정만 — lr/rank 변형은 Table 3(적응 전략)에 상세
        drop = ["lr 1e-5", "LoRA r8", "LoRA r32"]
        d = d[~d["Model"].str.contains("|".join(drop), regex=True)]
        d = d.sort_values("wape")
        out = pd.DataFrame({
            "Model": d["Model"], "Family": d["Family"],
            "WAPE": d["wape"].map(lambda v: f"{100*v:.2f}"),
            "MASE": d["mase_mean"].map(lambda v: f"{v:.3f}"),
            "WQL": d["wql"].map(lambda v: f"{v:.4f}"),
            "CRPS": d["crps"].map(lambda v: f"{v:.4f}"),
            "WPE": d["wpe"].map(lambda v: f"{100*v:+.2f}"),
        })
        write(f"table4_accuracy_{ds}",
              out,
              f"Forecast accuracy on FreshRetailNet-{tag} "
              f"(official 7-day evaluation, days without stockouts only). "
              f"WAPE and WPE in \\%; WPE $<0$ indicates under-forecasting.",
              "EQ = empirical residual quantiles added to a point forecaster; "
              "mask = stockout hours masked in the input context.")


def table3_adaptation() -> None:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[~df.run.str.contains("equ_")]  # R1: 비검열-잔차 EQ 변형은 부록 B.8 전용
    d = df[(df.eval_filter == "nostock") & (df.ver == "v2") & (df.dataset == "lt")].copy()
    if d.empty:
        return
    d["cfg"] = d["run"].str.split("/").str[-1].str.replace(r"_s\d$", "", regex=True)
    g = (d.groupby("cfg").agg(n_seeds=("seed", "nunique"),
                              wape=("wape", "mean"), wape_sd=("wape", "std"),
                              wpe=("wpe", "mean"), wql=("wql", "mean"),
                              secs=("train_sec", "mean")).reset_index())
    zs = df[(df.eval_filter == "nostock") & (df.run == "chronos2_lt_raw")]
    rows = []
    if len(zs):
        r = zs.iloc[0]
        rows.append({"Configuration": "Zero-shot", "Seeds": "--",
                     "WAPE": f"{100*r['wape']:.2f}", "WPE": f"{100*r['wpe']:+.2f}",
                     "WQL": f"{r['wql']:.4f}", "GPU min": "0.0", "Trainable": "0"})
    params = {}
    for run_dir in sorted((RESULTS / "finetune_v2").iterdir()):
        mf = run_dir / "meta.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            params[run_dir.name.rsplit("_s", 1)[0]] = m.get("n_trainable_params")
    for _, r in g.iterrows():
        rows.append({"Configuration": short_label("ft_" + r["cfg"]),
                     "Seeds": int(r["n_seeds"]),
                     "WAPE": f"{100*r['wape']:.2f} ({100*r['wape_sd']:.2f})"
                             if r["n_seeds"] > 1 else f"{100*r['wape']:.2f}",
                     "WPE": f"{100*r['wpe']:+.2f}",
                     "WQL": f"{r['wql']:.4f}",
                     "GPU min": f"{r['secs']/60:.1f}",
                     "Trainable": f"{params.get(r['cfg'], 0):,}"})
    write("table5_adaptation", pd.DataFrame(rows),
          "Adaptation strategies for Chronos-2 on FreshRetailNet-LT. "
          "WAPE and WPE in \\%; standard deviation over seeds in parentheses.",
          "Trainable = number of updated parameters; the base model has 120M. "
          "GPU minutes measured on a single NVIDIA A100 80GB.")


def table4_inventory() -> None:
    f = RESULTS / "inventory_lt_sl2_L0.csv"
    if not f.exists():
        return
    inv = pd.read_csv(f)
    inv = inv[~inv.run.str.contains("equ_")]
    # 본문 §5.4·Fig.5와 동일: 시나리오별 고정 CR (테스트셋 튜닝 배제)
    best = inv[inv["is_fixed_cr"]]
    best = best[[not is_ablation(r) for r in best["run"]]]
    piv = best.pivot_table(index="run", columns="scenario",
                           values="cost_per_unit_demand")
    cols = ["low_margin_waste_dominant", "medium_margin", "high_margin_stockout_dominant"]
    piv = piv[[c for c in cols if c in piv.columns]]
    fill = best.pivot_table(index="run", columns="scenario", values="fill_rate")
    piv = piv.sort_values(piv.columns[-1])
    out = pd.DataFrame({"Model": [short_label(i) for i in piv.index],
                        "Family": [family_of(i) for i in piv.index]})
    names = {"low_margin_waste_dominant": "$b=0.3$", "medium_margin": "$b=1.0$",
             "high_margin_stockout_dominant": "$b=3.0$"}
    for c in piv.columns:
        out[names[c]] = piv[c].map(lambda v: f"{v:.4f}").values
    out["Fill rate ($b=3$)"] = fill[cols[-1]].reindex(piv.index).map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "--").values
    write("table6_inventory", out,
          "Inventory cost per unit of demand under three cost scenarios "
          "(FreshRetailNet-LT, shelf life 2 days, lead time 0). All methods order "
          "at the same quantile within a scenario (0.5, 0.7 and 0.9 respectively).",
          "Unit purchase cost normalised to 1; $b$ is the unit shortage cost "
          "(lost margin). Lower is better. Fixing the ordering quantile avoids "
          "granting any method test-set freedom to tune its service level.")


def table5_censoring() -> None:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[~df.run.str.contains("equ_")]  # R1: 비검열-잔차 EQ 변형은 부록 B.8 전용
    d = df[df.eval_filter == "nostock"]
    def stat(runs):
        """단일 실행이면 값 그대로, 시드 복수면 평균(표준편차)."""
        s_ = d[d.run.isin(runs)]
        if s_.empty:
            return None, None
        w, p_ = s_["wape"], s_["wpe"]
        if len(s_) > 1:
            return (f"{100*w.mean():.2f} ({100*w.std(ddof=1):.2f})",
                    f"{100*p_.mean():+.2f} ({100*p_.std(ddof=1):.2f})")
        return f"{100*w.iloc[0]:.2f}", f"{100*p_.iloc[0]:+.2f}"

    pairs = [
        (["chronos2_50k_raw"], ["chronos2_50krec_raw"], "Chronos-2 (zero-shot)"),
        (["moirai2_50k_raw"], ["moirai2_50krec_raw"], "Moirai-2 (zero-shot)"),
        (["tft_50k_s2025_raw", "tft_50k_s1_raw", "tft_50k_s2_raw"],
         ["tft_50krec_raw", "tft_50krec_s1_raw", "tft_50krec_s2_raw"],
         "TFT (local, 3 seeds)"),
        (["dlinear_50k_raw"], ["dlinear_50krec_raw"], "DLinear (local)"),
    ]
    rows = []
    for cen, rec, label in pairs:
        wc, pc = stat(cen)
        wr, pr = stat(rec)
        if wc is None or wr is None:
            continue
        rows.append({"Model": label,
                     "WAPE (censored)": wc, "WAPE (recovered)": wr,
                     "WPE (censored)": pc, "WPE (recovered)": pr})
    if rows:
        write("table7_censoring", pd.DataFrame(rows),
              "Effect of latent demand recovery on accuracy and bias "
              "(FreshRetailNet-50K). Models are trained on censored sales versus "
              "recovered demand and evaluated against observed sales in both cases.",
              "WAPE and WPE in \\%. Recovery removes most of the downward bias "
              "for every model family.")


if __name__ == "__main__":
    print("generating tables ->", TABDIR)
    for fn in [table2_accuracy, table3_adaptation, table4_inventory, table5_censoring]:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
