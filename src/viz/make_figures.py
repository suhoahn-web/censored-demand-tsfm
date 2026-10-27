"""논문 그림 생성 (results/figures/).

본문 등장 순서에 맞춘 번호:
Fig.1 개념도 (framework_fig.py)
Fig.2 MCB 평균 순위 구간도
Fig.3 검열 처리 팩토리얼 — 편향(WPE)
Fig.4 수요 패턴 사분면별 평균 순위 (마스킹 효과)
Fig.5 정확도 vs 재고비용 산점도 (시나리오 3패싯)
Fig.6 서비스수준-재고 트레이드오프 곡선
Fig.7 편향·분포품질과 재고비용의 순위상관 교차

사용: python -m src.viz.make_figures --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from src.viz.style import (DOUBLE, FAMILY_COLORS, INK_2, ONE_HALF, REFLINE,
                           SINGLE, family_of, is_ablation, panel_label, savefig,
                           short_label, use_paper_style)

# 본 그림에 넣을 대표 실행 — 중복 설정(랭크 변형 등)은 하나만, ablation 제외
MAIN_RUNS_LT = [
    "chronos2_lt_raw", "tirex_lt_raw", "moirai2_lt_raw", "timesfm_lt_raw", "ttm_lt_raw",
    "chronos2_lt_mask",
    "ft_lora_r16_s0", "ft_lora_r16_lr1e-3_s0", "ft_full_s0", "ft_lora_r16_mask_s0",
    "lightgbm_lt_raw",
    "etseq_lt_raw", "crostoneq_lt_raw", "snaiveeq_lt_raw",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
FIGDIR = RESULTS / "figures"
SCEN = [("low_margin_waste_dominant", "b = 0.3 (waste-dominant)"),
        ("medium_margin", "b = 1.0"),
        ("high_margin_stockout_dominant", "b = 3.0 (stockout-dominant)")]


def load_accuracy(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[~df.run.str.contains("equ_")]
    df = df[(df.eval_filter == "nostock") & (df.dataset == dataset)]
    zs = df[df.adapt == "zero-shot"][["run", "wape", "wpe", "wql"]]
    ft = df[(df.adapt != "zero-shot") & (df["seed"] == 0)].copy()
    if len(ft):
        ft["run"] = "ft_" + ft["run"].str.split("/").str[-1]
        zs = pd.concat([zs, ft[["run", "wape", "wpe", "wql"]]], ignore_index=True)
    return zs.drop_duplicates("run")


def fig1_accuracy_vs_cost(dataset: str) -> None:
    inv = pd.read_csv(RESULTS / f"inventory_{dataset}_sl2_L0.csv")
    inv = inv[~inv.run.str.contains("equ_")]
    # 본문 §5.4와 동일하게 시나리오별 고정 CR에서 비교 (테스트셋 튜닝 없음)
    best = inv[inv["is_fixed_cr"]]
    acc = load_accuracy(dataset)
    d = best.merge(acc, on="run")
    d = d[d["run"].isin(MAIN_RUNS_LT)]
    d["family"] = d["run"].map(family_of)
    d["label"] = d["run"].map(short_label)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.9))
    for ax, (sc, title) in zip(axes, SCEN):
        g = d[d.scenario == sc]
        for fam, c in FAMILY_COLORS.items():
            s = g[g.family == fam]
            if len(s):
                ax.scatter(s["wape"], s["cost_per_unit_demand"], s=14, c=c,
                           edgecolors="white", linewidths=0.4, label=fam, zorder=3)
        if len(g) > 2:  # 회귀선 (EJOR Fig.5 관행: 파선)
            b, a = np.polyfit(g["wape"], g["cost_per_unit_demand"], 1)
            xs = np.linspace(g["wape"].min(), g["wape"].max(), 20)
            ax.plot(xs, a + b * xs, "--", color=REFLINE, lw=0.8, zorder=2)
            tau, p = kendalltau(g["wape"], g["cost_per_unit_demand"])
            title = title + f"   $\\tau$ = {tau:.2f}" + ("*" if p < 0.05 else "")
        texts = [ax.annotate(r["label"], (r["wape"], r["cost_per_unit_demand"]),
                             fontsize=5.4, color=INK_2)
                 for _, r in g.iterrows()]
        try:  # 라벨 충돌 자동 회피
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, expand=(1.15, 1.4),
                        arrowprops=dict(arrowstyle="-", color=REFLINE, lw=0.35))
        except ImportError:
            pass
        ax.margins(y=0.16)
        panel_label(ax, title)
        ax.set_xlabel("WAPE")
    axes[0].set_ylabel("Inventory cost per unit demand")
    present = [f for f in FAMILY_COLORS if (d["family"] == f).any()]
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=FAMILY_COLORS[f],
                          label=f) for f in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(present), fontsize=6.5,
               handletextpad=0.25, columnspacing=1.4, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(pad=0.4, rect=(0, 0.055, 1, 1))
    savefig(fig, FIGDIR / f"fig5_accuracy_vs_cost_{dataset}.png")


def fig2_mcb(dataset: str) -> None:
    mcb = pd.read_csv(RESULTS / f"mcb_{dataset}.csv", index_col=0)
    mcb = mcb[[not is_ablation(i) for i in mcb.index]]
    mcb = mcb.sort_values("mean_rank", ascending=False)
    labels = [short_label(i) for i in mcb.index]
    colors = [FAMILY_COLORS[family_of(i)] for i in mcb.index]
    y = np.arange(len(mcb))

    fig, ax = plt.subplots(figsize=(ONE_HALF, 0.155 * len(mcb) + 0.85))
    best_upper = mcb["upper"].min()
    ax.axvline(best_upper, color=REFLINE, ls="--", lw=0.8, zorder=1)
    for yi, (_, r), c in zip(y, mcb.iterrows(), colors):
        ax.plot([r["lower"], r["upper"]], [yi, yi], color=c, lw=1.8,
                solid_capstyle="round", zorder=3)
        ax.plot(r["mean_rank"], yi, "o", color=c, ms=3.0, mec="white", mew=0.5, zorder=4)
    ax.set_yticks(y, labels, fontsize=6.2)
    ax.set_ylim(-0.9, len(mcb) - 0.1)
    ax.set_xlabel("Mean rank (MASE, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.annotate("upper bound of the best model", xy=(best_upper, len(mcb) - 1.05),
                xytext=(6, 0), textcoords="offset points", fontsize=6,
                color=INK_2, va="center")
    present = [f for f in FAMILY_COLORS if any(family_of(i) == f for i in mcb.index)]
    handles = [plt.Line2D([], [], color=FAMILY_COLORS[f], lw=1.8, label=f) for f in present]
    ax.legend(handles=handles, loc="lower right", fontsize=6, handletextpad=0.4,
              labelspacing=0.22, borderpad=0.2)
    fig.tight_layout(pad=0.4)
    savefig(fig, FIGDIR / f"fig2_mcb_{dataset}.png")


def fig3_censoring_factorial() -> None:
    df = pd.read_csv(RESULTS / "summary_metrics.csv")
    df = df[~df.run.str.contains("equ_")]
    df = df[(df.eval_filter == "nostock")]
    rows = []
    pairs = [("chronos2_50k_raw", "chronos2_50krec_raw", "Chronos-2 (ZS)"),
             ("moirai2_50k_raw", "moirai2_50krec_raw", "Moirai-2 (ZS)")]
    for cen, rec, label in pairs:
        for run, cond in [(cen, "Censored sales"), (rec, "Recovered demand")]:
            r = df[df.run == run]
            if len(r):
                rows.append({"model": label, "cond": cond,
                             "wpe": float(r["wpe"].iloc[0]), "wape": float(r["wape"].iloc[0])})
    ft = df[(df.ver == "v2") & (df["seed"] == 0)]
    for run, cond in [("v2/c2_50k_lora_r16_s0", "Censored sales"),
                      ("v2/c2_rec_lora_r16_s0", "Recovered demand")]:
        r = ft[ft.run == run]
        if len(r):
            rows.append({"model": "Chronos-2 (LoRA)", "cond": cond,
                         "wpe": float(r["wpe"].iloc[0]), "wape": float(r["wape"].iloc[0])})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    models = list(dict.fromkeys(d["model"]))
    x = np.arange(len(models))
    w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(ONE_HALF, 2.3))
    for ax, metric, ylab in [(axes[0], "wpe", "WPE (bias)"), (axes[1], "wape", "WAPE")]:
        # 학술지 관행: 빗금 대신 무채색/유채색 단색 쌍 (before/after)
        for k, (cond, col) in enumerate([("Censored sales", "#b5b5b5"),
                                         ("Recovered demand", "#2a78d6")]):
            vals = [d[(d.model == m) & (d.cond == cond)][metric].mean() for m in models]
            ax.bar(x + (k - 0.5) * w, vals, w * 0.92, label=cond, color=col,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xticks(x, [m.replace(" (", "\n(") for m in models], fontsize=6.2)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        if metric == "wpe":
            ax.axhline(0, color=INK_2, lw=0.6)
    axes[0].legend(loc="lower right", fontsize=6, handletextpad=0.4)
    fig.tight_layout(pad=0.4)
    savefig(fig, FIGDIR / "fig3_censoring_factorial.png")


def fig4_tradeoff(dataset: str) -> None:
    inv = pd.read_csv(RESULTS / f"inventory_{dataset}_sl2_L0.csv")
    inv = inv[~inv.run.str.contains("equ_")]
    inv = inv[inv.scenario == "medium_margin"]
    show = [r for r in inv["run"].unique()
            if any(k in r for k in ["chronos2_lt_raw", "lora_r16_lr1e-3", "lightgbm",
                                    "etseq", "ets_", "tirex_lt_raw", "crostoneq"])]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for run in show:
        g = inv[inv.run == run].sort_values("cr")
        if len(g) < 2:
            ax.plot(g["avg_inventory"], g["fill_rate"], "o",
                    color=FAMILY_COLORS[family_of(run)], ms=4, mec="white", mew=0.5, zorder=4)
        else:
            ax.plot(g["avg_inventory"], g["fill_rate"], "-o",
                    color=FAMILY_COLORS[family_of(run)], lw=1.0, ms=2.6, zorder=3)
        last = g.iloc[-1]
        ax.annotate(short_label(run), (last["avg_inventory"], last["fill_rate"]),
                    textcoords="offset points", xytext=(3, -1), fontsize=5.8, color=INK_2)
    ax.set_xlabel("Average inventory (units/series-day)")
    ax.set_ylabel("Achieved fill rate")
    fig.tight_layout(pad=0.4)
    savefig(fig, FIGDIR / f"fig6_tradeoff_{dataset}.png")


def fig5_quadrant(dataset: str) -> None:
    mat = pd.read_parquet(RESULTS / f"per_series_mase_{dataset}.parquet")
    cls = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / f"{dataset}_classification.parquet")
    cls = cls.set_index(["store_id", "product_id"])
    ranks = mat.rank(axis=1).join(cls[["quadrant"]])
    order = ["smooth", "erratic", "intermittent", "lumpy"]
    rq = ranks.groupby("quadrant")[mat.columns].mean().T[order]
    show = [c for c in rq.index if any(k in c for k in
            ["chronos2_lt_raw", "chronos2_lt_mask", "tirex_lt_mask", "moirai2_lt_mask",
             "lightgbm", "ets_lt_raw", "ft_lora_r16_s0"])]
    # 학술지 관행(예: Theodorou et al. 2025 EJOR Fig.4): 방법별 고유 색 + 우측 범례
    method_palette = ["#1f6fb2", "#e07b39", "#2e8b57", "#c8443c",
                      "#7b5bb5", "#8c6d31", "#3aa6b9"]
    fig, ax = plt.subplots(figsize=(SINGLE + 1.1, 2.5))
    x = np.arange(len(order))
    for run, col in zip(show, method_palette):
        style = "--" if "mask" in run else "-"
        ax.plot(x, rq.loc[run], style, marker="o", ms=3, color=col, lw=1.1,
                label=short_label(run), zorder=3)
    ax.set_xticks(x, order, fontsize=6.5)
    ax.set_xlabel("Demand pattern (ADI $\\times$ CV$^2$ quadrant)")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.invert_yaxis()
    ax.set_xlim(-0.2, len(order) - 0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6,
              frameon=False, handlelength=1.6, labelspacing=0.45)
    fig.tight_layout(pad=0.4)
    savefig(fig, FIGDIR / f"fig4_quadrant_{dataset}.png")


def fig0_framework() -> None:
    from src.viz.framework_fig import draw
    draw()


def fig6_bias_crossover(dataset: str) -> None:
    """편향·분포품질과 재고비용의 순위상관이 발주 분위수에 따라 교차하는 양상."""
    d = pd.read_csv(RESULTS / f"bias_crossover_{dataset}.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    ax.axhline(0, color=INK_2, lw=0.6, zorder=1)
    series = [("tau_abs_wpe", "p_abs_wpe", "|WPE| (bias)", FAMILY_COLORS["Statistical"]),
              ("tau_wql", "p_wql", "WQL (distribution)", FAMILY_COLORS["TSFM (zero-shot)"])]
    for col, pcol, label, c in series:
        ax.plot(d["cr"], d[col], "-", color=c, lw=1.2, zorder=3)
        sig = d[d[pcol] < 0.05]
        ax.plot(d["cr"], d[col], "o", color=c, ms=3.2, mfc="white", mew=1.0, zorder=4)
        ax.plot(sig["cr"], sig[col], "o", color=c, ms=3.4, zorder=5)
        ax.annotate(label, (d["cr"].iloc[-1], d[col].iloc[-1]),
                    textcoords="offset points", xytext=(-2, 7 if col == "tau_wql" else -10),
                    fontsize=6, color=c, ha="right")
    ax.set_xlabel("Ordering quantile (critical ratio)")
    ax.set_ylabel(r"Kendall $\tau$ with inventory cost")
    ax.set_ylim(-0.62, 0.75)
    ax.text(0.02, 0.97, "filled markers: $p<0.05$", transform=ax.transAxes,
            fontsize=5.8, color=INK_2, va="top")
    fig.tight_layout(pad=0.4)
    savefig(fig, FIGDIR / f"fig7_bias_crossover_{dataset}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    args = ap.parse_args()
    use_paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fn in [fig0_framework,
               lambda: fig1_accuracy_vs_cost(args.dataset),
               lambda: fig2_mcb(args.dataset),
               fig3_censoring_factorial,
               lambda: fig4_tradeoff(args.dataset),
               lambda: fig5_quadrant(args.dataset),
               lambda: fig6_bias_crossover(args.dataset)]:
        try:
            fn()
            print("ok:", fn.__name__ if hasattr(fn, "__name__") else "fig")
        except Exception as e:
            print("FAIL:", type(e).__name__, e)
    print("figures ->", FIGDIR)


if __name__ == "__main__":
    main()
