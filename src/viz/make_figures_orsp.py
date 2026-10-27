"""OR Spectrum 개정판 신규 그림.

  fig_origins        다중 origin WAPE·순위 안정성            (Fig. 2)
  fig_regimes        발주분위수별 tau(예측속성, 비용)         (Fig. 3, 핵심)
  fig_calibration    명목 분위수 대비 경험적 커버리지          (Fig. 4)
  fig_costq          비용구조 x 발주분위수 최저비용 지형       (Fig. 5)
  fig_eqabl          EQ 분해: 점 / 재중심 / 분산 / 전체        (Fig. 6)

사용: python -m src.viz.make_figures_orsp
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.viz.style import (DOUBLE, GRID, INK, INK_2, ONE_HALF, REFLINE, SINGLE,
                           family_of, panel_label, savefig, short_label,
                           use_paper_style)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
FIGS = RESULTS / "figures"
# 본문 텍스트 폭(16.0cm = 6.30in)에 맞춰 그림을 그린다 —
# 축소되면 폰트가 스프링거 최소치(약 2mm)를 밑돈다.
PAGE = 6.30

# 8색 범주형 팔레트 — 고정 순서, 순환 금지.
# scripts/validate_palette.py 통과 (인접 CVD dE 6.9는 마커 형태로 2차 부호화,
# 초록의 표면 대비 경고는 모든 계열에 범례+직접 라벨을 두어 해소).
CAT8 = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7", "#00868b",
        "#8a6d3b", "#b05fb0"]
CAT_MK = ["o", "s", "^", "D", "v", "P", "X", "*"]

PROP_STYLE = {
    "abs_wpe": ("|WPE|  (centring error)", "#e34948", "o", "-"),
    "wql": ("WQL  (distributional quality)", "#2a78d6", "s", "-"),
    "abs_cov_err": ("|coverage error| at $q$", "#1baf7a", "^", "--"),
    "wape": ("WAPE  (point accuracy)", "#52514e", "d", ":"),
}


def fig_origins() -> None:
    d = pd.read_csv(RESULTS / "multi_origin_lt.csv")
    keep = ["chronos2_lt_raw", "moirai2_lt_raw", "tirex_lt_raw",
            "chronos2_lt_mask", "ets_lt_raw", "etseq_lt_raw",
            "crostoneq_lt_raw", "ttm_lt_raw"]
    sub = d[d.method.isin(keep)]
    p = sub.pivot(index="origin", columns="method", values="wape") * 100
    # 순위는 15개 기법 전체 안에서 계산한다 (표시만 8개)
    rank_all = d.pivot(index="origin", columns="method", values="wape").rank(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.9),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(p.index))
    for k, m in enumerate(keep):
        axes[0].plot(x, p[m], marker=CAT_MK[k], ms=3.0, lw=1.0, color=CAT8[k],
                     label=short_label(m), alpha=0.95)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([o[2:7] for o in p.index], rotation=45, ha="right")
    axes[0].set_xlabel("forecast origin")
    axes[0].set_ylabel("WAPE (%)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2,
                   fontsize=6.2, handlelength=1.8, columnspacing=1.0)
    axes[0].text(0.985, 0.02, "(a)", transform=axes[0].transAxes, ha="right",
                 va="bottom", fontsize=7, color=INK)

    # (b) 기법별 순위 분포 — 평균과 최소-최대 범위 (스파게티 대신)
    order = rank_all[keep].mean().sort_values().index.tolist()
    for i, m in enumerate(order):
        k = keep.index(m)
        r = rank_all[m]
        axes[1].plot([r.min(), r.max()], [i, i], lw=1.6, color=CAT8[k],
                     alpha=0.55, solid_capstyle="round")
        axes[1].plot(r, [i] * len(r), marker="|", ls="none", ms=5,
                     color=CAT8[k], alpha=0.8)
        axes[1].plot(r.mean(), i, marker=CAT_MK[k], ms=4.5, color=CAT8[k],
                     mec="white", mew=0.6, zorder=4)
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([short_label(m) for m in order], fontsize=6.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("rank within origin (1 = best of 15)")
    axes[1].set_xlim(0.4, 15.6)
    axes[1].text(0.985, 0.02, "(b)", transform=axes[1].transAxes, ha="right",
                 va="bottom", fontsize=7, color=INK)
    fig.tight_layout()
    savefig(fig, FIGS / "fig_origins.png")
    print("  fig_origins")


def fig_regimes() -> None:
    d = pd.read_csv(RESULTS / "property_regimes_lt.csv")
    bs = [0.3, 1.0, 3.0]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.7), sharey=True)
    for ax, b in zip(axes, bs):
        s = d[np.isclose(d.b, b)].sort_values("q")
        for prop, (lab, col, mk, ls) in PROP_STYLE.items():
            y = s[f"tau_{prop}"].to_numpy()
            lo = s[f"tau_{prop}_lo"].to_numpy()
            hi = s[f"tau_{prop}_hi"].to_numpy()
            ok = np.isfinite(y)
            ax.plot(s.q[ok], y[ok], marker=mk, ms=3.0, lw=1.2, ls=ls, color=col,
                    label=lab)
            ax.fill_between(s.q[ok], lo[ok], hi[ok], color=col, alpha=0.16,
                            linewidth=0)
        ax.axhline(0, color=REFLINE, lw=0.7, ls="--")
        ax.set_xlabel("ordering quantile $q$")
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8"])
        ax.set_xlim(0.15, 1.0)
        panel_label(ax, f"$b = {b}$")
        ax.set_ylim(-0.62, 0.92)
    axes[0].set_ylabel(r"Kendall $\tau$ with inventory cost")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
                   fontsize=6.5, handlelength=2.2)
    fig.tight_layout()
    savefig(fig, FIGS / "fig_regimes.png")
    print("  fig_regimes")


def fig_calibration() -> None:
    c = pd.read_csv(RESULTS / "quantile_calibration_lt.csv")
    sel = ["ft_lora_r16_lr1e-3_s0", "chronos2_lt_raw", "tirex_lt_raw",
           "moirai2_lt_raw", "lightgbm_lt_raw", "etseq_lt_raw",
           "crostoneq_lt_raw"]
    c = c[c.run.isin(sel) & c.nominal.between(0.09, 0.91)]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.6))
    for k, m in enumerate(sel):
        s = c[c.run == m].sort_values("nominal")
        axes[0].plot(s.nominal, s.coverage, marker=CAT_MK[k], ms=3.0, lw=1.0,
                     color=CAT8[k], alpha=0.95, label=short_label(m))
        axes[1].plot(s.nominal, s.coverage_error, marker=CAT_MK[k], ms=3.0,
                     lw=1.0, color=CAT8[k], alpha=0.95)
    axes[0].plot([0.1, 0.9], [0.1, 0.9], color=REFLINE, lw=0.7, ls="--",
                 label="perfect calibration")
    axes[1].axhline(0, color=REFLINE, lw=0.7, ls="--")
    axes[0].set_xlabel("nominal quantile")
    axes[0].set_ylabel("empirical coverage")
    axes[1].set_xlabel("nominal quantile")
    axes[1].set_ylabel("coverage error (empirical $-$ nominal)")
    axes[0].legend(loc="upper left", fontsize=6.2, handlelength=1.8)
    for ax, tag in [(axes[0], "(a) reliability"), (axes[1], "(b) coverage error")]:
        ax.text(0.985, 0.02, tag, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, color=INK)
    fig.tight_layout()
    savefig(fig, FIGS / "fig_calibration.png")
    print("  fig_calibration")


def fig_costq() -> None:
    g = pd.read_csv(RESULTS / "cost_q_grid_lt_sl2_L0.csv")
    best = g.loc[g.groupby(["b", "q"])["cost_per_unit_demand"].idxmin()]
    piv = best.pivot(index="b", columns="q", values="run")
    cost = (g.loc[g.groupby(["b", "q"])["cost_per_unit_demand"].idxmin()]
            .pivot(index="b", columns="q", values="cost_per_unit_demand"))
    names = sorted(set(piv.to_numpy().ravel()))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7",
               "#8a6d3b", "#00a0a0", "#b05fb0"]
    cmap = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(PAGE, 3.0))
    bs, qs = list(piv.index), list(piv.columns)
    for i, b in enumerate(bs):
        for j, q in enumerate(qs):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor=cmap[piv.loc[b, q]],
                                       edgecolor="white", linewidth=1.2,
                                       alpha=0.85))
            ax.text(j, i + 0.22, f"{cost.loc[b, q]:.3f}", ha="center",
                    va="center", fontsize=6.0, color="white")
    ax.set_xlim(-.5, len(qs) - .5)
    ax.set_ylim(-.5, len(bs) - .5)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"{q:g}" for q in qs])
    ax.set_yticks(range(len(bs)))
    ax.set_yticklabels([f"{b:g}" for b in bs])
    ax.set_xlabel("ordering quantile $q$")
    ax.set_ylabel("lost margin $b$")
    ax.grid(False)
    qstar = [(min(range(len(qs)), key=lambda j: abs(qs[j] - b / (1 + b))), i)
             for i, b in enumerate(bs)]
    ax.plot([j for j, _ in qstar], [i - 0.18 for _, i in qstar], color=INK,
            lw=1.0, ls="--", marker="o", ms=3, mfc="white", zorder=5)
    handles = [Line2D([], [], marker="s", ls="none", ms=5, color=cmap[n],
                      label=short_label(n)) for n in names]
    handles.append(Line2D([], [], color=INK, ls="--", marker="o", ms=3,
                          mfc="white", label="$q^{*} = b/(1+b)$"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=6, handletextpad=0.5)
    fig.tight_layout()
    savefig(fig, FIGS / "fig_costq.png")
    print("  fig_costq")


def fig_eqabl() -> None:
    # 사전지정 분위수 기준 (테스트셋 q 선택 금지 — eq_ablation_q_selection_audit.md).
    # A/B는 유일한 운영점 q=0.5, C/D는 해당 시나리오의 사전지정 q.
    d = pd.read_csv(RESULTS / "eq_ablation_lt.csv")
    PRE_Q = {1.0: 0.7, 3.0: 0.9}
    order = ["A_point", "B_recenter", "C_spread", "D_fullEQ"]
    models = ["ets", "croston", "sba", "tsb", "snaive"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"]

    # sharex 금지: C/D의 운영 분위수가 패널(b)마다 달라 축 라벨이 다르다
    fig, axes = plt.subplots(1, 2, figsize=(PAGE * 0.88, 2.7))
    for ax, (b, qp) in zip(axes, PRE_Q.items()):
        s = d[np.isclose(d.b, b)]
        vals = {}
        for v in order:
            vq = 0.5 if v in ("A_point", "B_recenter") else qp
            sub = s[(s.variant == v) & np.isclose(s.q, vq)]
            vals[v] = sub.set_index("model")["cost_per_unit_demand"]
        x = np.arange(len(order))
        for k, m in enumerate(models):
            ax.plot(x, [vals[v][m] for v in order], marker="o", ms=3.2, lw=1.1,
                    color=cols[k], label=m.upper() if m != "snaive" else "sNaive")
        nl = chr(10)
        nice = {"A_point": f"A{nl}point{nl}($q$=0.5)",
                "B_recenter": f"B{nl}recentre{nl}($q$=0.5)",
                "C_spread": f"C{nl}spread{nl}($q$={qp:g})",
                "D_fullEQ": f"D{nl}full EQ{nl}($q$={qp:g})"}
        ax.set_xticks(x)
        ax.set_xticklabels([nice[v] for v in order], fontsize=6.2)
        panel_label(ax, f"$b = {b:g}$")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("inventory cost per unit demand")
    axes[0].legend(loc="upper right", fontsize=6.2, ncol=2)
    fig.tight_layout()
    savefig(fig, FIGS / "fig_eqabl.png")
    print("  fig_eqabl")


if __name__ == "__main__":
    use_paper_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_origins()
    fig_regimes()
    fig_calibration()
    fig_costq()
    fig_eqabl()
    print("done")
