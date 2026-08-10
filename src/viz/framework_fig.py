"""Fig.1 개념도 — 검열의 자기강화 순환과 세 개입 지점.

이름이 아니라 경로를 그린다: 수요가 가용재고에 잘려 판매로 기록되고, 그 판매로
학습한 모델이 낮게 예측하며, 낮은 예측이 적은 발주를 낳아 다음 기 가용재고를 줄인다.
세 처리(마스킹·복원·발주 분위수)가 이 순환의 어디에 개입하는지 표시한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.viz.style import FAMILY_COLORS, INK_2, ONE_HALF, savefig, use_paper_style

FIGDIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def draw() -> None:
    orange = FAMILY_COLORS["TSFM (fine-tuned)"]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", fs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=INK_2, lw=0.7, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.35)

    def arrow(p1, p2, color=INK_2, style="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
                                     color=color, lw=0.8, linestyle=style,
                                     shrinkA=1, shrinkB=1, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}"))

    def lab(x, y, t, color=INK_2, fs=5.9, ha="center", va="center"):
        ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, zorder=6,
                linespacing=1.4, bbox=dict(fc="white", ec="none", pad=0.9))

    # 상단 파이프라인
    box(1, 38, 18, 13, "Latent\ndemand $D_t$")
    box(26, 38, 21, 13, "Observed sales\n$y_t=\\min(D_t,A_t)$")
    box(56, 38, 19, 13, "Forecast\ndistribution")
    box(81, 38, 18, 13, "Order\nquantity")
    arrow((19, 44.5), (26, 44.5))
    arrow((47, 44.5), (56, 44.5))
    lab(51.5, 47.6, "fit")
    arrow((75, 44.5), (81, 44.5))

    # 되먹임 순환: 발주 → 가용재고 → 절단
    box(26, 12, 21, 11, "Availability $A_t$", fc="#f2f2ef")
    arrow((90, 38), (47, 19), rad=0.22)
    lab(83, 24.5, "sets next\nperiod's stock")
    arrow((32, 23), (32, 38))
    lab(24.5, 30.5, "truncates", ha="right")

    # 개입 지점
    arrow((41, 34), (41, 27.5), color=orange, style="--")
    lab(43, 30.5, "(1) mask\n(2) recover", color=orange, ha="left")
    ax.plot([88, 88], [38, 34.5], ls="--", lw=0.8, color=orange, zorder=2)
    lab(88, 32.8, "(3) ordering\nquantile", color=orange, va="top")

    # 평가
    box(56, 12, 43, 11, "Scored on uncensored days only:\n"
                        "accuracy, bias, and inventory cost", fc="#f2f2ef", fs=6.2)
    arrow((62, 38), (62, 23), style=":")

    ax.text(1, 8.5, "Censoring is self-reinforcing: a downward-biased forecast orders "
                    "less,\nwhich raises the chance of another stockout next period.",
            fontsize=6, color=INK_2, va="top", linespacing=1.5)
    fig.tight_layout(pad=0.2)
    savefig(fig, FIGDIR / "fig1_framework.png")


if __name__ == "__main__":
    use_paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    draw()
    print("saved:", FIGDIR / "fig1_framework.png")
