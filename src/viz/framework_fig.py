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
    ax.set_ylim(0, 60)
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
                linespacing=1.4)

    # 사각 순환: 판매(좌상) → 예측(우상) → 발주(우하) → 가용재고(좌하) → 판매
    box(21, 42, 24, 13, "Observed sales\n$y_t=\\min(D_t,A_t)$")
    box(60, 42, 19, 13, "Forecast\ndistribution")
    box(60, 5, 19, 13, "Order\nquantity $q_t$")
    box(21, 5, 24, 13, "Available stock\n$A_t$", fc="#f2f2ef")
    box(1, 24, 15, 12, "Latent\ndemand $D_t$")

    arrow((16, 30), (26, 42), rad=-0.18)                       # 수요 → 판매
    arrow((45, 48.5), (60, 48.5))                              # 판매 → 예측 (fit)
    lab(52.5, 51.4, "fit")
    lab(52.5, 45.6, "(1) mask\n(2) recover", color=orange, fs=5.7)
    arrow((69.5, 42), (69.5, 18))                              # 예측 → 발주
    lab(71.3, 30, "(3) order at\nquantile $q$", color=orange, ha="left", fs=5.7)
    arrow((60, 11.5), (45, 11.5))                              # 발주 → 재고
    lab(52.5, 14.3, "next period's\nstock", fs=5.5)
    arrow((33, 18), (33, 42))                                  # 재고 → 판매 (절단)
    lab(31.3, 30, "truncates", ha="right")

    # 평가 (점선 유입)
    box(84, 20, 15.5, 20, "Evaluation on\nuncensored\ndays: accuracy,\nbias, cost",
        fc="#f2f2ef", fs=6.0)
    arrow((79, 45.5), (88, 40), style=":")
    arrow((79, 14.5), (88, 20), style=":")

    fig.tight_layout(pad=0.2)
    savefig(fig, FIGDIR / "fig1_framework.png")


if __name__ == "__main__":
    use_paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    draw()
    print("saved:", FIGDIR / "fig1_framework.png")
