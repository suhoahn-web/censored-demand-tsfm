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


EDGE = "#8f8f8b"
BLUE = "#cddcee"     # 데이터·산출물 (Sousa et al. 2025 Fig.1 관행)
ORANGE = "#f3cfae"   # 방법·처리
GRAY = "#e7e7e3"     # 평가


def draw() -> None:
    """ESWA 관행(키페이퍼 Sousa et al. 2025, Fig. 1)을 따른 상하 파이프라인
    플로차트: 단계별 점선 프레임 + 좌측 회전 라벨 + 셰브런 화살표."""
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc=fc, ec=EDGE, lw=0.6, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3)

    def chevron(cx, ytop, ybot):
        ax.add_patch(Polygon([(cx - 2.2, ytop), (cx + 2.2, ytop), (cx, ybot)],
                             closed=True, fc="#e9e9e5", ec=EDGE, lw=0.5, zorder=2))

    def frame(x, y, w, h, label):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="none", ec=EDGE, lw=0.7, ls=(0, (4, 2.5)),
                                    zorder=1))
        ax.text(x + 2.2, y + h / 2, label, ha="center", va="center", rotation=90,
                fontsize=6.2, fontstyle="italic", color=INK_2, zorder=2)

    # 데이터
    box(27, 90, 46, 8.5,
        "Hourly sales with stockout annotations:  $y_t=\\min(D_t,A_t)$", BLUE)
    chevron(50, 89.2, 86.4)

    # 1단계: 검열 처리
    frame(7, 66, 89, 19, "Censoring\ntreatments")
    box(13, 69, 24, 12, "Raw sales\n(as recorded)", ORANGE)
    box(41.5, 69, 24, 12, "Stockout masking\n(censored days\n$\\to$ missing)", ORANGE)
    box(70, 69, 24, 12, "Latent demand\nrecovery\n(imputation model)", ORANGE)
    chevron(50, 65.2, 62.4)

    # 2단계: 예측
    frame(7, 41, 89, 20, "Forecasting")
    box(11.5, 44, 20, 13.5, "Foundation\nmodels,\nzero-shot ($\\times 5$)", ORANGE)
    box(33.5, 44, 20, 13.5, "Chronos-2\nadapted (LoRA /\nfull fine-tuning)", ORANGE)
    box(55.5, 44, 20, 13.5, "LightGBM\n(Tweedie and\nquantile)", ORANGE)
    box(77.5, 44, 17, 13.5, "Classical +\nempirical\nquantiles", ORANGE)
    chevron(50, 40.2, 37.4)

    # 3단계: 발주 결정
    frame(7, 22.5, 89, 12.5, "Decision")
    box(20, 25, 60, 8,
        "Order-up-to replenishment at fixed quantile $q$\n"
        "(three cost scenarios; shelf life, FIFO issuing, waste)", BLUE)
    chevron(50, 21.7, 18.9)

    # 평가
    box(20, 9, 60, 8.5,
        "Evaluation on uncensored days:\naccuracy, bias, WQL; inventory cost and "
        "fill rate", GRAY)

    fig.tight_layout(pad=0.3)
    savefig(fig, FIGDIR / "fig1_framework.png")


if __name__ == "__main__":
    use_paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    draw()
    print("saved:", FIGDIR / "fig1_framework.png")
