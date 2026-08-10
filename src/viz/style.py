"""논문 그림 공통 스타일 (Elsevier/ESWA 규격 + 예측 문헌 관행).

관행 근거: Theodorou et al. (2025, EJOR) Fig. 4–6 — 소형 패싯, 좌상단 패싯 라벨,
직접 텍스트 라벨, 파선 회귀/기준선, 얇은 축, 작은 산세리프 폰트, 우측 범례.

레이아웃 규격:
  단일 컬럼 90mm(3.54in) / 1.5컬럼 140mm(5.51in) / 더블 190mm(7.48in)

색: 모델 계열(5)에 고정 순서로 배정 — 순환 금지. dataviz 검증 통과
(인접 CVD ΔE=9.2 ≥ 8, 정상시야 27.6 ≥ 15). aqua의 표면 대비 WARN은
모든 그림이 직접 라벨을 다는 것으로 해소한다.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE, ONE_HALF, DOUBLE = 3.54, 5.51, 7.48

# 계열 색 — 고정 순서 (dataviz: 절대 순환하지 않음)
FAMILY_COLORS = {
    "TSFM (zero-shot)": "#2a78d6",
    "TSFM (fine-tuned)": "#eb6834",
    "Tree/ML": "#1baf7a",
    "Local DL": "#4a3aa7",
    "Statistical": "#e34948",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
REFLINE = "#8a8985"


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,          # Elsevier 권장 벡터/고해상
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(ax, text: str) -> None:
    """좌상단 패싯 라벨 (EJOR Fig.4 'R=3' 관행)."""
    ax.text(0.015, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK, bbox=dict(facecolor="white", edgecolor="none",
                                             pad=1.2, alpha=0.85))


def family_of(run: str) -> str:
    r = run.lower()
    if r.startswith("ft_") or "lora" in r or "full_s" in r:
        return "TSFM (fine-tuned)"
    if any(k in r for k in ["chronos", "tirex", "moirai", "timesfm", "ttm"]):
        return "TSFM (zero-shot)"
    if "lightgbm" in r:
        return "Tree/ML"
    if "tft" in r or "dlinear" in r:
        return "Local DL"
    return "Statistical"


BASE_NAMES = {"chronos2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2",
              "timesfm": "TimesFM", "ttm": "TTM", "lightgbm": "LightGBM",
              "ets": "ETS", "croston": "Croston", "tsb": "TSB", "sba": "SBA",
              "snaive": "sNaive", "tft": "TFT", "dlinear": "DLinear"}


def short_label(run: str) -> str:
    """그림용 모델명 — 조건(mask/rank/lr/복원)을 잃지 않고 구별 가능하게."""
    if run.startswith("ft_"):  # 파인튜닝 실행
        r = run[3:]
        base = "C2"
        if "rec" in r:
            base += " [rec]"
        if "full" in r:
            tag = f"{base} full-FT"
        else:
            rank = r.split("_r")[1].split("_")[0] if "_r" in r else "?"
            tag = f"{base} LoRA r{rank}"
            if "lr1e-3" in r:
                tag += " (lr 1e-3)"
            elif "lr1e-5" in r:
                tag += " (lr 1e-5)"
        if "mask" in r:
            tag += " + mask"
        return tag

    parts = run.split("_")
    head = parts[0]
    eq = head.endswith("eq")
    key = head[:-2] if eq else head
    name = BASE_NAMES.get(key, key)
    if eq:
        name += "+EQ"
    if "mask" in run:
        name += " + mask"
    if "rec" in run:
        name += " [rec]"
    return name


def is_ablation(run: str) -> bool:
    """맥락 길이 ablation 등 본 그림에서 제외할 실행."""
    return run.startswith("ctxabl_")


def savefig(fig, path, also_pdf: bool = True) -> None:
    fig.savefig(path, dpi=600)
    if also_pdf:
        fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
