"""팔레트 검증 (dataviz 스킬 validate_palette.js의 Python 이식판).

node 미설치 환경용. 동일 상수·동일 변환(Machado 2009 CVD, OKLab ΔE×100)을 사용한다.
검사: 밝기 밴드 / 채도 하한 / 인접쌍 CVD 분리 / 정상시야 하한 / 표면 대비.

사용: python scripts/validate_palette.py "#2a78d6,#eb6834,..." [--mode light]
"""

import argparse
import math

CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
BAND = {"light": (0.35, 0.75), "dark": (0.55, 0.90)}
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
CHROMA_MIN = 0.05

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1, max(0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    rows = []

    off = [c for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    rows.append(("Lightness band", not off, f"outside: {off}" if off else f"all {len(palette)} in L {lo}-{hi}"))

    low_c = [c for c in palette if oklch(c)[1] < CHROMA_MIN]
    ok &= not low_c
    rows.append(("Chroma floor", not low_c, f"low chroma: {low_c}" if low_c else "all above floor"))

    pairs = [(i, i + 1) for i in range(len(palette) - 1)]
    worst_cvd, worst_pair = 99.0, None
    for i, j in pairs:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_pair = d, (palette[i], palette[j])
    cvd_ok = worst_cvd >= CVD_TARGET
    cvd_warn = CVD_FLOOR <= worst_cvd < CVD_TARGET
    ok &= (worst_cvd >= CVD_FLOOR)
    rows.append(("Adjacent CVD sep", cvd_ok or cvd_warn,
                 f"min dE={worst_cvd:.1f} {worst_pair}" + (" [WARN: needs secondary encoding]" if cvd_warn else "")))

    worst_n, npair = 99.0, None
    for i, j in pairs:
        d = delta_e(palette[i], palette[j])
        if d < worst_n:
            worst_n, npair = d, (palette[i], palette[j])
    n_ok = worst_n >= NORMAL_FLOOR
    ok &= n_ok
    rows.append(("Normal-vision floor", n_ok, f"min dE={worst_n:.1f} {npair}"))

    bad = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rows.append(("Surface contrast", not bad,
                 f"low: {bad} [WARN: needs labels/table]" if bad else f"all >= {CONTRAST_MIN}:1"))

    return ok, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    a = ap.parse_args()
    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    ok, rows = validate(pal, a.mode)
    for name, passed, detail in rows:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:22s} {detail}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
