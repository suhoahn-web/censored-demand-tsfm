"""실험 통계 검정: 시계열별 오차 → MCB + Friedman–Nemenyi.

근거 (설계서 v2.0 8절):
  - MCB (Multiple Comparisons with the Best): Koning, Franses, Hibon & Stekler (2005, IJF)
  - Friedman–Nemenyi: Demšar (2006, JMLR). M5는 MCB만 사용했고 계층 시계열의
    독립성 위배를 스스로 한계로 인정 → 우리는 두 검정 병행 + 동일 주의 문구.

시계열별 지표는 MASE(계절 naive 스케일, m=7)를 사용한다 — 0-안전하고 스케일 자유로워
5만 개 시계열 순위 통합에 적합 (Kolassa 2016 논의 반영, MAPE 금지).

사용: python -m src.eval.stat_tests --dataset lt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TRUTH_OF = {"frn50krec": "frn50k"}


def per_series_mase(dataset: str, nostock: bool = True) -> pd.DataFrame:
    """모델별 시계열별 MASE 행렬 (index=series, columns=model)."""
    tds = TRUTH_OF.get(dataset, dataset)
    truth = pd.read_parquet(PROCESSED / f"{tds}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    truth = truth.sort_values(KEY + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    if nostock:
        truth = truth[truth["stock_hour6_22_cnt"] == 0]

    ins = pd.read_parquet(PROCESSED / f"{tds}_daily_train.parquet",
                          columns=KEY + ["dt", "sale_amount"]).sort_values(KEY + ["dt"])
    scale = (ins.groupby(KEY)["sale_amount"]
             .apply(lambda s: np.abs(s.to_numpy()[7:] - s.to_numpy()[:-7]).mean())
             .rename("scale"))

    cols = {}
    runs = []
    for meta_path in sorted((RESULTS / "zeroshot").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("cutoff_date") or meta["dataset"] != dataset:
            continue
        name = meta_path.name.replace(".parquet.meta.json", "")
        # 맥락길이 ablation 제외: 본 비교 대상이 아니며, TiRex·Moirai-2는 내부
        # 컨텍스트 한계 탓에 절단본이 전체본과 동일한 예측을 내 순위에 동점을 만든다.
        if name.startswith("ctxabl_"):
            continue
        if re.search(r"_s\d+_raw$", name):
            continue  # 시드 복제본은 Table 7 집계 전용
        if "equ_" in name:
            continue  # EQ 비검열-잔차 변형(R1 E4)은 부록 B.8 전용
        runs.append((name, str(meta_path).replace(".meta.json", "")))
    ft = RESULTS / "finetune_v2"
    if ft.exists():
        for run_dir in sorted(ft.iterdir()):
            mf, pf = run_dir / "meta.json", run_dir / "predictions.parquet"
            if not (mf.exists() and pf.exists()):
                continue
            meta = json.loads(mf.read_text())
            if meta["dataset"] != dataset or meta.get("seed") != 0:
                continue
            runs.append((f"ft_{run_dir.name}", str(pf)))

    for name, path in runs:
        pred = pd.read_parquet(path)
        if "q0.5" not in pred.columns:
            continue
        m = truth[KEY + ["h", "sale_amount"]].merge(
            pred[KEY + ["h", "q0.5"]], on=KEY + ["h"], how="inner")
        mae = (m.assign(e=(m["sale_amount"] - m["q0.5"]).abs())
               .groupby(KEY)["e"].mean())
        cols[name] = (mae / scale.reindex(mae.index)).replace([np.inf, -np.inf], np.nan)
        print(f"  per-series MASE: {name}", flush=True)

    mat = pd.DataFrame(cols).dropna()
    return mat


def mcb_intervals(mat: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """평균 순위 + MCB 신뢰구간 (Koning et al. 2005 방식의 Nemenyi 임계거리 기반)."""
    from scipy.stats import studentized_range
    ranks = mat.rank(axis=1)
    mean_rank = ranks.mean().sort_values()
    k, n = mat.shape[1], mat.shape[0]
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    out = pd.DataFrame({"mean_rank": mean_rank})
    out["lower"] = out["mean_rank"] - cd / 2
    out["upper"] = out["mean_rank"] + cd / 2
    best_upper = out["upper"].iloc[0]
    out["sig_worse_than_best"] = out["lower"] > best_upper
    out.attrs["cd"] = cd
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    cache = RESULTS / f"per_series_mase_{args.dataset}.parquet"
    if cache.exists() and not args.recompute:
        mat = pd.read_parquet(cache)
        print(f"cached matrix: {cache.name}")
    else:
        mat = per_series_mase(args.dataset)
        mat.to_parquet(cache)
    print(f"\nmatrix: {mat.shape[0]} series x {mat.shape[1]} models")

    from scipy.stats import friedmanchisquare
    stat, p = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    print(f"\nFriedman: chi2={stat:.1f}, p={p:.3e}")

    mcb = mcb_intervals(mat, args.alpha)
    print(f"\n=== MCB (평균 순위, 낮을수록 우수), 임계거리 CD={mcb.attrs['cd']:.4f} ===")
    print(mcb.round(3).to_string())

    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(mat.to_numpy())
        nem.index = nem.columns = mat.columns
        nem.to_csv(RESULTS / f"nemenyi_{args.dataset}.csv")
        top2 = mcb.index[:2]
        print(f"\nNemenyi p({top2[0]} vs {top2[1]}) = {nem.loc[top2[0], top2[1]]:.4f}")
    except ImportError:
        print("\n(scikit-posthocs 미설치 — Nemenyi 생략)")

    mcb.to_csv(RESULTS / f"mcb_{args.dataset}.csv")
    print("\nsaved:", RESULTS / f"mcb_{args.dataset}.csv")


if __name__ == "__main__":
    main()
