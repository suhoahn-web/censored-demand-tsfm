"""P0-2 / P0-7: 시계열 단위 오차 집계와 분위수 캘리브레이션(경험적 커버리지).

두 가지 리뷰어 방어 목적:
  1) WQL은 적정 스코어링 규칙이지 캘리브레이션 측정치가 아니다. 본문이 "상단 꼬리
     캘리브레이션"을 주장하려면 명목 분위수 대비 **경험적 커버리지**를 직접 재야 한다.
  2) 매장-상품 시계열 1만 개는 독립 반복이 아니다(같은 매장·같은 상품이 반복 등장).
     군집 부트스트랩을 하려면 시계열 단위 분해값이 필요하다.

산출:
  results/per_series_diag_{dataset}.parquet  — run × 시계열 오차 분해
  results/quantile_calibration_{dataset}.csv — run × 명목분위수 커버리지

사용: python -m src.eval.per_series_diagnostics --dataset lt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.inventory_sim import collect_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TRUTH_OF = {"frn50krec": "frn50k"}
COV_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]   # 발주 분위수 격자와 맞춘 커버리지 측정 지점


def load_truth(dataset: str, nostock_only: bool = True) -> pd.DataFrame:
    """공식 프로토콜: 품절 시간이 0인 타깃 날만 채점."""
    t = pd.read_parquet(
        PROCESSED / f"{TRUTH_OF.get(dataset, dataset)}_daily_eval.parquet",
        columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    t = t.sort_values(KEY + ["dt"]).reset_index(drop=True)
    t["h"] = t.groupby(KEY).cumcount() + 1
    if nostock_only:
        t = t[t["stock_hour6_22_cnt"] == 0]
    return t[KEY + ["h", "sale_amount"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    args = ap.parse_args()

    truth = load_truth(args.dataset)
    print(f"scored rows (uncensored days): {len(truth):,}", flush=True)

    diag_rows, cal_rows = [], []
    for name, pred in collect_predictions(args.dataset):
        m = pred.merge(truth, on=KEY + ["h"], how="inner")
        if m.empty:
            continue
        y = m["sale_amount"].to_numpy()
        qcols = sorted((float(c[1:]), c) for c in m.columns if c.startswith("q"))
        yhat = m["q0.5"].to_numpy()

        # --- 시계열 단위 분해 (부트스트랩 재집계용) ---
        g = pd.DataFrame({
            "store_id": m["store_id"], "product_id": m["product_id"],
            "y": y, "abs_err": np.abs(y - yhat), "err": yhat - y,
        })
        # WQL 분자: 2 × 평균 pinball. 시계열별로 합을 두고 나중에 재집계한다.
        pin = np.zeros(len(m))
        for q, c in qcols:
            e = y - m[c].to_numpy()
            pin += np.maximum(q * e, (q - 1) * e)
        g["pinball"] = 2.0 * pin / len(qcols)
        # 군집 부트스트랩용 커버리지 히트 수 (명목 분위수별)
        have = {q: c for q, c in qcols}
        spec = {}
        for lvl in COV_LEVELS:
            if lvl in have:
                g[f"hit{lvl}"] = (y <= m[have[lvl]].to_numpy()).astype(float)
                spec[f"hits_{lvl}"] = (f"hit{lvl}", "sum")
        agg = g.groupby(KEY, as_index=False).agg(
            n_days=("y", "size"), y_sum=("y", "sum"),
            abs_err_sum=("abs_err", "sum"), err_sum=("err", "sum"),
            pinball_sum=("pinball", "sum"), **spec)
        agg.insert(0, "run", name)
        diag_rows.append(agg)

        # --- 분위수 캘리브레이션 ---
        for q, c in qcols:
            cov = float((y <= m[c].to_numpy()).mean())
            cal_rows.append({"run": name, "nominal": q, "coverage": cov,
                             "coverage_error": cov - q, "n": len(m)})
        print(f"  {name}: {len(qcols)} quantiles", flush=True)

    diag = pd.concat(diag_rows, ignore_index=True)
    # 기존 시계열 MASE를 붙여 둔다 (부트스트랩에서 함께 재집계)
    mp = RESULTS / f"per_series_mase_{args.dataset}.parquet"
    if mp.exists():
        mase = (pd.read_parquet(mp).stack().rename("mase")
                .reset_index().rename(columns={"level_2": "run"}))
        diag = diag.merge(mase, on=KEY + ["run"], how="left")
    out1 = RESULTS / f"per_series_diag_{args.dataset}.parquet"
    diag.to_parquet(out1, index=False)

    cal = pd.DataFrame(cal_rows)
    out2 = RESULTS / f"quantile_calibration_{args.dataset}.csv"
    cal.to_csv(out2, index=False)

    print(f"\nsaved: {out1.name} ({len(diag):,} rows, {diag.run.nunique()} runs)")
    print(f"saved: {out2.name} ({len(cal):,} rows)")
    print("\n=== 상단 분위수 커버리지 (명목 0.9) ===")
    top = cal[np.isclose(cal.nominal, 0.9)].sort_values("coverage_error")
    print(top[["run", "coverage", "coverage_error"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
