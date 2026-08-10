"""Rolling origin 평가 — 설계서가 약속한 3개 origin의 안정성 검증.

공식 origin(마지막 7일) 외에 컨텍스트를 7일·14일 앞당긴 두 origin에서도 예측을
생성해 두었다(`ctxabl_`이 아닌 `*_o1_*`, `*_o2_*` 파일). 이들의 정답은 train 패널
안에 있으므로 별도로 구성해야 하며, `evaluate_runs`는 cutoff_date가 있는 실행을
건너뛴다. 이 모듈이 그 공백을 메운다.

사용: python -m src.eval.rolling_origin --dataset lt
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.metrics import evaluate_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
HORIZON = 7


def truth_after(dataset: str, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cutoff 다음 7일을 정답으로, 그 이전을 in-sample로 잘라 반환."""
    train = pd.read_parquet(PROCESSED / f"{dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    cut = pd.Timestamp(cutoff)
    future = train[(train["dt"] > cut) &
                   (train["dt"] <= cut + pd.Timedelta(days=HORIZON))].copy()
    future = future.sort_values(KEY + ["dt"])
    insample = train[train["dt"] <= cut][KEY + ["dt", "sale_amount"]]
    return future, insample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    args = ap.parse_args()

    rows = []
    for meta_path in sorted((RESULTS / "zeroshot").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        cutoff = meta.get("cutoff_date")
        if not cutoff or meta["dataset"] != args.dataset:
            continue
        pred = pd.read_parquet(str(meta_path).replace(".meta.json", ""))
        truth, insample = truth_after(args.dataset, cutoff)
        name = meta_path.name.replace(".parquet.meta.json", "")
        for nostock in [False, True]:
            m = evaluate_forecast(pred, truth, insample, nostock_only=nostock)
            rows.append({"run": name, "cutoff": cutoff,
                         "mask": meta["mask_stockout"],
                         "eval_filter": "nostock" if nostock else "all", **m})
        print(f"  evaluated: {name} (cutoff {cutoff})", flush=True)

    if not rows:
        print("no rolling-origin runs found")
        return
    df = pd.DataFrame(rows)
    out = RESULTS / f"rolling_origin_{args.dataset}.csv"
    df.to_csv(out, index=False)

    # 공식 origin 결과를 함께 붙여 origin 간 변동을 본다
    off = pd.read_csv(RESULTS / "summary_metrics.csv")
    off = off[(off.eval_filter == "nostock") & (off.dataset == args.dataset) &
              (off.run.isin(["chronos2_lt_raw", "chronos2_lt_mask"]))]
    print("\n=== origin별 WAPE / WPE (nostock 기준) ===")
    show = df[df.eval_filter == "nostock"][["run", "cutoff", "wape", "wpe"]]
    print(show.round(4).to_string(index=False))
    print("\n공식 origin (2025-07-13 컷):")
    print(off[["run", "wape", "wpe"]].round(4).to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
