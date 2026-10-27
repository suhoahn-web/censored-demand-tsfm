"""E3 (JORS R1, Minor 8): 저장된 검증-origin 예측으로 임의 점예측 모델에 경험적 분위수 부여.

`add_empirical_quantiles.py`는 고전 5종을 statsforecast로 재적합해 검증 잔차를 얻는다.
TTM처럼 외부 러너가 예측하는 모델은 검증 origin(train 마지막 −7일)에서 생성한
예측 파일을 입력으로 받아 같은 절차(시계열 평균 스케일링, horizon별 풀링, 분위수
교차 방지)를 적용한다.

사용: python -m src.experiments.add_eq_from_val_preds --dataset lt \
        --test-pred results/zeroshot/ttm_lt_raw.parquet \
        --val-pred results/zeroshot/ttm_lt_val_raw.parquet --name ttmeq
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
HORIZON = 7
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k"])
    ap.add_argument("--test-pred", required=True)
    ap.add_argument("--val-pred", required=True)
    ap.add_argument("--name", required=True, help="출력 접두사, 예: ttmeq")
    ap.add_argument("--uncensored-only", action="store_true")
    args = ap.parse_args()

    train = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    eval_keys = (pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                                 columns=KEY).drop_duplicates())
    train = train.merge(eval_keys, on=KEY)
    last = train["dt"].max()
    cutoff = last - pd.Timedelta(days=HORIZON)
    holdout = train[train["dt"] > cutoff].sort_values(KEY + ["dt"]).copy()
    holdout["h"] = holdout.groupby(KEY).cumcount() + 1
    if args.uncensored_only:
        holdout = holdout[holdout["stock_hour6_22_cnt"] == 0]
    scale = (train.groupby(KEY)["sale_amount"].mean().replace(0, np.nan)
             .fillna(1.0).rename("scale"))

    val = pd.read_parquet(PROJECT_ROOT / args.val_pred)
    vmeta = json.loads(Path(str(PROJECT_ROOT / args.val_pred) + ".meta.json").read_text())
    assert pd.Timestamp(vmeta["cutoff_date"]) == cutoff, \
        f"validation cutoff {vmeta['cutoff_date']} != {cutoff.date()}"
    merged = val.merge(holdout[KEY + ["h", "sale_amount"]], on=KEY + ["h"], how="inner")
    merged = merged.join(scale, on=KEY)
    print(f"validation rows: {len(merged)}", flush=True)
    e = ((merged["sale_amount"] - merged["q0.5"]) / merged["scale"]).to_numpy()
    hs = merged["h"].to_numpy()
    qtab = {h: np.nanquantile(e[hs == h], QUANTILES) for h in range(1, HORIZON + 1)}

    pred = pd.read_parquet(PROJECT_ROOT / args.test_pred).join(scale, on=KEY)
    tmeta = json.loads(Path(str(PROJECT_ROOT / args.test_pred) + ".meta.json").read_text())
    point = pred["q0.5"].to_numpy()
    out = pred[KEY + ["h"]].copy()
    for j, q in enumerate(QUANTILES):
        adj = np.array([qtab[h][j] for h in pred["h"].to_numpy()])
        out[f"q{q}"] = np.clip(point + pred["scale"].to_numpy() * adj, 0, None)
    qcols = [f"q{q}" for q in QUANTILES]
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)

    dst = RESULTS / "zeroshot" / f"{args.name}_{args.dataset}_raw.parquet"
    out.to_parquet(dst, index=False)
    meta = {"model": f"{tmeta.get('model')}+empirical-quantiles",
            "base_run": Path(args.test_pred).name, "validation_run": Path(args.val_pred).name,
            "checkpoint_hash": tmeta.get("checkpoint_hash"),
            "dataset": args.dataset, "mask_stockout": tmeta.get("mask_stockout", False),
            "cutoff_date": None, "quantiles": QUANTILES, "horizon": HORIZON,
            "n_series": int(out.groupby(KEY).ngroups),
            "quantile_source": "empirical residuals at validation origin (train last -7d)"
                               + (", uncensored days only" if args.uncensored_only
                                  else ", all validation days"),
            "validation_rows": int(len(merged)),
            "scaled_by": "series mean of train sale_amount"}
    Path(str(dst) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("saved:", dst.name, flush=True)


if __name__ == "__main__":
    main()
