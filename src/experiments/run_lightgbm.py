"""LightGBM 베이스라인 (M5 우승 레시피): 풀링 global 모델 + Tweedie/분위수 목적함수.

설계 (Makridakis et al. 2022 M5 정리 + notes/makridakis2022-m5-ijf.md):
  - 전 시계열 풀링 학습(cross-learning), 시계열 ID는 범주형 특성
  - Tweedie 손실(0 질량 + 우측 왜도) = 점예측, 추가로 분위수 목적함수로 분포 예측
  - 특성: origin 기준 시차/이동평균 + 타깃일 캘린더·공변량 + horizon
  - 미래 누출 방지: 모든 시차는 forecast origin 이전만 사용 (절대 규칙 4)

출력은 zero-shot 러너와 동일 스키마 → 동일 평가·뉴스벤더 코드 사용.
사용: python -m src.experiments.run_lightgbm --dataset lt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
KEY = ["store_id", "product_id"]
HORIZON = 7
LAGS = [1, 2, 3, 7, 14, 28]
ROLLS = [7, 28]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

COV = ["discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature"]


def build_samples(panel: pd.DataFrame, origins: list[pd.Timestamp],
                  with_target: bool) -> pd.DataFrame:
    """origin별로 h=1..7 샘플 생성. 시차 특성은 origin 이전 관측만 사용."""
    # dropna=False: 예측 구간(타깃 전부 NaN) 열이 삭제되지 않도록
    wide = panel.pivot_table(index=KEY, columns="dt", values="sale_amount",
                             dropna=False)
    dates = list(wide.columns)
    date_pos = {d: i for i, d in enumerate(dates)}
    cov_idx = panel.set_index(KEY + ["dt"])[COV]
    rows = []
    for origin in origins:
        p = date_pos[origin]
        hist = wide.iloc[:, :p + 1]
        feat = pd.DataFrame(index=wide.index)
        for lag in LAGS:
            feat[f"lag{lag}"] = hist.iloc[:, -lag] if hist.shape[1] >= lag else np.nan
        for w in ROLLS:
            feat[f"roll{w}"] = hist.iloc[:, -w:].mean(axis=1)
        feat["roll7_std"] = hist.iloc[:, -7:].std(axis=1)
        feat["nonzero28"] = (hist.iloc[:, -28:] > 0).mean(axis=1)
        for h in range(1, HORIZON + 1):
            tp = p + h
            if tp >= len(dates):
                continue
            tdate = dates[tp]
            f = feat.copy()
            f["h"] = h
            f["dow"] = tdate.dayofweek
            f["origin"] = origin
            f = f.join(cov_idx.xs(tdate, level="dt"), how="left")
            if with_target:
                f["y"] = wide.iloc[:, tp]
            rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["y"]) if with_target else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--n-origins", type=int, default=20, help="학습 origin 수")
    ap.add_argument("--num-boost-round", type=int, default=300)
    ap.add_argument("--out-dir", default="results/zeroshot")
    ap.add_argument("--cutoff-date", default=None,
                    help="(R1 E2) origin별 재학습: train을 이 날짜까지로 자르고 다음 7일을 예측 "
                         "(타깃·공변량은 train 패널에서 구성, 이력 90일 미만 시계열 제외)")
    args = ap.parse_args()

    import lightgbm as lgb

    train = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount"] + COV)
    eval_keys = (pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                                 columns=KEY).drop_duplicates())
    train = train.merge(eval_keys, on=KEY)
    ev = pd.read_parquet(PROCESSED / f"{args.dataset}_daily_eval.parquet",
                         columns=KEY + ["dt"] + COV)
    tag = ""
    if args.cutoff_date:
        cut = pd.Timestamp(args.cutoff_date)
        fut = train[(train["dt"] > cut) & (train["dt"] <= cut + pd.Timedelta(days=HORIZON))]
        train = train[train["dt"] <= cut]
        n_obs = train.groupby(KEY)["dt"].size()
        keep = n_obs[n_obs >= 90].index
        train = train.set_index(KEY).loc[keep].reset_index()
        fut = fut.set_index(KEY).loc[fut.set_index(KEY).index.intersection(keep)].reset_index()
        ev = fut[KEY + ["dt"] + COV]
        tag = f"_mo{args.cutoff_date.replace('-', '')}"
        print(f"origin {args.cutoff_date}: {len(keep)} series with >=90 days", flush=True)

    dates = sorted(train["dt"].unique())
    last = dates[-1]
    # 학습 origin: 마지막 origin에서 7일씩 거슬러 (타깃이 train 내에 있어야 함)
    train_origins = [dates[-(1 + 7 * (i + 1))] for i in range(args.n_origins)
                     if len(dates) > 7 * (i + 1)]
    print(f"train origins: {len(train_origins)} ({train_origins[-1].date()}~{train_origins[0].date()})",
          flush=True)

    tr = build_samples(train, train_origins, with_target=True)
    print("train samples:", tr.shape, flush=True)

    # 예측용: 마지막 origin, 공변량은 eval 구간 실제값 사용 (known covariates)
    panel_all = pd.concat([train, ev.assign(sale_amount=np.nan)], ignore_index=True)
    te = build_samples(panel_all, [last], with_target=False)
    print("predict samples:", te.shape, flush=True)

    feats = [c for c in tr.columns if c not in ["y", "origin"]]
    cat = ["store_id", "product_id", "dow", "holiday_flag", "activity_flag"]
    for df in (tr, te):
        for c in cat:
            df[c] = df[c].astype("category")
    dtrain = lgb.Dataset(tr[feats], label=tr["y"], categorical_feature=cat,
                         free_raw_data=False)

    base = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                verbosity=-1, num_threads=-1, seed=0)

    preds = {}
    m = lgb.train({**base, "objective": "tweedie", "tweedie_variance_power": 1.1},
                  dtrain, num_boost_round=args.num_boost_round)
    preds["tweedie"] = np.clip(m.predict(te[feats]), 0, None)
    print("tweedie done", flush=True)
    for q in QUANTILES:
        mq = lgb.train({**base, "objective": "quantile", "alpha": q},
                       dtrain, num_boost_round=args.num_boost_round)
        preds[f"q{q}"] = np.clip(mq.predict(te[feats]), 0, None)
        print(f"quantile {q} done", flush=True)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = te[KEY + ["h"]].copy()
    for q in QUANTILES:
        rows[f"q{q}"] = preds[f"q{q}"]
    rows["tweedie_point"] = preds["tweedie"]
    rows = rows.sort_values(KEY + ["h"]).reset_index(drop=True)
    out_path = out_dir / f"lightgbm_{args.dataset}{tag}_raw.parquet"
    rows.to_parquet(out_path, index=False)
    meta = {"model": "lightgbm", "dataset": args.dataset, "mask_stockout": False,
            "cutoff_date": args.cutoff_date, "quantiles": QUANTILES, "horizon": HORIZON,
            "n_series": rows.groupby(KEY).ngroups, "n_origins": len(train_origins),
            "num_boost_round": args.num_boost_round, "seed": 0,
            "objectives": ["tweedie", "quantile"]}
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
