"""P0-6: 다중 forecast origin 결과 집계 — 순위 안정성.

단일 origin에서 얻은 순위는 origin 이동만으로 흔들릴 수 있다. 8개 origin(56일 간격,
12개월)에서 같은 기법들을 다시 평가하고, 기법별 평균·변동·평균순위·상위권 진입 빈도를
보고한다.

범위: zero-shot TSFM과 origin별 재적합한 고전 기법만. 파인튜닝 Chronos-2와 LightGBM은
LT train 패널 전체로 학습했으므로 과거 origin의 정답 구간이 학습에 포함되어 제외한다.

산출:
  results/multi_origin_{dataset}.csv         — origin × 기법 지표
  results/multi_origin_summary_{dataset}.csv — 기법별 요약

사용: python -m src.eval.multi_origin_summary --dataset lt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.rolling_origin import truth_after

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
TAG_RE = re.compile(r"_mo(\d{8})_")


def prepare_origin(dataset: str, cutoff: str):
    """origin 하나에 대해 정답(7일 완비·무품절일만)과 MASE 스케일을 한 번만 만든다."""
    truth, insample = truth_after(dataset, cutoff)
    truth = truth.sort_values(KEY + ["dt"]).copy()
    truth["h"] = truth.groupby(KEY).cumcount() + 1
    cnt = truth.groupby(KEY)["h"].transform("size")
    truth = truth[cnt == 7]
    scored = truth[truth.stock_hour6_22_cnt == 0][KEY + ["h", "sale_amount"]]

    ins = insample.sort_values(KEY + ["dt"])
    lag = ins.groupby(KEY)["sale_amount"].shift(7)
    ins = ins.assign(d=(ins["sale_amount"] - lag).abs())
    scale = ins.groupby(KEY)["d"].mean().replace(0, np.nan).rename("scale")
    return scored, scale


def metrics_for(pred: pd.DataFrame, scored: pd.DataFrame,
                scale: pd.Series) -> dict:
    """벡터화한 WAPE/WPE/MASE/WQL/CRPS (무품절일 채점)."""
    m = pred.merge(scored, on=KEY + ["h"], how="inner")
    if m.empty:
        return {}
    y = m["sale_amount"].to_numpy()
    yhat = m["q0.5"].to_numpy()
    qcols = sorted((float(c[1:]), c) for c in m.columns if c.startswith("q"))
    pin = np.zeros(len(m))
    for q, c in qcols:
        e = y - m[c].to_numpy()
        pin += np.maximum(q * e, (q - 1) * e)
    pin *= 2.0 / len(qcols)
    per = (pd.DataFrame({"k1": m.store_id, "k2": m.product_id,
                         "ae": np.abs(y - yhat)})
           .groupby(["k1", "k2"])["ae"].mean())
    per.index.names = KEY
    mase = (per / scale.reindex(per.index)).replace([np.inf, -np.inf], np.nan)
    return {"wape": float(np.abs(y - yhat).sum() / y.sum()),
            "wpe": float((yhat - y).sum() / y.sum()),
            "mase_mean": float(np.nanmean(mase)),
            "wql": float(pin.sum() / y.sum()),
            "crps": float(pin.mean() / 2 * 2),
            "n_series": int(m.groupby(KEY).ngroups)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt")
    args = ap.parse_args()

    rows = []
    cache = {}
    for meta_path in sorted((RESULTS / "zeroshot").glob(f"*_mo*_*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("dataset") != args.dataset or not meta.get("cutoff_date"):
            continue
        name = meta_path.name.replace(".parquet.meta.json", "")
        m = TAG_RE.search(name)
        if not m:
            continue
        cutoff = meta["cutoff_date"]
        if cutoff not in cache:
            cache[cutoff] = prepare_origin(args.dataset, cutoff)
            print(f"origin {cutoff} prepared", flush=True)
        scored, scale = cache[cutoff]
        pred = pd.read_parquet(str(meta_path).replace(".meta.json", ""))
        met = metrics_for(pred, scored, scale)
        if not met:
            continue
        method = TAG_RE.sub("_", name)
        rows.append({"method": method, "origin": cutoff, "run": name, **met})
        print(f"  {name}: WAPE {met['wape']:.4f} (n={met['n_series']})", flush=True)

    if not rows:
        print("no multi-origin runs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"multi_origin_{args.dataset}.csv", index=False)

    # 기법이 전 origin에 존재하는 경우만 순위 비교 (결측 origin은 순위를 왜곡)
    full_cov = df.groupby("method")["origin"].nunique()
    keep = full_cov[full_cov == df.origin.nunique()].index
    d = df[df.method.isin(keep)].copy()
    d["rank_wape"] = d.groupby("origin")["wape"].rank()
    d["rank_wql"] = d.groupby("origin")["wql"].rank()

    top3 = (d[d.rank_wape <= 3].groupby("method").size()
            / d.origin.nunique()).rename("top3_freq")
    summ = d.groupby("method").agg(
        n_origins=("origin", "nunique"),
        wape_mean=("wape", "mean"), wape_sd=("wape", "std"),
        wape_min=("wape", "min"), wape_max=("wape", "max"),
        wpe_mean=("wpe", "mean"), wpe_sd=("wpe", "std"),
        wql_mean=("wql", "mean"), wql_sd=("wql", "std"),
        rank_wape_mean=("rank_wape", "mean"), rank_wape_sd=("rank_wape", "std"),
        rank_wql_mean=("rank_wql", "mean"),
    ).join(top3).fillna({"top3_freq": 0.0}).sort_values("rank_wape_mean")
    summ.to_csv(RESULTS / f"multi_origin_summary_{args.dataset}.csv")

    print(f"\n=== {d.origin.nunique()}개 origin, {len(summ)}개 기법 ===")
    print(summ.round(4).to_string())
    print("\norigin 간 WAPE 변동폭(기법 평균):",
          round(float(summ.wape_sd.mean()), 4),
          "| 기법 간 WAPE 격차(origin 평균):",
          round(float(d.groupby("origin")["wape"].apply(np.ptp).mean()), 4))


if __name__ == "__main__":
    main()
