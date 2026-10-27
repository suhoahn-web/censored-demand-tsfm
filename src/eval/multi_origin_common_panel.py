"""P1 Task 5: 다중 origin 결과를 '공통 패널'에서 재계산한다.

기존 분석은 origin마다 가용 시계열이 5,500~6,849개로 달라, origin 간 변동에
(1) 시점 효과와 (2) 상품 구성 변화가 섞여 있다. 모든 origin에서
  - 90일 이상 이력(min-context 필터 통과 = 예측 존재)과
  - 7일 완비 정답 및 1일 이상의 무품절 채점일
을 가진 시계열 교집합으로 제한해 시점 효과만 남긴다.

산출:
  results/multi_origin_common_panel_ids.csv
  results/multi_origin_common_panel_results.csv
  results/multi_origin_common_panel_summary.csv

사용: python -m src.eval.multi_origin_common_panel
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.multi_origin_summary import TAG_RE, metrics_for, prepare_origin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]


def main() -> None:
    dataset = "lt"
    metas = sorted((RESULTS / "zeroshot").glob("*_mo*_*.meta.json"))
    by_origin = {}
    for mp in metas:
        meta = json.loads(mp.read_text())
        if meta.get("dataset") != dataset or not meta.get("cutoff_date"):
            continue
        by_origin.setdefault(meta["cutoff_date"], []).append(mp)
    origins = sorted(by_origin)
    print(f"origins: {len(origins)}")

    # ---- 공통 패널: 각 origin에서 (예측 존재 ∩ 무품절 채점일>=1) 시계열의 교집합 ----
    scored_cache, scale_cache, eligible = {}, {}, {}
    for cut in origins:
        scored, scale = prepare_origin(dataset, cut)
        scored_cache[cut], scale_cache[cut] = scored, scale
        # 예측이 존재하는 시계열: chronos2 raw 파일 기준 (min-context 필터 공통)
        ref = next(p for p in by_origin[cut] if "chronos2" in p.name
                   and "raw" in p.name)
        pred = pd.read_parquet(str(ref).replace(".meta.json", ""), columns=KEY)
        have_pred = set(map(tuple, pred.drop_duplicates().to_numpy()))
        have_scored = set(map(tuple,
                              scored[KEY].drop_duplicates().to_numpy()))
        eligible[cut] = have_pred & have_scored
        print(f"  {cut}: pred {len(have_pred):,} & scored {len(have_scored):,} "
              f"-> {len(eligible[cut]):,}")

    common = set.intersection(*eligible.values())
    ids = pd.DataFrame(sorted(common), columns=KEY)
    ids.to_csv(RESULTS / "multi_origin_common_panel_ids.csv", index=False)
    print(f"\ncommon panel: {len(ids):,} series | "
          f"{ids.store_id.nunique():,} stores | "
          f"{ids.product_id.nunique():,} products")

    # ---- 공통 패널에서 지표 재계산 ----
    rows = []
    common_idx = pd.MultiIndex.from_frame(ids)
    for cut in origins:
        scored = scored_cache[cut]
        scored = scored[scored.set_index(KEY).index.isin(common_idx)]
        scale = scale_cache[cut]
        for mp in by_origin[cut]:
            name = mp.name.replace(".parquet.meta.json", "")
            pred = pd.read_parquet(str(mp).replace(".meta.json", ""))
            met = metrics_for(pred, scored, scale)
            if not met:
                continue
            rows.append({"method": TAG_RE.sub("_", name), "origin": cut, **met})
        print(f"  evaluated origin {cut}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multi_origin_common_panel_results.csv", index=False)

    full_cov = df.groupby("method")["origin"].nunique()
    keep = full_cov[full_cov == len(origins)].index
    d = df[df.method.isin(keep)].copy()
    d["rank_wape"] = d.groupby("origin")["wape"].rank()
    d["rank_wql"] = d.groupby("origin")["wql"].rank()
    summ = d.groupby("method").agg(
        wape_mean=("wape", "mean"), wape_sd=("wape", "std"),
        wape_min=("wape", "min"), wape_max=("wape", "max"),
        wql_mean=("wql", "mean"),
        rank_wape_mean=("rank_wape", "mean"), rank_wape_sd=("rank_wape", "std"),
        rank_wql_mean=("rank_wql", "mean")).sort_values("rank_wape_mean")
    summ.to_csv(RESULTS / "multi_origin_common_panel_summary.csv")
    print("\n=== 공통 패널 요약 ===")
    print(summ.round(4).to_string())

    # ---- 비교표: 원 패널 대 공통 패널 ----
    orig = pd.read_csv(RESULTS / "multi_origin_summary_lt.csv").set_index("method")
    zs = ["chronos2_lt_raw", "moirai2_lt_raw", "tirex_lt_raw"]
    eqs = ["etseq_lt_raw", "crostoneq_lt_raw", "sbaeq_lt_raw", "tsbeq_lt_raw"]
    print("\n=== 원 패널(가변) 대 공통 패널 ===")
    c2o, c2c = orig.loc["chronos2_lt_raw"], summ.loc["chronos2_lt_raw"]
    print(f"Chronos-2 WAPE range : {c2o.wape_min:.4f}-{c2o.wape_max:.4f}"
          f"  vs  {c2c.wape_min:.4f}-{c2c.wape_max:.4f}")
    print(f"Chronos-2 WAPE sd    : {c2o.wape_sd:.4f}  vs  {c2c.wape_sd:.4f}")
    print(f"WQL 1위 유지(mean rank): {orig.loc['chronos2_lt_raw','rank_wql_mean']:.2f}"
          f"  vs  {c2c.rank_wql_mean:.2f}")
    print(f"zero-shot WAPE sd    : {orig.loc[zs,'wape_sd'].mean():.4f}"
          f"  vs  {summ.loc[zs,'wape_sd'].mean():.4f}")
    print(f"classical+EQ WAPE sd : {orig.loc[eqs,'wape_sd'].mean():.4f}"
          f"  vs  {summ.loc[eqs,'wape_sd'].mean():.4f}")
    print(f"rank sd(전 기법 평균)  : {orig.rank_wape_sd.mean():.2f}"
          f"  vs  {summ.rank_wape_sd.mean():.2f}")


if __name__ == "__main__":
    main()
