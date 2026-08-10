"""TFT 시드별 예측을 통합 스키마로 변환 (lab06 실행).

집계 방식 통일이 목적: 원 리포의 `*_res_*.parquet`는 **일자별 WAPE의 평균**이고
우리 파이프라인은 **전체 합산 WAPE**라 0.1%p대 차이가 난다. 논문 전체를 한 방식으로
맞추기 위해, 시드별 예측 원본을 우리 스키마로 옮겨 동일한 평가 코드를 태운다.

산출: results/zeroshot/tft_{50k|50krec}_s{seed}_raw.parquet (+ meta)
사용: python scripts/integrate_tft_seeds.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
TFT = HOME / "tsfm-demand/external/frn-50k-baseline/demand_forecasting/TFT"
OUT = HOME / "tsfm-demand/results/zeroshot"
KEY = ["store_id", "product_id"]
HORIZON = 7

JOBS = [
    ("censored_predictions_s2025.parquet", "tft_50k_s2025_raw", "frn50k", "censored sales", 2025),
    ("censored_predictions_s1.parquet", "tft_50k_s1_raw", "frn50k", "censored sales", 1),
    ("censored_predictions_s2.parquet", "tft_50k_s2_raw", "frn50k", "censored sales", 2),
    ("recovered_predictions_s1.parquet", "tft_50krec_s1_raw", "frn50k", "recovered demand", 1),
    ("recovered_predictions_s2.parquet", "tft_50krec_s2_raw", "frn50k", "recovered demand", 2),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name, ds, td, seed in JOBS:
        p = TFT / src
        if not p.exists():
            print(f"missing: {src}")
            continue
        d = pd.read_parquet(p).sort_values(KEY + ["dt"]).reset_index(drop=True)
        d["h"] = d.groupby(KEY).cumcount() + 1
        rows = d[KEY + ["h"]].copy()
        rows["q0.5"] = np.clip(d["prediction"].to_numpy(), 0, None)
        out_path = OUT / f"{name}.parquet"
        rows.to_parquet(out_path, index=False)
        meta = {"model": "tft", "dataset": ds, "training_data": td, "seed": seed,
                "mask_stockout": False, "cutoff_date": None, "quantiles": [0.5],
                "horizon": HORIZON, "n_series": rows.groupby(KEY).ngroups,
                "point_only": True,
                "note": "TFT seed replicate; scored with the same pooled-WAPE "
                        "convention as all other models"}
        Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved: {name} ({rows.shape[0]} rows)")
