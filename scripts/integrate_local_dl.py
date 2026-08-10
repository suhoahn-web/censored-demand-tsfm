"""TFT·DLinear 예측을 통합 스키마로 변환 (lab06 실행).

원 논문 리포는 자체 포맷으로 저장하므로 그대로는 MCB/재고 시뮬레이션에 못 들어간다.
  TFT     : {cond}_epoch4_predictions.parquet (store_id/product_id/dt/prediction)
  DLinear : real_prediction.npy (350000,) = 50000 시계열 × 7일 평탄화

DLinear는 순서 정보가 없으므로 (store_id, product_id, dt) 정렬 가정으로 매핑한 뒤
원 리포가 보고한 WAPE와 대조해 검증한다 (불일치 시 저장하지 않음).
점예측이므로 q0.5만 기록 — 분위수는 add_empirical_quantiles로 별도 부여.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
BASE = HOME / "tsfm-demand"
TFT_DIR = BASE / "external/frn-50k-baseline/demand_forecasting/TFT"
DL_DIR = BASE / "external/frn-50k-baseline/demand_forecasting/DLinear/results"
OUT = BASE / "results/zeroshot"
KEY = ["store_id", "product_id"]
HORIZON = 7


def save(rows: pd.DataFrame, name: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.parquet"
    rows.to_parquet(p, index=False)
    meta = {"model": name.split("_")[0], "dataset": "frn50k", "mask_stockout": False,
            "cutoff_date": None, "quantiles": [0.5], "horizon": HORIZON,
            "n_series": rows.groupby(KEY).ngroups, "point_only": True, "note": note}
    Path(str(p) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("saved:", p.name, rows.shape)


def wape(y, yhat, mask):
    y, yhat = np.asarray(y)[mask], np.asarray(yhat)[mask]
    return float(np.abs(y - yhat).sum() / y.sum())


# ── TFT ────────────────────────────────────────────────────────────────────────
for cond, tag in [("censored", "tft_50k_raw"), ("recovered", "tft_50krec_raw")]:
    f = TFT_DIR / f"{cond}_epoch4_predictions.parquet"
    if not f.exists():
        print("missing:", f.name)
        continue
    d = pd.read_parquet(f).sort_values(KEY + ["dt"]).reset_index(drop=True)
    d["h"] = d.groupby(KEY).cumcount() + 1
    rows = d[KEY + ["h"]].copy()
    rows["q0.5"] = np.clip(d["prediction"].to_numpy(), 0, None)
    save(rows, tag, f"TFT ({cond}) from frn-50k-baseline, epoch4")

# ── DLinear ────────────────────────────────────────────────────────────────────
eval_df = pd.read_parquet(BASE / "data/processed/frn50k_daily_eval.parquet",
                          columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
eval_df = eval_df.sort_values(KEY + ["dt"]).reset_index(drop=True)
eval_df["h"] = eval_df.groupby(KEY).cumcount() + 1

DL_RUNS = [("DLinear_censored_dataset_ftMS_sl62_ma28_pl7_dlin_censored_0", "dlinear_50k_raw"),
           ("DLinear_recovered_dataset_ftMS_sl62_ma28_pl7_dlin_recovered_fix_0",
            "dlinear_50krec_raw")]
for folder, tag in DL_RUNS:
    npy = DL_DIR / folder / "real_prediction.npy"
    ref = DL_DIR / folder / "metrics_res.parquet"
    if not npy.exists():
        print("missing:", folder)
        continue
    pred = np.clip(np.load(npy).reshape(-1), 0, None)
    if len(pred) != len(eval_df):
        print(f"length mismatch {folder}: {len(pred)} vs {len(eval_df)}")
        continue
    # 검증: 원 리포 보고 WAPE(품절 0인 날만)와 대조
    mask = (eval_df["stock_hour6_22_cnt"] == 0).to_numpy()
    got = wape(eval_df["sale_amount"], pred, mask)
    want = float(pd.read_parquet(ref).query("group_type=='psd>=0'")["wape"].mean())
    ok = abs(got - want) < 0.02
    print(f"{tag}: mapped WAPE {got:.4f} vs repo {want:.4f} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  skipped (ordering assumption failed)")
        continue
    rows = eval_df[KEY + ["h"]].copy()
    rows["q0.5"] = pred
    save(rows, tag, f"DLinear from frn-50k-baseline ({folder})")
