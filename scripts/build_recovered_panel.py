"""복원수요(demand.parquet) → frn50krec daily 패널 생성 (lab06 실행용).

train: sale_amount를 복원치(sale_amount_pred)로 교체한 50K 일 단위 패널.
eval: 원 50K eval 그대로 (평가는 항상 관측 판매 기준 — 절대 규칙과 원 논문 프로토콜 유지).
"""

from pathlib import Path

import pandas as pd

HOME = Path.home()
DEMAND = HOME / "tsfm-demand/external/frn-50k-baseline/latent_demand_recovery/exp/demand/demand.parquet"
PROCESSED = HOME / "tsfm-demand/data/processed"

COLS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt",
        "discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature"]

df = pd.read_parquet(DEMAND)
df["sale_amount"] = df["sale_amount_pred"]
df["dt"] = pd.to_datetime(df["dt"])
df = df[COLS].sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
df.to_parquet(PROCESSED / "frn50krec_daily_train.parquet", index=False)

ev = pd.read_parquet(PROCESSED / "frn50k_daily_eval.parquet")
ev.to_parquet(PROCESSED / "frn50krec_daily_eval.parquet", index=False)
print("recovered panel:", df.shape, "| eval copied:", ev.shape)
