"""FreshRetailNet-50K 로딩 유틸리티.

원본 parquet(행 = store×product×일, 24시간 배열 포함)을 읽어
분석용 long/wide 포맷으로 변환한다. 원본은 절대 수정하지 않는다.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "FreshRetailNet-50K" / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SERIES_KEY = ["store_id", "product_id"]

# 원 논문 프로토콜: train 90일(2024-03-28~06-25), eval 7일(2024-06-26~07-02)
TRAIN_START, TRAIN_END = "2024-03-28", "2024-06-25"
EVAL_START, EVAL_END = "2024-06-26", "2024-07-02"


def load_split(split: str = "train", columns: list[str] | None = None) -> pd.DataFrame:
    """지정 split의 원본 parquet을 읽는다. columns로 필요한 열만 제한 가능."""
    path = RAW_DIR / f"{split}.parquet"
    df = pd.read_parquet(path, columns=columns)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


def daily_panel(split: str = "train") -> pd.DataFrame:
    """일 단위 패널: series key + dt + 판매량/품절시간수/공변량 (시간 배열 제외)."""
    cols = SERIES_KEY + [
        "city_id", "dt", "sale_amount", "stock_hour6_22_cnt",
        "discount", "holiday_flag", "activity_flag",
        "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
    ]
    return load_split(split, columns=cols)


def series_id(df: pd.DataFrame) -> pd.Series:
    """store_id-product_id 결합 문자열 ID."""
    return df["store_id"].astype(str) + "_" + df["product_id"].astype(str)
