"""평가지표: WAPE, MASE, WPE, WQL, CRPS(분위수 근사).

MAPE 금지 (CLAUDE.md 절대 규칙 2; Kolassa 2016 — 실측 0에서 미정의 + (-1)-중앙값 최적 편향).
WAPE/MASE의 중앙값-최적 한계는 WPE·WQL/CRPS 병행으로 보완 (설계서 v2.0 7절).
"""

import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Σ|y-ŷ| / Σy. 집계 수준 절대오차."""
    return float(np.abs(y - yhat).sum() / y.sum())


def wpe(y: np.ndarray, yhat: np.ndarray) -> float:
    """Σ(ŷ-y) / Σy. 음수 = 과소예측(하향 편향) — 검열 가설의 핵심 지표."""
    return float((yhat - y).sum() / y.sum())


def mase(y: np.ndarray, yhat: np.ndarray, y_insample: np.ndarray, m: int = 7) -> float:
    """계절 naive(m) 스케일 MAE. 시계열 1개 단위로 호출."""
    scale = np.abs(y_insample[m:] - y_insample[:-m]).mean()
    if scale == 0:
        return np.nan
    return float(np.abs(y - yhat).mean() / scale)


def wql(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    """가중 분위수 손실 (M5 Uncertainty의 SPL 계보). q_preds: {분위수: 예측배열}."""
    losses = []
    for q, yhat_q in q_preds.items():
        e = y - yhat_q
        losses.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(2 * np.mean(losses) / np.abs(y).mean())


def crps_from_quantiles(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    """분위수 격자 기반 CRPS 근사 (pinball 평균의 2배; Gneiting & Ranjan 2011)."""
    qs = sorted(q_preds)
    pinballs = []
    for q in qs:
        e = y - q_preds[q]
        pinballs.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(2 * np.mean(pinballs))


def evaluate_forecast(pred: pd.DataFrame, truth: pd.DataFrame, insample: pd.DataFrame,
                      key: list[str] = ("store_id", "product_id"),
                      nostock_only: bool = False) -> dict:
    """예측 parquet(run_zeroshot 산출) × 정답 → 전체 지표 dict.

    pred: key + h + q* 열. truth: key + dt + sale_amount (h 순서 정렬 가정).
    insample: key + dt + sale_amount (MASE 스케일용 train 구간).
    nostock_only: 원 논문 공식 프로토콜 — 품절 시간 0인 타깃 날만 채점
      (frn-50k-baseline predictTFT.py: `stock_hour6_22_cnt == 0` 필터).
    """
    key = list(key)
    truth = truth.sort_values(key + ["dt"]).reset_index(drop=True)
    truth["h"] = truth.groupby(key).cumcount() + 1
    cols = key + ["h", "sale_amount"]
    if "stock_hour6_22_cnt" in truth.columns:
        cols.append("stock_hour6_22_cnt")
    merged = pred.merge(truth[cols], on=key + ["h"], how="inner")
    if nostock_only:
        merged = merged[merged["stock_hour6_22_cnt"] == 0]
    y = merged["sale_amount"].to_numpy()
    q_cols = {float(c[1:]): merged[c].to_numpy() for c in merged.columns if c.startswith("q")}
    yhat = q_cols[0.5]

    mase_vals = []
    ins_g = dict(list(insample.sort_values(key + ["dt"]).groupby(key)["sale_amount"]))
    for k, g in merged.groupby(key):
        if k in ins_g:
            mase_vals.append(mase(g["sale_amount"].to_numpy(), g["q0.5"].to_numpy(),
                                  ins_g[k].to_numpy()))
    return {
        "wape": wape(y, yhat),
        "wpe": wpe(y, yhat),
        "mase_mean": float(np.nanmean(mase_vals)),
        "wql": wql(y, q_cols),
        "crps": crps_from_quantiles(y, q_cols),
        "n_series": merged.groupby(key).ngroups,
    }
