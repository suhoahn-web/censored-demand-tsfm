"""R1 (second pass): hour-profile demand path.

The *uplift* path scales a censored day's sales by 16/(16-s), i.e. it assumes a
flat hourly demand profile over the 16 trading hours (06:00-22:00). Stockouts
in this data are mostly terminal (they start in the afternoon or evening and
last until closing), and the hourly profile is far from flat, so the flat
scaling tends to over-state the lost demand. The *profile* path replaces the
flat assumption by the series' own hourly profile, estimated on uncensored
days of the 90 days before the forecast origin (training data only):

    p_i(h) = share of series i's trading-hour sales falling in hour h,
             h = 6..21, over uncensored days of the 90-day window;
    demand estimate for a censored day
           = sales outside trading hours
             + (sales in in-stock trading hours) / sum_{h in-stock} p_i(h).

Fallbacks: a series with fewer than MIN_DAYS uncensored days (or zero sales)
in the window takes the pooled profile of the same product across stores, and
failing that the global profile. Whole-day stockouts (no in-stock trading
hour) are replaced by the series' uncensored mean and the estimate is capped
at three times that mean, exactly as for the uplift path, and never falls
below recorded sales.

The raw parquet files carry the 24-hour sales and stock-status vectors; the
processed daily files do not, so this module reads the raw files directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = {"lt": PROJECT_ROOT / "data/raw/FreshRetailNet-LT/data",
       "frn50k": PROJECT_ROOT / "data/raw/FreshRetailNet-50K/data"}
KEY = ["store_id", "product_id"]
OPEN, CLOSE = 6, 22
WINDOW_DAYS = 90
MIN_DAYS = 14
UPLIFT_CAP = 3.0
HCOLS = KEY + ["dt", "sale_amount", "hours_sale", "hours_stock_status",
               "stock_hour6_22_cnt"]


def _read_raw(dataset: str, split: str, start: str | None, end_excl: str | None,
              columns=HCOLS) -> pd.DataFrame:
    filters = []
    if start is not None:
        filters.append(("dt", ">=", start))
    if end_excl is not None:
        filters.append(("dt", "<", end_excl))
    return pd.read_parquet(RAW[dataset] / f"{split}.parquet", columns=columns,
                           filters=filters or None)


def hourly_profiles(dataset: str, end_excl: str, window: int = WINDOW_DAYS,
                    min_days: int = MIN_DAYS) -> tuple[pd.DataFrame, dict]:
    """Series-level hourly profile over the trading hours (16 columns, sums to 1).

    Estimated on uncensored days (no stockout hour between 06:00 and 22:00) in
    the `window` days before `end_excl` (exclusive), training data only.
    Returns the profile table (indexed by series) and a dict with the fallback
    counts."""
    start = (pd.Timestamp(end_excl) - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    tr = _read_raw(dataset, "train", start, end_excl)
    tr = tr[tr.stock_hour6_22_cnt == 0]
    hs = np.stack(tr["hours_sale"].to_numpy())[:, OPEN:CLOSE].astype(float)
    tot = pd.DataFrame(hs, columns=range(OPEN, CLOSE))
    tot[KEY] = tr[KEY].to_numpy()
    tot["n_days"] = 1
    by_series = tot.groupby(KEY).sum()
    by_product = tot.drop(columns="store_id").groupby("product_id").sum()
    g = tot.drop(columns=KEY + ["n_days"]).sum()
    global_p = (g / g.sum()).to_numpy()

    hours = list(range(OPEN, CLOSE))
    s_mass = by_series[hours].sum(axis=1)
    ok_s = (by_series["n_days"] >= min_days) & (s_mass > 0)
    p_series = by_series[hours].div(s_mass.where(s_mass > 0, 1.0), axis=0)
    p_mass = by_product[hours].sum(axis=1)
    ok_p = (by_product["n_days"] >= min_days) & (p_mass > 0)
    p_product = by_product[hours].div(p_mass.where(p_mass > 0, 1.0), axis=0)
    return (p_series, ok_s, p_product, ok_p, global_p), {
        "window_start": start, "window_end_excl": end_excl,
        "uncensored_days": int(len(tr)), "series_with_profile": int(ok_s.sum()),
        "series_total": int(len(by_series))}


def _lookup(profiles, keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Profile matrix (rows × 16) for the given series keys with fallbacks.
    Returns (P, level) where level is 0=series, 1=product, 2=global."""
    p_series, ok_s, p_product, ok_p, global_p = profiles
    n = len(keys)
    P = np.tile(global_p, (n, 1))
    level = np.full(n, 2, dtype=int)
    idx = pd.MultiIndex.from_frame(keys[KEY])
    ok_series = ok_s.reindex(idx, fill_value=False).to_numpy().astype(bool)
    P[ok_series] = p_series.reindex(idx[ok_series]).to_numpy()
    level[ok_series] = 0
    ok_prod = ok_p.reindex(keys["product_id"], fill_value=False).to_numpy().astype(bool)
    m = ~ok_series & ok_prod
    P[m] = p_product.reindex(keys.loc[m, "product_id"]).to_numpy()
    level[m] = 1
    return P, level


def profile_demand(rows: pd.DataFrame, profiles, clean_mean: np.ndarray,
                   cap: float = UPLIFT_CAP) -> tuple[np.ndarray, dict]:
    """Per-row demand estimate under the hour-profile path.

    `rows` must carry KEY, sale_amount, hours_sale, hours_stock_status,
    stock_hour6_22_cnt (one row per series-day); `clean_mean` is the series'
    uncensored-day mean aligned with `rows` (used for whole-day stockouts and
    the cap)."""
    hs = np.stack(rows["hours_sale"].to_numpy()).astype(float)
    st = np.stack(rows["hours_stock_status"].to_numpy()).astype(int)
    y = rows["sale_amount"].to_numpy().astype(float)
    P, level = _lookup(profiles, rows[KEY + []].reset_index(drop=True))
    instock = (st[:, OPEN:CLOSE] == 0)
    y_trade = hs[:, OPEN:CLOSE]
    y_out = y - y_trade.sum(axis=1)
    y_in = (y_trade * instock).sum(axis=1)
    mass = (P * instock).sum(axis=1)
    cens = rows["stock_hour6_22_cnt"].to_numpy() > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.where(mass > 0, y_out + y_in / np.maximum(mass, 1e-9), clean_mean)
    est = np.minimum(est, np.maximum(cap * clean_mean, y))
    est = np.maximum(est, y)
    est = np.where(cens, est, y)   # uncensored days: recorded sales
    info = {"rows": int(len(rows)), "censored_rows": int(cens.sum()),
            "level_series": int((level == 0).sum()), "level_product": int((level == 1).sum()),
            "level_global": int((level == 2).sum()),
            "whole_day_stockouts": int((cens & (mass <= 0)).sum()),
            "capped": int((cens & (est >= np.maximum(cap * clean_mean, y) - 1e-12) & (mass > 0)).sum())}
    return est, info


def clean_mean_of(t: pd.DataFrame) -> pd.Series:
    """Series' mean sales over uncensored days (fallback: all days), aligned to t."""
    cm = (t[t.stock_hour6_22_cnt == 0].groupby(KEY)["sale_amount"].mean()
          .rename("clean_mean"))
    out = t.join(cm, on=KEY)["clean_mean"]
    return out.fillna(t.groupby(KEY)["sale_amount"].transform("mean"))


def profile_path_eval(dataset: str) -> tuple[pd.DataFrame, dict]:
    """Hour-profile demand for the official evaluation week (wide: series × h)."""
    ev = _read_raw(dataset, "eval", None, None)
    ev = ev.sort_values(KEY + ["dt"]).reset_index(drop=True)
    ev["h"] = ev.groupby(KEY).cumcount() + 1
    profiles, meta = hourly_profiles(dataset, ev["dt"].min())
    est, info = profile_demand(ev, profiles, clean_mean_of(ev).to_numpy())
    ev["profile"] = est
    wide = ev.pivot_table(index=KEY, columns="h", values="profile").sort_index()
    meta.update(info)
    meta["total_observed"] = float(ev["sale_amount"].sum())
    meta["total_profile"] = float(est.sum())
    return wide, meta


def profile_path_origin(dataset: str, cutoff: str, horizon: int = 7
                        ) -> tuple[pd.DataFrame, dict]:
    """Hour-profile demand for the 7 days after a training-period origin."""
    cut = pd.Timestamp(cutoff)
    start = (cut + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_excl = (cut + pd.Timedelta(days=horizon + 1)).strftime("%Y-%m-%d")
    t = _read_raw(dataset, "train", start, end_excl)
    t = t.sort_values(KEY + ["dt"]).reset_index(drop=True)
    t["h"] = t.groupby(KEY).cumcount() + 1
    t = t[t.groupby(KEY)["h"].transform("size") == horizon].reset_index(drop=True)
    profiles, meta = hourly_profiles(dataset, start)
    est, info = profile_demand(t, profiles, clean_mean_of(t).to_numpy())
    t["profile"] = est
    wide = t.pivot_table(index=KEY, columns="h", values="profile").sort_index()
    meta.update(info)
    return wide, meta
