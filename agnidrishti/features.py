"""Leakage-safe weekly features per cell. CLI: ``python -m agnidrishti.features``.

Features for forecast date ``week_start`` use only data dated strictly before it
(daily weather up to ``week_start - 1 day``, hotspots from earlier weeks).
"""

import logging

import numpy as np
import pandas as pd

from agnidrishti import config
from agnidrishti.labels import make_labels, week_starts
from agnidrishti.utils import setup_logging

log = logging.getLogger(__name__)

DATASET_PATH = config.DATA_PROCESSED / "dataset.parquet"
DRY_DAYS_CAP = 365
FIRE_DAYS_CAP = 730
HIST_WEEKS = (4, 12, 52)

FEATURES = [
    *(f"{v}_mean_{w}d" for v in ("tmax", "rh", "wind", "precip") for w in (7, 14)),
    "days_since_rain",
    "fwi_7d",
    "elev_mean",
    "slope_mean",
    *(f"hotspots_{k}w" for k in HIST_WEEKS),
    "days_since_fire",
    "woy_sin",
    "woy_cos",
]


def fosberg_fwi(tmax_c: pd.Series, rh: pd.Series, wind_kmh: pd.Series) -> pd.Series:
    """Fosberg Fire Weather Index (0-100) from temperature, relative humidity and wind."""
    t = tmax_c * 9 / 5 + 32
    h = rh.clip(0, 100)
    m = np.where(
        h < 10,
        0.03229 + 0.281073 * h - 0.000578 * h * t,
        np.where(
            h < 50,
            2.22749 + 0.160107 * h - 0.01478 * t,
            21.0606 + 0.005565 * h**2 - 0.00035 * h * t - 0.483199 * h,
        ),
    )
    x = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * x + 1.5 * x**2 - 0.5 * x**3
    u = wind_kmh * 0.621371
    return pd.Series(np.clip(eta * np.sqrt(1 + u**2) / 0.3002, 0, 100), index=tmax_c.index)


def daily_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Per-cell daily rolling weather features; each row uses only that day and earlier."""
    w = weather.sort_values(["cell_id", "date"]).reset_index(drop=True)
    g = w.groupby("cell_id")
    for var in ("tmax", "rh", "wind", "precip"):
        for win in (7, 14):
            w[f"{var}_mean_{win}d"] = g[var].transform(
                lambda s, n=win: s.rolling(n, min_periods=1).mean()
            )
    rain_day = w["date"].where(w["precip"] >= config.RAIN_THRESHOLD_MM)
    w["days_since_rain"] = (w["date"] - rain_day.groupby(w["cell_id"]).ffill()).dt.days
    # Simplified FWI: Fosberg index damped by rain in the last 3 days.
    rain3 = g["precip"].transform(lambda s: s.rolling(3, min_periods=1).sum()).fillna(0)
    w["fwi"] = fosberg_fwi(w["tmax"], w["rh"], w["wind"]) * np.exp(-0.1 * rain3)
    w["fwi_7d"] = w.groupby("cell_id")["fwi"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    keep = [c for c in FEATURES if c in w.columns]
    return w[["cell_id", "date", *keep]]


def hotspot_features(
    fires: pd.DataFrame, cell_ids: list[str], weeks: pd.DatetimeIndex
) -> pd.DataFrame:
    """Past hotspot counts (4/12/52 weeks before ``week_start``) and days since last hotspot."""
    f = fires[["cell_id", "acq_date"]].copy()
    f["acq_date"] = pd.to_datetime(f["acq_date"]).dt.normalize()
    f["wk"] = f["acq_date"] - pd.to_timedelta(f["acq_date"].dt.weekday, unit="D")
    first = min(weeks.min(), f["wk"].min()) if len(f) else weeks.min()
    all_weeks = pd.date_range(first, weeks.max(), freq="W-MON")
    counts = (
        f.groupby(["wk", "cell_id"]).size().unstack(fill_value=0)
        .reindex(index=all_weeks, columns=cell_ids, fill_value=0)
    )
    out = pd.MultiIndex.from_product([cell_ids, weeks], names=["cell_id", "week_start"])
    out = out.to_frame(index=False)
    for k in HIST_WEEKS:
        # shift(1): week_start's own week is excluded (it is the label window).
        hist = counts.rolling(k, min_periods=1).sum().shift(1).fillna(0)
        long = hist.stack().rename(f"hotspots_{k}w").rename_axis(["week_start", "cell_id"])
        out = out.merge(long.reset_index(), on=["cell_id", "week_start"], how="left")

    left = out[["cell_id", "week_start"]].assign(asof=lambda d: d["week_start"] - pd.Timedelta(days=1))
    right = f[["cell_id", "acq_date"]].drop_duplicates().sort_values("acq_date")
    last = pd.merge_asof(
        left.sort_values("asof"), right, left_on="asof", right_on="acq_date",
        by="cell_id", direction="backward",
    )
    last["days_since_fire"] = (last["week_start"] - last["acq_date"]).dt.days
    out = out.merge(last[["cell_id", "week_start", "days_since_fire"]], on=["cell_id", "week_start"])
    out["days_since_fire"] = out["days_since_fire"].fillna(FIRE_DAYS_CAP).clip(upper=FIRE_DAYS_CAP)
    return out


def build_features(
    weather: pd.DataFrame,
    fires: pd.DataFrame,
    terrain: pd.DataFrame,
    weeks: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble the feature table with one row per (cell_id, week_start)."""
    cell_ids = sorted(terrain["cell_id"])
    daily = daily_weather_features(weather).rename(columns={"date": "asof"})
    base = pd.MultiIndex.from_product([cell_ids, weeks], names=["cell_id", "week_start"])
    df = base.to_frame(index=False)
    df["asof"] = df["week_start"] - pd.Timedelta(days=1)
    df = df.merge(daily, on=["cell_id", "asof"], how="left").drop(columns="asof")
    df["days_since_rain"] = df["days_since_rain"].fillna(DRY_DAYS_CAP).clip(upper=DRY_DAYS_CAP)
    df = df.merge(terrain, on="cell_id", how="left")
    df = df.merge(hotspot_features(fires, cell_ids, weeks), on=["cell_id", "week_start"])
    woy = df["week_start"].dt.isocalendar().week.astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    return df[["cell_id", "week_start", *FEATURES]]


def main() -> None:
    """CLI entry point: build features + labels and save ``dataset.parquet``."""
    setup_logging()
    p = config.DATA_PROCESSED
    weather = pd.read_parquet(p / "weather.parquet")
    fires = pd.read_parquet(p / "firms.parquet")
    terrain = pd.read_parquet(p / "terrain.parquet")
    weeks = week_starts(f"{config.START_YEAR}-01-01", f"{config.END_YEAR}-12-31")
    feats = build_features(weather, fires, terrain, weeks)
    labels = make_labels(fires, sorted(terrain["cell_id"]), weeks)
    ds = feats.merge(labels, on=["cell_id", "week_start"])
    n_nan = ds[FEATURES].isna().sum()
    if n_nan.any():
        log.warning("NaN feature values (handled at training time):\n%s", n_nan[n_nan > 0])
    ds.to_parquet(DATASET_PATH, index=False)
    log.info("Saved %d rows x %d features to %s (positive rate %.4f)",
             len(ds), len(FEATURES), DATASET_PATH, ds["y"].mean())


if __name__ == "__main__":
    main()
