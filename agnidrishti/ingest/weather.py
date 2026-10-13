"""Open-Meteo historical daily weather per grid-cell centroid, cached per cell-year."""

import json
import logging
import time

import pandas as pd

from agnidrishti import config
from agnidrishti.grid import load_grid
from agnidrishti.utils import http_get

log = logging.getLogger(__name__)

API = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = config.DATA_RAW / "weather"
OUT_PATH = config.DATA_PROCESSED / "weather.parquet"
VARIABLES = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "relative_humidity_2m_mean": "rh",
    "wind_speed_10m_max": "wind",
    "precipitation_sum": "precip",
}


def fetch_cell_year(cell_id: str, lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch (or load from cache) one year of daily weather for one cell centroid."""
    path = CACHE_DIR / f"{cell_id}_{year}.json"
    if not path.exists():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": ",".join(VARIABLES),
            "timezone": "Asia/Kolkata",
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(http_get(API, params).text)
        tmp.replace(path)
        time.sleep(1.0)  # be polite to the free API
    daily = json.loads(path.read_text())["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date", **VARIABLES})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "cell_id", cell_id)
    return df


def run() -> pd.DataFrame:
    """Download weather for all cells and years and save ``data/processed/weather.parquet``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    frames = []
    for i, cell in enumerate(grid.itertuples(), 1):
        for year in range(config.START_YEAR, config.END_YEAR + 1):
            frames.append(fetch_cell_year(cell.cell_id, cell.lat, cell.lon, year))
        log.info("Weather cell %d/%d (%s)", i, len(grid), cell.cell_id)
    df = pd.concat(frames, ignore_index=True)
    nans = df[list(VARIABLES.values())].isna().sum()
    if nans.any():
        log.warning("Missing weather values (kept as NaN, filled in features):\n%s", nans[nans > 0])
    df.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d cell-days to %s", len(df), OUT_PATH)
    return df
