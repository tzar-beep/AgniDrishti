"""NASA FIRMS VIIRS active-fire archive download, filtering and cell assignment."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd

from agnidrishti import config
from agnidrishti.grid import download_boundary, load_grid
from agnidrishti.utils import FetchError, http_get

log = logging.getLogger(__name__)

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CACHE_DIR = config.DATA_RAW / "firms"
OUT_PATH = config.DATA_PROCESSED / "firms.parquet"
WORKERS = 4  # parallel requests; well under the FIRMS 5000-per-10-minute key limit


def get_map_key() -> str:
    """Return the FIRMS MAP_KEY from the environment or raise a clear error."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not set. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put it in .env "
            "(FIRMS_MAP_KEY=...) or your environment."
        )
    return key


def fetch_chunk(key: str, start: date) -> pd.DataFrame:
    """Fetch (or load from cache) FIRMS detections for DAY_RANGE days from ``start``."""
    path = CACHE_DIR / f"{config.FIRMS_SOURCE}_{start.isoformat()}.csv"
    if not path.exists():
        w, s, e, n = config.BBOX
        url = f"{API}/{key}/{config.FIRMS_SOURCE}/{w},{s},{e},{n}/{config.FIRMS_DAY_RANGE}/{start}"
        text = http_get(url).text
        if not text.startswith("latitude"):
            raise FetchError(f"Unexpected FIRMS response for {start}: {text[:200]!r}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)  # atomic, so an interrupted run never leaves a partial cache file
    return pd.read_csv(path, dtype={"confidence": str})


def download(start_year: int = config.START_YEAR, end_year: int = config.END_YEAR) -> pd.DataFrame:
    """Download all chunks for the configured years and return raw detections."""
    key = get_map_key()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    first, end = date(start_year, 1, 1), date(end_year, 12, 31)
    n_chunks = math.ceil(((end - first).days + 1) / config.FIRMS_DAY_RANGE)
    starts = [first + timedelta(days=config.FIRMS_DAY_RANGE * i) for i in range(n_chunks)]
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, df in enumerate(pool.map(lambda d: fetch_chunk(key, d), starts), 1):
            frames.append(df)
            if i % 50 == 0 or i == n_chunks:
                log.info("FIRMS chunks %d/%d", i, n_chunks)
    raw = pd.concat(frames, ignore_index=True)
    raw["acq_date"] = pd.to_datetime(raw["acq_date"])
    return raw[(raw["acq_date"].dt.year >= start_year) & (raw["acq_date"].dt.year <= end_year)]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop low-confidence detections, clip to the state polygon and assign ``cell_id``."""
    n0 = len(raw)
    df = raw[raw["confidence"].isin(config.FIRMS_KEEP_CONFIDENCE)]
    log.info("Dropped %d low-confidence detections of %d", n0 - len(df), n0)

    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    inside = pts.within(download_boundary().union_all())
    log.info("Dropped %d detections outside the state polygon", int((~inside).sum()))
    pts = pts[inside]

    grid = load_grid()[["cell_id", "geometry"]]
    joined = gpd.sjoin(pts, grid, how="left", predicate="within")
    missing = joined["cell_id"].isna().sum()
    if missing:
        log.warning("Dropped %d in-state detections not inside any kept grid cell", missing)
    joined = joined.dropna(subset=["cell_id"])
    cols = ["cell_id", "latitude", "longitude", "acq_date", "confidence", "frp"]
    return pd.DataFrame(joined[cols]).reset_index(drop=True)


def run() -> pd.DataFrame:
    """Download, clean and save FIRMS detections to ``data/processed/firms.parquet``."""
    df = clean(download())
    df.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d detections to %s", len(df), OUT_PATH)
    return df
