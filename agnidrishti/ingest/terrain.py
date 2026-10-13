"""Mean elevation and slope per cell from the Open-Meteo elevation API (Copernicus DEM 90 m)."""

import json
import logging

import numpy as np
import pandas as pd

from agnidrishti import config
from agnidrishti.grid import load_grid
from agnidrishti.utils import http_get

log = logging.getLogger(__name__)

API = "https://api.open-meteo.com/v1/elevation"
CACHE_DIR = config.DATA_RAW / "terrain"
OUT_PATH = config.DATA_PROCESSED / "terrain.parquet"
BATCH = 100  # max coordinates per request


def sample_points(lat: float, lon: float, n: int = config.TERRAIN_SAMPLES) -> np.ndarray:
    """Return an (n, n, 2) array of (lat, lon) sample points inside a cell centred at lat/lon."""
    offs = ((np.arange(n) + 0.5) / n - 0.5) * config.GRID_RES
    lats, lons = np.meshgrid(lat + offs, lon + offs, indexing="ij")
    return np.stack([lats, lons], axis=-1)


def fetch_elevations(cell_id: str, pts: np.ndarray) -> np.ndarray:
    """Fetch (or load from cache) elevations for the sample points of one cell."""
    path = CACHE_DIR / f"{cell_id}.json"
    if not path.exists():
        flat = pts.reshape(-1, 2)
        elev: list[float] = []
        for i in range(0, len(flat), BATCH):
            chunk = flat[i : i + BATCH]
            params = {
                "latitude": ",".join(f"{v:.5f}" for v in chunk[:, 0]),
                "longitude": ",".join(f"{v:.5f}" for v in chunk[:, 1]),
            }
            elev += http_get(API, params).json()["elevation"]
        path.write_text(json.dumps(elev))
    return np.array(json.loads(path.read_text()), dtype=float).reshape(pts.shape[:2])


def slope_degrees(z: np.ndarray, lat: float, spacing_deg: float) -> np.ndarray:
    """Slope in degrees from an elevation grid with spacing given in degrees."""
    dy = spacing_deg * 111_320
    dx = dy * np.cos(np.radians(lat))
    gy, gx = np.gradient(z, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def run() -> pd.DataFrame:
    """Compute elev_mean and slope_mean for all cells and save ``terrain.parquet``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    spacing = config.GRID_RES / config.TERRAIN_SAMPLES
    rows = []
    for cell in load_grid().itertuples():
        z = fetch_elevations(cell.cell_id, sample_points(cell.lat, cell.lon))
        rows.append({
            "cell_id": cell.cell_id,
            "elev_mean": float(np.nanmean(z)),
            "slope_mean": float(np.nanmean(slope_degrees(z, cell.lat, spacing))),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Saved terrain for %d cells to %s", len(df), OUT_PATH)
    return df
