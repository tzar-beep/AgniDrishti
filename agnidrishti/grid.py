"""Uttarakhand boundary download and 0.25 degree analysis grid."""

import io
import logging
import zipfile

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from agnidrishti import config
from agnidrishti.utils import http_get, setup_logging

log = logging.getLogger(__name__)

BOUNDARY_PATH = config.DATA_RAW / "uttarakhand_gadm41.geojson"
GRID_PATH = config.DATA_PROCESSED / "grid.parquet"


def download_boundary() -> gpd.GeoDataFrame:
    """Download GADM 4.1 India level-1 once and cache the Uttarakhand polygon."""
    if BOUNDARY_PATH.exists():
        return gpd.read_file(BOUNDARY_PATH)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    zip_path = config.DATA_RAW / "gadm41_IND_1.json.zip"
    if not zip_path.exists():
        log.info("Downloading GADM boundaries from %s", config.GADM_URL)
        zip_path.write_bytes(http_get(config.GADM_URL, timeout=600).content)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        states = gpd.read_file(io.BytesIO(zf.read(name)))
    state = states[states["NAME_1"] == config.STATE_NAME][["NAME_1", "geometry"]]
    if state.empty:
        raise ValueError(f"{config.STATE_NAME} not found in GADM file")
    state.to_file(BOUNDARY_PATH, driver="GeoJSON")
    log.info("Cached boundary to %s", BOUNDARY_PATH)
    return state


def build_grid(
    boundary: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float] = config.BBOX,
    res: float = config.GRID_RES,
) -> gpd.GeoDataFrame:
    """Build square cells over ``bbox`` and keep those whose centroid is in ``boundary``.

    ``cell_id`` is derived from the global grid index (lat/lon offset from -90/-180), so it
    stays stable if the bounding box changes.
    """
    poly = boundary.union_all()
    rows = []
    for lat0 in np.arange(bbox[1], bbox[3], res):
        for lon0 in np.arange(bbox[0], bbox[2], res):
            cell = box(lon0, lat0, lon0 + res, lat0 + res)
            if poly.contains(cell.centroid):
                lat_i = round((lat0 + 90) / res)
                lon_i = round((lon0 + 180) / res)
                rows.append({
                    "cell_id": f"r{lat_i}c{lon_i}",
                    "lat": round(lat0 + res / 2, 4),
                    "lon": round(lon0 + res / 2, 4),
                    "geometry": cell,
                })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def load_grid() -> gpd.GeoDataFrame:
    """Load the processed grid, building it first if needed."""
    if not GRID_PATH.exists():
        main()
    return gpd.read_parquet(GRID_PATH)


def main() -> None:
    """CLI entry point: build and save the grid."""
    setup_logging()
    grid = build_grid(download_boundary())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(GRID_PATH)
    log.info("Saved %d cells to %s", len(grid), GRID_PATH)


if __name__ == "__main__":
    main()
