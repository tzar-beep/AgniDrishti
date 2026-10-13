"""Uttarakhand boundary download and 0.25 degree analysis grid."""

import io
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from agnidrishti import config
from agnidrishti.utils import http_get, setup_logging

log = logging.getLogger(__name__)

BOUNDARY_PATH = config.DATA_RAW / "uttarakhand_gadm41.geojson"
DISTRICTS_PATH = config.DATA_RAW / "uttarakhand_districts_gadm41.geojson"
GRID_PATH = config.DATA_PROCESSED / "grid.parquet"


def _gadm_uttarakhand(level: int, out: Path) -> gpd.GeoDataFrame:
    """Download GADM 4.1 India at ``level`` once and cache the Uttarakhand features."""
    if out.exists():
        return gpd.read_file(out)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = config.GADM_URL.replace("_1.json", f"_{level}.json")
    zip_path = config.DATA_RAW / url.rsplit("/", 1)[1]
    if not zip_path.exists():
        log.info("Downloading GADM level %d from %s", level, url)
        zip_path.write_bytes(http_get(url, timeout=600).content)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        gdf = gpd.read_file(io.BytesIO(zf.read(name)))
    cols = ["NAME_1", "geometry"] if level == 1 else ["NAME_1", "NAME_2", "geometry"]
    sub = gdf[gdf["NAME_1"] == config.STATE_NAME][cols]
    if sub.empty:
        raise ValueError(f"{config.STATE_NAME} not found in GADM level {level}")
    sub.to_file(out, driver="GeoJSON")
    log.info("Cached %s", out)
    return sub


def download_boundary() -> gpd.GeoDataFrame:
    """Uttarakhand state polygon (GADM level 1)."""
    return _gadm_uttarakhand(1, BOUNDARY_PATH)


def download_districts() -> gpd.GeoDataFrame:
    """Uttarakhand district polygons (GADM level 2, column ``NAME_2``)."""
    return _gadm_uttarakhand(2, DISTRICTS_PATH)


def assign_districts(grid: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a ``district`` column: the district containing each cell centroid (nearest if none)."""
    utm = grid.to_crs(32644)
    pts = gpd.GeoDataFrame(utm[["cell_id"]], geometry=utm.geometry.centroid, crs=utm.crs)
    joined = gpd.sjoin_nearest(pts, districts.to_crs(32644)[["NAME_2", "geometry"]])
    names = joined.drop_duplicates("cell_id").set_index("cell_id")["NAME_2"]
    # GADM writes multi-word names without spaces ("TehriGarhwal"); "Garhwal" is Pauri Garhwal.
    names = names.str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True).replace("Garhwal", "Pauri Garhwal")
    return grid.assign(district=grid["cell_id"].map(names))


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
    grid = assign_districts(build_grid(download_boundary()), download_districts())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(GRID_PATH)
    log.info("Saved %d cells to %s", len(grid), GRID_PATH)


if __name__ == "__main__":
    main()
