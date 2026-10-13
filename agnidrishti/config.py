"""Project-wide configuration: paths, study area, time windows and thresholds."""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"

# Study area. Bounding box is (min_lon, min_lat, max_lon, max_lat), aligned to the grid.
STATE_NAME = "Uttarakhand"
BBOX = (77.5, 28.5, 81.25, 31.5)
GRID_RES = 0.25
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json.zip"

# Time windows (inclusive years).
START_YEAR = 2015
END_YEAR = 2024
TRAIN_YEARS = (2015, 2021)
VAL_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2024)

# FIRMS: VIIRS S-NPP standard-processing archive. Low-confidence ("l") detections are dropped,
# and so are static land sources such as industrial flares (VIIRS "type" 2) and offshore (3).
FIRMS_SOURCE = "VIIRS_SNPP_SP"
FIRMS_KEEP_CONFIDENCE = ("n", "h")
FIRMS_KEEP_TYPES = (0,)  # presumed vegetation fire
FIRMS_DAY_RANGE = 5  # max days per FIRMS area request

# Labels / features.
HORIZON_DAYS = 7
RAIN_THRESHOLD_MM = 2.0

# Terrain: points sampled per cell side for elevation/slope.
TERRAIN_SAMPLES = 11

MODEL_NAME = "xgb_v1"
