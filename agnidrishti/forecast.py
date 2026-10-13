"""Live 7-day outlook. CLI: ``python -m agnidrishti.forecast [--week-start YYYY-MM-DD]``.

Builds features for the next forecast week from the Open-Meteo *forecast* API (past 92 days
plus up to 7 forecast days) and recent FIRMS detections (standard-processing archive where
available, near-real-time after that), then scores them with the trained model.
"""

import argparse
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta

import joblib
import pandas as pd

from agnidrishti import config
from agnidrishti.alerts import assign_levels
from agnidrishti.explain import top_k_contributions
from agnidrishti.features import FEATURES, build_features
from agnidrishti.grid import load_grid
from agnidrishti.ingest import firms
from agnidrishti.ingest.weather import VARIABLES
from agnidrishti.utils import http_get, setup_logging

log = logging.getLogger(__name__)

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
AVAIL_API = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"
NRT_SOURCE = "VIIRS_SNPP_NRT"
PRED_PATH = config.DATA_PROCESSED / "forecast_predictions.parquet"
SHAP_PATH = config.DATA_PROCESSED / "forecast_shap_top5.parquet"
FEAT_PATH = config.DATA_PROCESSED / "forecast_features.parquet"
PAST_DAYS = 92  # Open-Meteo maximum
HISTORY_DAYS = 371  # enough for the 52-week hotspot count


def next_week_start(today: date) -> pd.Timestamp:
    """The next Monday on or after ``today``."""
    return pd.Timestamp(today + timedelta(days=(7 - today.weekday()) % 7))


def fetch_weather(issue: date) -> pd.DataFrame:
    """Daily weather per cell: the last 92 days plus 8 forecast days, cached per issue date."""
    cache = config.DATA_RAW / "forecast" / issue.isoformat() / "weather"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for cell in load_grid().itertuples():
        path = cache / f"{cell.cell_id}.json"
        if not path.exists():
            params = {"latitude": cell.lat, "longitude": cell.lon, "daily": ",".join(VARIABLES),
                      "past_days": PAST_DAYS, "forecast_days": 8, "timezone": "Asia/Kolkata"}
            path.write_text(http_get(FORECAST_API, params).text)
        daily = json.loads(path.read_text())["daily"]
        df = pd.DataFrame(daily).rename(columns={"time": "date", **VARIABLES})
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df.assign(cell_id=cell.cell_id))
    return pd.concat(frames, ignore_index=True)


def sp_last_date(key: str) -> date:
    """Last date covered by the FIRMS standard-processing archive."""
    text = http_get(f"{AVAIL_API}/{key}/{config.FIRMS_SOURCE}").text
    avail = pd.read_csv(io.StringIO(text))
    return pd.Timestamp(avail["max_date"].iloc[0]).date()


def fetch_recent_fires(key: str, start: date, end: date, issue: date) -> pd.DataFrame:
    """Detections from ``start`` to ``end``: SP archive chunks, then NRT for the rest."""
    sp_end = sp_last_date(key)
    log.info("FIRMS SP archive available to %s; using %s after that", sp_end, NRT_SOURCE)
    cache = config.DATA_RAW / "forecast" / issue.isoformat() / "firms"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    day = start
    while day <= end:
        last = day + timedelta(days=config.FIRMS_DAY_RANGE - 1)
        if last <= sp_end:  # complete SP chunks can reuse the training cache
            jobs.append((day, config.FIRMS_SOURCE, firms.CACHE_DIR))
        else:
            jobs.append((day, NRT_SOURCE, cache))
        day += timedelta(days=config.FIRMS_DAY_RANGE)
    firms.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=firms.WORKERS) as pool:
        frames = list(pool.map(lambda j: firms.fetch_chunk(key, j[0], j[1], j[2]), jobs))
    raw = pd.concat([f for f in frames if not f.empty] or frames[:1], ignore_index=True)
    raw["acq_date"] = pd.to_datetime(raw["acq_date"])
    raw = raw[(raw["acq_date"].dt.date >= start) & (raw["acq_date"].dt.date <= end)]
    return firms.clean(raw)


def run(week_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Score the outlook week and save forecast predictions, SHAP and features."""
    key = firms.get_map_key()
    issue = datetime.now(UTC).date()
    ws = week_start if week_start is not None else next_week_start(issue)
    log.info("Issuing outlook on %s for week starting %s", issue, ws.date())
    sidecar = json.loads((config.MODELS_DIR / f"{config.MODEL_NAME}.json").read_text())
    model = joblib.load(config.MODELS_DIR / f"{config.MODEL_NAME}.joblib")
    terrain = pd.read_parquet(config.DATA_PROCESSED / "terrain.parquet")

    weather = fetch_weather(issue)
    weather = weather[weather["date"] < ws]
    n_fcst = (weather["date"].dt.date >= issue).sum() // weather["cell_id"].nunique()
    log.info("Weather: %d observed + %d forecast days per cell", PAST_DAYS, n_fcst)
    fire_end = min(issue, (ws - pd.Timedelta(days=1)).date())
    fires = fetch_recent_fires(key, (ws - pd.Timedelta(days=HISTORY_DAYS)).date(), fire_end, issue)
    log.info("Recent detections used: %d", len(fires))

    feats = build_features(weather, fires, terrain, pd.DatetimeIndex([ws]))
    feats["days_since_rain"] = feats["days_since_rain"].clip(upper=PAST_DAYS)
    probs = pd.Series(model.predict_proba(feats[FEATURES])[:, 1], index=feats.index)
    issued_at = datetime.now(UTC).isoformat(timespec="seconds")
    preds = feats[["cell_id", "week_start"]].assign(
        prob=probs, y=pd.array([pd.NA] * len(feats), dtype="Int64"),
        level=assign_levels(probs, sidecar["alert_thresholds"]), issued_at=issued_at,
    )
    preds.to_parquet(PRED_PATH, index=False)
    top_k_contributions(model, feats[FEATURES], feats[["cell_id", "week_start"]]).to_parquet(
        SHAP_PATH, index=False)
    feats.assign(y=pd.array([pd.NA] * len(feats), dtype="Int64")).to_parquet(FEAT_PATH, index=False)
    log.info("Outlook saved: mean P %.3f, levels %s", probs.mean(),
             preds["level"].value_counts().to_dict())
    return preds


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-start", type=pd.Timestamp, default=None)
    args = parser.parse_args()
    setup_logging()
    run(args.week_start)


if __name__ == "__main__":
    main()
