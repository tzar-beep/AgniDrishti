"""API routes. All data comes from precomputed tables held in ``app.state.store``."""

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, Response
from shapely.geometry import mapping

from agnidrishti.alerts import ADVICE, LEVELS, assign_levels, district_bulletin, to_cap
from agnidrishti.api import schemas
from agnidrishti.features import FEATURES

router = APIRouter(prefix="/api")
HISTORY_WEEKS = 12
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"


@dataclass
class Store:
    """In-memory copies of the precomputed prediction, explanation and feature tables."""

    geometry: dict[str, dict]
    districts: dict[str, str]
    predictions: pd.DataFrame
    shap: pd.DataFrame
    dataset: pd.DataFrame
    model_info: dict
    dates: list[str]
    forecast_dates: list[str]

    @classmethod
    def load(cls, processed_dir: Path, model_json: Path) -> "Store":
        """Load hindcast tables, plus the live outlook (``forecast_*.parquet``) if present."""
        grid = gpd.read_parquet(processed_dir / "grid.parquet")
        model_info = json.loads(model_json.read_text())
        preds = pd.read_parquet(processed_dir / "predictions.parquet")
        shap = pd.read_parquet(processed_dir / "shap_top5.parquet")
        dataset = pd.read_parquet(processed_dir / "dataset.parquet")
        preds["y"] = preds["y"].astype("Int64")
        dataset["y"] = dataset["y"].astype("Int64")
        forecast_dates: list[str] = []
        fp = processed_dir / "forecast_predictions.parquet"
        if fp.exists():
            fpred = pd.read_parquet(fp)
            known = set(preds["week_start"])
            fpred = fpred[~fpred["week_start"].isin(known)]  # hindcast wins if both exist
            fdates = set(fpred["week_start"])
            forecast_dates = sorted(pd.DatetimeIndex(list(fdates)).strftime("%Y-%m-%d"))
            fshap = pd.read_parquet(processed_dir / "forecast_shap_top5.parquet")
            ffeat = pd.read_parquet(processed_dir / "forecast_features.parquet")
            preds = pd.concat([preds, fpred[["cell_id", "week_start", "y", "prob"]]], ignore_index=True)
            shap = pd.concat([shap, fshap[fshap["week_start"].isin(fdates)]], ignore_index=True)
            dataset = pd.concat([dataset, ffeat[ffeat["week_start"].isin(fdates)]], ignore_index=True)
        thresholds = model_info.get("alert_thresholds")
        preds["level"] = assign_levels(preds["prob"], thresholds) if thresholds else None
        return cls(
            geometry={r.cell_id: mapping(r.geometry) for r in grid.itertuples()},
            districts=dict(zip(grid["cell_id"], grid.get("district", grid["cell_id"]), strict=True)),
            predictions=preds,
            shap=shap,
            dataset=dataset,
            model_info=model_info,
            dates=sorted(preds["week_start"].dt.strftime("%Y-%m-%d").unique()),
            forecast_dates=forecast_dates,
        )


def _store(request: Request) -> Store:
    return request.app.state.store


def _check_date(store: Store, date: str) -> pd.Timestamp:
    if date not in store.dates:
        raise HTTPException(404, f"No predictions for date {date}; see /api/dates")
    return pd.Timestamp(date)


def _check_cell(store: Store, cell_id: str) -> None:
    if cell_id not in store.geometry:
        raise HTTPException(404, f"Unknown cell_id {cell_id}")


def _clean(v: float) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _obs(v) -> int | None:
    return None if pd.isna(v) else int(v)


@router.get("/health", response_model=schemas.Health)
def health(request: Request) -> schemas.Health:
    """Liveness check with basic table sizes."""
    s = _store(request)
    return schemas.Health(status="ok", n_cells=len(s.geometry), n_dates=len(s.dates))


@router.get("/dates", response_model=schemas.Dates)
def dates(request: Request) -> schemas.Dates:
    """All prediction dates (week starts); ``forecast_dates`` are live-outlook weeks."""
    s = _store(request)
    return schemas.Dates(dates=s.dates, forecast_dates=s.forecast_dates)


@router.get("/risk", response_model=schemas.RiskCollection)
def risk(request: Request, date: str = Query(..., pattern=DATE_RE)):
    """GeoJSON FeatureCollection of cell probabilities for one prediction date."""
    s = _store(request)
    ts = _check_date(s, date)
    day = s.predictions[s.predictions["week_start"] == ts]
    feats = [
        schemas.RiskFeature(
            geometry=s.geometry[r.cell_id],
            properties={"cell_id": r.cell_id, "prob": round(float(r.prob), 4),
                        "observed": _obs(r.y), "level": r.level,
                        "district": s.districts.get(r.cell_id)},
        )
        for r in day.itertuples()
    ]
    return schemas.RiskCollection(date=date, is_forecast=date in s.forecast_dates, features=feats)


def _bulletin(s: Store, date: str) -> list[dict]:
    if not s.model_info.get("alert_thresholds"):
        raise HTTPException(404, "Model has no alert thresholds; retrain to enable alerts")
    ts = _check_date(s, date)
    day = s.predictions[s.predictions["week_start"] == ts][["cell_id", "prob"]]
    day = day.assign(district=day["cell_id"].map(s.districts))
    return district_bulletin(day, s.model_info["alert_thresholds"])


@router.get("/alerts", response_model=schemas.Alerts)
def alerts(request: Request, date: str = Query(..., pattern=DATE_RE)) -> schemas.Alerts:
    """District alert bulletin (level, max probability, advice) for one prediction date."""
    s = _store(request)
    return schemas.Alerts(
        date=date, is_forecast=date in s.forecast_dates,
        thresholds=s.model_info["alert_thresholds"] if s.model_info.get("alert_thresholds") else {},
        districts=_bulletin(s, date), advice=ADVICE, levels=LEVELS,
    )


@router.get("/alerts/cap", response_class=Response)
def alerts_cap(request: Request, date: str = Query(..., pattern=DATE_RE)) -> Response:
    """The same bulletin as a CAP 1.2 XML alert (districts at High or Very High only)."""
    s = _store(request)
    xml = to_cap(date, _bulletin(s, date), is_forecast=date in s.forecast_dates)
    return Response(xml, media_type="application/cap+xml")


@router.get("/cells/{cell_id}/history", response_model=schemas.History)
def history(request: Request, cell_id: str, date: str | None = None) -> schemas.History:
    """Features, labels and predicted probability for the 12 weeks up to ``date``."""
    s = _store(request)
    _check_cell(s, cell_id)
    end = _check_date(s, date) if date else pd.Timestamp(s.dates[-1])
    start = end - pd.Timedelta(weeks=HISTORY_WEEKS)
    d = s.dataset[(s.dataset["cell_id"] == cell_id) & (s.dataset["week_start"] > start)
                  & (s.dataset["week_start"] <= end)].sort_values("week_start")
    probs = s.predictions[s.predictions["cell_id"] == cell_id].set_index("week_start")["prob"]
    rows = [
        schemas.HistoryRow(
            week_start=r["week_start"].strftime("%Y-%m-%d"),
            y=_obs(r["y"]),
            prob=_clean(probs.get(r["week_start"])),
            features={f: _clean(r[f]) for f in FEATURES},
        )
        for _, r in d.iterrows()
    ]
    return schemas.History(cell_id=cell_id, rows=rows)


@router.get("/cells/{cell_id}/explain", response_model=schemas.Explanation)
def explain(request: Request, cell_id: str, date: str) -> schemas.Explanation:
    """Top-5 SHAP contributions (log-odds) for one cell and prediction date."""
    s = _store(request)
    _check_cell(s, cell_id)
    ts = _check_date(s, date)
    rows = s.shap[(s.shap["cell_id"] == cell_id) & (s.shap["week_start"] == ts)]
    pred = s.predictions[(s.predictions["cell_id"] == cell_id) & (s.predictions["week_start"] == ts)]
    if rows.empty or pred.empty:
        raise HTTPException(404, f"No explanation for {cell_id} on {date}")
    contribs = [
        schemas.Contribution(rank=int(r.rank), feature=r.feature, value=float(r.value),
                             contribution=float(r.contribution))
        for r in rows.sort_values("rank").itertuples()
    ]
    return schemas.Explanation(cell_id=cell_id, date=date, prob=float(pred["prob"].iloc[0]),
                               contributions=contribs)


@router.get("/model", response_model=schemas.ModelInfo)
def model(request: Request) -> schemas.ModelInfo:
    """Model version, features, training window, hyperparameters and metrics."""
    return schemas.ModelInfo(**_store(request).model_info)
