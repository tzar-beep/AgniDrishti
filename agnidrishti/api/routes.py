"""API routes. All data comes from precomputed tables held in ``app.state.store``."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from shapely.geometry import mapping

from agnidrishti.api import schemas
from agnidrishti.features import FEATURES

router = APIRouter(prefix="/api")
HISTORY_WEEKS = 12


@dataclass
class Store:
    """In-memory copies of the precomputed prediction, explanation and feature tables."""

    geometry: dict[str, dict]
    predictions: pd.DataFrame
    shap: pd.DataFrame
    dataset: pd.DataFrame
    model_info: dict
    dates: list[str]

    @classmethod
    def load(cls, processed_dir: Path, model_json: Path) -> "Store":
        """Load all tables from disk."""
        grid = gpd.read_parquet(processed_dir / "grid.parquet")
        preds = pd.read_parquet(processed_dir / "predictions.parquet")
        dates = sorted(preds["week_start"].dt.strftime("%Y-%m-%d").unique())
        return cls(
            geometry={r.cell_id: mapping(r.geometry) for r in grid.itertuples()},
            predictions=preds,
            shap=pd.read_parquet(processed_dir / "shap_top5.parquet"),
            dataset=pd.read_parquet(processed_dir / "dataset.parquet"),
            model_info=json.loads(model_json.read_text()),
            dates=dates,
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
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


@router.get("/health", response_model=schemas.Health)
def health(request: Request) -> schemas.Health:
    """Liveness check with basic table sizes."""
    s = _store(request)
    return schemas.Health(status="ok", n_cells=len(s.geometry), n_dates=len(s.dates))


@router.get("/dates", response_model=schemas.Dates)
def dates(request: Request) -> schemas.Dates:
    """All prediction dates (week starts) available in the predictions table."""
    return schemas.Dates(dates=_store(request).dates)


@router.get("/risk", response_model=schemas.RiskCollection)
def risk(request: Request, date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """GeoJSON FeatureCollection of cell probabilities for one prediction date."""
    s = _store(request)
    ts = _check_date(s, date)
    day = s.predictions[s.predictions["week_start"] == ts]
    feats = [
        schemas.RiskFeature(
            geometry=s.geometry[r.cell_id],
            properties={"cell_id": r.cell_id, "prob": round(float(r.prob), 4)},
        )
        for r in day.itertuples()
    ]
    return schemas.RiskCollection(date=date, features=feats)


@router.get("/cells/{cell_id}/history", response_model=schemas.History)
def history(request: Request, cell_id: str, date: str | None = None) -> schemas.History:
    """Features, labels and predicted probability for the 12 weeks up to ``date``."""
    s = _store(request)
    _check_cell(s, cell_id)
    end = _check_date(s, date) if date else pd.Timestamp(s.dates[-1])
    d = s.dataset[(s.dataset["cell_id"] == cell_id) & (s.dataset["week_start"] <= end)]
    d = d.sort_values("week_start").tail(HISTORY_WEEKS)
    probs = s.predictions[s.predictions["cell_id"] == cell_id].set_index("week_start")["prob"]
    rows = [
        schemas.HistoryRow(
            week_start=r["week_start"].strftime("%Y-%m-%d"),
            y=int(r["y"]),
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
