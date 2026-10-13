"""Pydantic response models for the AgniDrishti API."""

from typing import Any

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    n_cells: int
    n_dates: int


class Dates(BaseModel):
    dates: list[str]
    forecast_dates: list[str] = []


class RiskFeature(BaseModel):
    type: str = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any]


class RiskCollection(BaseModel):
    type: str = "FeatureCollection"
    date: str
    is_forecast: bool = False
    features: list[RiskFeature]


class DistrictAlert(BaseModel):
    district: str
    level: str
    max_prob: float
    mean_prob: float
    n_cells: int
    n_elevated: int
    cells: list[str]
    advice: str


class Alerts(BaseModel):
    date: str
    is_forecast: bool
    thresholds: dict[str, float]
    levels: list[str]
    advice: dict[str, str]
    districts: list[DistrictAlert]


class HistoryRow(BaseModel):
    week_start: str
    y: int | None
    prob: float | None
    features: dict[str, float | None]


class History(BaseModel):
    cell_id: str
    rows: list[HistoryRow]


class Contribution(BaseModel):
    rank: int
    feature: str
    value: float
    contribution: float


class Explanation(BaseModel):
    cell_id: str
    date: str
    prob: float
    contributions: list[Contribution]


class ModelInfo(BaseModel):
    version: str
    trained_at: str | None = None
    features: list[str]
    training_window: dict[str, Any]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    alert_thresholds: dict[str, float] | None = None
