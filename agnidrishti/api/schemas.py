"""Pydantic response models for the AgniDrishti API."""

from typing import Any

from pydantic import BaseModel, Field


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


class TimeSeries(BaseModel):
    dates: list[str]
    mean_prob: list[float]
    observed: list[int | None]
    elevated: list[int]
    n_cells: int


class Evaluation(BaseModel):
    year: int
    prob: list[float]
    y: list[int]


class Scenario(BaseModel):
    date: str
    d_tmax: float = Field(0.0, ge=-10, le=10, description="Shift of Tmax means, deg C")
    d_rh: float = Field(0.0, ge=-50, le=50, description="Shift of RH means, percentage points")
    wind_mult: float = Field(1.0, ge=0, le=3, description="Multiplier on wind means")
    days_since_rain: int | None = Field(None, ge=0, le=365)


class SimulationResult(BaseModel):
    cell_id: str
    date: str
    base_prob: float
    prob: float
    features: dict[str, float]
    contributions: list[Contribution]
