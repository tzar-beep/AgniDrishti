import json

import geopandas as gpd
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from agnidrishti.api.main import create_app
from agnidrishti.features import FEATURES

DATES = pd.date_range("2024-01-01", periods=14, freq="W-MON")
CELLS = ["r477c1031", "r477c1032"]
FORECAST = pd.Timestamp("2026-10-19")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    gpd.GeoDataFrame(
        {"cell_id": CELLS, "lat": [29.375, 29.375], "lon": [77.875, 78.125],
         "district": ["Hardwar", "Dehradun"]},
        geometry=[box(77.75, 29.25, 78.0, 29.5), box(78.0, 29.25, 78.25, 29.5)],
        crs=4326,
    ).to_parquet(d / "grid.parquet")

    def tables(dates, prefix, prob, y):
        keys = pd.MultiIndex.from_product([CELLS, dates], names=["cell_id", "week_start"])
        keys = keys.to_frame(index=False)
        keys.assign(y=y, prob=prob).to_parquet(d / f"{prefix}predictions.parquet")
        ds = keys.assign(**{f: 1.0 for f in FEATURES}, y=y)
        ds.to_parquet(d / ("forecast_features.parquet" if prefix else "dataset.parquet"))
        shap = keys.loc[keys.index.repeat(5)].reset_index(drop=True)
        shap = shap.assign(rank=list(range(1, 6)) * len(keys), feature=FEATURES[0], value=1.0,
                           contribution=0.1)
        shap.to_parquet(d / f"{prefix}shap_top5.parquet")

    tables(DATES, "", [0.2] * 14 + [0.9] * 14, 0)
    tables([FORECAST], "forecast_", 0.95, pd.array([pd.NA, pd.NA], dtype="Int64"))
    model_json = d / "model.json"
    model_json.write_text(json.dumps({
        "version": "xgb_v1", "features": FEATURES, "training_window": {},
        "hyperparameters": {}, "metrics": {"xgboost": {"test": {"pr_auc": 0.5}}},
        "alert_thresholds": {"Moderate": 0.3, "High": 0.6, "Very High": 0.8},
    }))
    return TestClient(create_app(d, model_json))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "n_cells": 2, "n_dates": 15}


def test_dates(client):
    body = client.get("/api/dates").json()
    assert body["dates"][0] == "2024-01-01" and len(body["dates"]) == 15
    assert body["forecast_dates"] == ["2026-10-19"]


def test_risk(client):
    body = client.get("/api/risk", params={"date": "2024-01-08"}).json()
    assert body["type"] == "FeatureCollection"
    assert {f["properties"]["cell_id"] for f in body["features"]} == set(CELLS)
    assert body["features"][0]["properties"]["prob"] == 0.2
    assert body["features"][0]["properties"]["observed"] == 0
    assert body["features"][0]["geometry"]["type"] == "Polygon"
    assert body["is_forecast"] is False


def test_risk_forecast_has_no_observation(client):
    body = client.get("/api/risk", params={"date": "2026-10-19"}).json()
    assert body["is_forecast"] is True
    assert all(f["properties"]["observed"] is None for f in body["features"])
    assert {f["properties"]["level"] for f in body["features"]} == {"Very High"}


def test_risk_unknown_date(client):
    assert client.get("/api/risk", params={"date": "1999-01-04"}).status_code == 404
    assert client.get("/api/risk", params={"date": "bad"}).status_code == 422


def test_history(client):
    body = client.get(f"/api/cells/{CELLS[0]}/history", params={"date": "2024-04-01"}).json()
    assert len(body["rows"]) == 12
    assert body["rows"][-1]["week_start"] == "2024-04-01"
    assert set(body["rows"][0]["features"]) == set(FEATURES)


def test_history_forecast_week(client):
    rows = client.get(f"/api/cells/{CELLS[0]}/history").json()["rows"]
    assert [r["week_start"] for r in rows] == ["2026-10-19"] and rows[0]["y"] is None


def test_alerts(client):
    body = client.get("/api/alerts", params={"date": "2024-01-08"}).json()
    levels = {d["district"]: d["level"] for d in body["districts"]}
    assert levels == {"Hardwar": "Low", "Dehradun": "Very High"}
    assert body["districts"][0]["district"] == "Dehradun"  # highest level first
    assert body["levels"] == ["Low", "Moderate", "High", "Very High"]


def test_alerts_cap(client):
    r = client.get("/api/alerts/cap", params={"date": "2026-10-19"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/cap+xml")
    assert "urn:oasis:names:tc:emergency:cap:1.2" in r.text
    assert r.text.count("<info>") == 2 and "<status>Actual</status>" in r.text
    assert client.get("/api/alerts", params={"date": "1999-01-04"}).status_code == 404


def test_explain(client):
    body = client.get(f"/api/cells/{CELLS[0]}/explain", params={"date": "2024-01-08"}).json()
    assert len(body["contributions"]) == 5
    assert body["prob"] == pytest.approx(0.2)


def test_unknown_cell_404(client):
    assert client.get("/api/cells/nope/history").status_code == 404
    r = client.get("/api/cells/nope/explain", params={"date": "2024-01-08"})
    assert r.status_code == 404


def test_model(client):
    assert client.get("/api/model").json()["version"] == "xgb_v1"


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "leaflet" in r.text.lower()
