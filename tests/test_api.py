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


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    gpd.GeoDataFrame(
        {"cell_id": CELLS, "lat": [29.375, 29.375], "lon": [77.875, 78.125]},
        geometry=[box(77.75, 29.25, 78.0, 29.5), box(78.0, 29.25, 78.25, 29.5)],
        crs=4326,
    ).to_parquet(d / "grid.parquet")
    keys = pd.MultiIndex.from_product([CELLS, DATES], names=["cell_id", "week_start"])
    keys = keys.to_frame(index=False)
    pd.concat([keys, pd.DataFrame({"y": 0, "prob": 0.2}, index=keys.index)], axis=1).to_parquet(
        d / "predictions.parquet"
    )
    ds = keys.copy()
    for f in FEATURES:
        ds[f] = 1.0
    ds["y"] = 0
    ds.to_parquet(d / "dataset.parquet")
    shap = keys.loc[keys.index.repeat(5)].reset_index(drop=True)
    shap["rank"] = list(range(1, 6)) * len(keys)
    shap["feature"] = FEATURES[0]
    shap["value"] = 1.0
    shap["contribution"] = 0.1
    shap.to_parquet(d / "shap_top5.parquet")
    model_json = d / "model.json"
    model_json.write_text(json.dumps({
        "version": "xgb_v1", "features": FEATURES, "training_window": {},
        "hyperparameters": {}, "metrics": {"xgboost": {"test": {"pr_auc": 0.5}}},
    }))
    return TestClient(create_app(d, model_json))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "n_cells": 2, "n_dates": 14}


def test_dates(client):
    dates = client.get("/api/dates").json()["dates"]
    assert dates[0] == "2024-01-01" and len(dates) == 14


def test_risk(client):
    body = client.get("/api/risk", params={"date": "2024-01-08"}).json()
    assert body["type"] == "FeatureCollection"
    assert {f["properties"]["cell_id"] for f in body["features"]} == set(CELLS)
    assert body["features"][0]["properties"]["prob"] == 0.2


def test_risk_unknown_date(client):
    assert client.get("/api/risk", params={"date": "1999-01-04"}).status_code == 404
    assert client.get("/api/risk", params={"date": "bad"}).status_code == 422


def test_history(client):
    body = client.get(f"/api/cells/{CELLS[0]}/history").json()
    assert len(body["rows"]) == 12
    assert set(body["rows"][0]["features"]) == set(FEATURES)


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
