import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from agnidrishti.features import FEATURES, build_features
from agnidrishti.labels import week_starts


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    cells = ["c1", "c2"]
    dates = pd.date_range("2020-01-01", "2021-12-31")
    weather = pd.DataFrame(
        [(c, d) for c in cells for d in dates], columns=["cell_id", "date"]
    )
    n = len(weather)
    weather["tmax"] = rng.normal(25, 5, n)
    weather["tmin"] = weather["tmax"] - 10
    weather["rh"] = rng.uniform(10, 90, n)
    weather["wind"] = rng.uniform(0, 30, n)
    weather["precip"] = rng.exponential(1.5, n)
    fire_days = rng.choice(len(weather), 120, replace=False)
    fires = weather.iloc[fire_days][["cell_id", "date"]].rename(columns={"date": "acq_date"})
    terrain = pd.DataFrame({"cell_id": cells, "elev_mean": [500.0, 2000.0], "slope_mean": [5.0, 20.0]})
    weeks = week_starts("2020-01-01", "2021-12-31")
    return weather, fires, terrain, weeks


@pytest.mark.parametrize("t", ["2020-06-01", "2021-03-15", "2021-11-29"])
def test_features_unchanged_when_future_removed(synthetic, t):
    weather, fires, terrain, weeks = synthetic
    t = pd.Timestamp(t)
    full = build_features(weather, fires, terrain, weeks)
    past_weeks = weeks[weeks <= t]
    trunc = build_features(
        weather[weather["date"] < t], fires[fires["acq_date"] < t], terrain, past_weeks
    )
    a = full[full["week_start"] == t].reset_index(drop=True)
    b = trunc[trunc["week_start"] == t].reset_index(drop=True)
    assert len(a) == 2
    pdt.assert_frame_equal(a, b)


def test_no_missing_features(synthetic):
    df = build_features(*synthetic)
    assert not df[FEATURES].isna().any().any()
