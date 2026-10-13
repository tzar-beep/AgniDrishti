import pandas as pd
import pytest

from agnidrishti.features import FEATURES
from agnidrishti.simulate import apply_scenario


@pytest.fixture
def row():
    r = pd.Series(1.0, index=FEATURES)
    r[["tmax_mean_7d", "tmax_mean_14d"]] = 30.0
    r[["rh_mean_7d", "rh_mean_14d"]] = 40.0
    r[["wind_mean_7d", "wind_mean_14d"]] = 10.0
    r["fwi_7d"] = 12.0
    return r


def test_no_change_is_identity(row):
    pd.testing.assert_series_equal(apply_scenario(row), row)


def test_hotter_drier_windier_raises_fwi(row):
    r = apply_scenario(row, d_tmax=5, d_rh=-20, wind_mult=1.5, days_since_rain=40)
    assert r["tmax_mean_14d"] == 35 and r["rh_mean_7d"] == 20 and r["wind_mean_7d"] == 15
    assert r["days_since_rain"] == 40 and r["fwi_7d"] > row["fwi_7d"]


def test_humidity_is_clipped(row):
    assert apply_scenario(row, d_rh=-90)["rh_mean_7d"] == 0
    assert apply_scenario(row, d_rh=90)["rh_mean_14d"] == 100
