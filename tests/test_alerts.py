import numpy as np
import pandas as pd

from agnidrishti.alerts import assign_levels, compute_thresholds, district_bulletin, to_cap

THR = {"Moderate": 0.3, "High": 0.6, "Very High": 0.8}


def test_thresholds_are_increasing_quantiles():
    thr = compute_thresholds(np.linspace(0, 1, 101))
    assert thr == {"Moderate": 0.5, "High": 0.75, "Very High": 0.9}


def test_assign_levels_boundaries():
    probs = pd.Series([0.0, 0.3, 0.59, 0.6, 0.8, 1.0])
    assert assign_levels(probs, THR).tolist() == [
        "Low", "Moderate", "Moderate", "High", "Very High", "Very High"]


def test_bulletin_takes_worst_cell_per_district_and_sorts():
    day = pd.DataFrame({
        "cell_id": ["a", "b", "c"], "prob": [0.1, 0.85, 0.4], "district": ["X", "X", "Y"]})
    b = district_bulletin(day, THR)
    assert [d["district"] for d in b] == ["X", "Y"]
    assert b[0]["level"] == "Very High" and b[0]["cells"] == ["b", "a"] and b[0]["n_elevated"] == 1
    assert b[1]["level"] == "Moderate"


def test_cap_only_includes_high_and_above():
    day = pd.DataFrame({"cell_id": ["a", "b"], "prob": [0.9, 0.1], "district": ["X", "Y"]})
    xml = to_cap("2024-04-15", district_bulletin(day, THR), is_forecast=True)
    assert xml.count("<info>") == 1 and "<severity>Extreme</severity>" in xml
    assert "<expires>2024-04-22T00:00:00+05:30</expires>" in xml
