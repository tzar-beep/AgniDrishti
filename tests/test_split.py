import pandas as pd

from agnidrishti import config
from agnidrishti.labels import week_starts
from agnidrishti.train import split_by_time


def test_temporal_split_has_no_overlap():
    weeks = week_starts(f"{config.START_YEAR}-01-01", f"{config.END_YEAR}-12-31")
    df = pd.DataFrame({"cell_id": "c", "week_start": weeks, "y": 0})
    train, val, test = split_by_time(df)
    horizon = pd.Timedelta(days=config.HORIZON_DAYS - 1)
    assert len(train) + len(val) + len(test) == len(df)
    # Neither the forecast dates nor the label windows overlap across splits.
    assert train["week_start"].max() + horizon < val["week_start"].min()
    assert val["week_start"].max() + horizon < test["week_start"].min()
    assert set(train["week_start"]).isdisjoint(val["week_start"])
    assert set(val["week_start"]).isdisjoint(test["week_start"])
    assert test["week_start"].dt.year.unique().tolist() == [2024]
