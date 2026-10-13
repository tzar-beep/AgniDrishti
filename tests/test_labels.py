import pandas as pd

from agnidrishti.labels import make_labels, week_starts


def test_week_starts_are_mondays_with_full_window():
    weeks = week_starts("2024-01-01", "2024-01-31")
    assert all(w.weekday() == 0 for w in weeks)
    assert weeks[-1] == pd.Timestamp("2024-01-22")  # 01-29 window would end 02-04


def test_label_window_is_next_seven_days():
    weeks = week_starts("2024-01-01", "2024-01-31")
    fires = pd.DataFrame({
        "cell_id": ["a", "a", "b"],
        "acq_date": pd.to_datetime(["2024-01-07", "2024-01-08", "2024-01-21"]),
    })
    lab = make_labels(fires, ["a", "b"], weeks).set_index(["cell_id", "week_start"])["y"]
    assert lab[("a", pd.Timestamp("2024-01-01"))] == 1  # Sunday 01-07 is in week 01-01
    assert lab[("a", pd.Timestamp("2024-01-08"))] == 1
    assert lab[("a", pd.Timestamp("2024-01-15"))] == 0
    assert lab[("b", pd.Timestamp("2024-01-15"))] == 1
    assert lab[("b", pd.Timestamp("2024-01-22"))] == 0
    assert lab.sum() == 3


def test_no_fires_gives_all_zero():
    weeks = week_starts("2024-01-01", "2024-01-31")
    empty = pd.DataFrame({"cell_id": [], "acq_date": pd.to_datetime([])})
    assert make_labels(empty, ["a"], weeks)["y"].sum() == 0
