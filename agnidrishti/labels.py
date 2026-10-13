"""Weekly binary labels: did a fire hotspot occur in the cell in the next 7 days?"""

import logging

import pandas as pd

from agnidrishti import config

log = logging.getLogger(__name__)


def week_starts(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Mondays (ISO week starts) whose full 7-day label window lies within [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mondays = pd.date_range(start, end, freq="W-MON")
    return mondays[mondays + pd.Timedelta(days=config.HORIZON_DAYS - 1) <= end]


def make_labels(fires: pd.DataFrame, cell_ids: list[str], weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one row per (cell_id, week_start) with y=1 if any detection is in [ws, ws+7d).

    ``fires`` must already be confidence-filtered and have ``cell_id`` and ``acq_date``.
    """
    base = pd.MultiIndex.from_product([cell_ids, weeks], names=["cell_id", "week_start"])
    base = base.to_frame(index=False)
    if fires.empty:
        return base.assign(y=0)
    f = fires[["cell_id", "acq_date"]].copy()
    f["acq_date"] = pd.to_datetime(f["acq_date"]).dt.normalize()
    # Map each detection to every week_start whose window contains it.
    hits = []
    for lag in range(config.HORIZON_DAYS):
        ws = f["acq_date"] - pd.Timedelta(days=lag)
        hits.append(pd.DataFrame({"cell_id": f["cell_id"], "week_start": ws}))
    hits = pd.concat(hits).drop_duplicates()
    hits = hits[hits["week_start"].isin(weeks)].assign(y=1)
    out = base.merge(hits, on=["cell_id", "week_start"], how="left")
    out["y"] = out["y"].fillna(0).astype(int)
    log.info("Labels: %d rows, positive rate %.4f", len(out), out["y"].mean())
    return out
