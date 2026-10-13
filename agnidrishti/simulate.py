"""What-if scenarios: perturb a cell's weather features and re-score them with the model."""

import pandas as pd

from agnidrishti.features import fosberg_fwi


def apply_scenario(
    row: pd.Series,
    d_tmax: float = 0.0,
    d_rh: float = 0.0,
    wind_mult: float = 1.0,
    days_since_rain: int | None = None,
) -> pd.Series:
    """Return a copy of a feature row with shifted temperature/humidity, scaled wind and an
    optional dry-spell length. ``fwi_7d`` is rescaled by the ratio of the Fosberg index at the
    new versus original 7-day means, so the fire weather index stays consistent.
    """
    r = row.copy()
    for w in (7, 14):
        r[f"tmax_mean_{w}d"] = row[f"tmax_mean_{w}d"] + d_tmax
        r[f"rh_mean_{w}d"] = min(max(row[f"rh_mean_{w}d"] + d_rh, 0.0), 100.0)
        r[f"wind_mean_{w}d"] = row[f"wind_mean_{w}d"] * wind_mult
    if days_since_rain is not None:
        r["days_since_rain"] = float(days_since_rain)

    def ffwi(x: pd.Series) -> float:
        return float(fosberg_fwi(pd.Series([x["tmax_mean_7d"]]), pd.Series([x["rh_mean_7d"]]),
                                 pd.Series([x["wind_mean_7d"]])).iloc[0])

    old, new = ffwi(row), ffwi(r)
    r["fwi_7d"] = row["fwi_7d"] * new / old if old > 0 else new
    return r
