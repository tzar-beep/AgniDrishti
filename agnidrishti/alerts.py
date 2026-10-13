"""Alert levels, public advice and district bulletins (JSON and CAP 1.2 XML)."""

from datetime import UTC, datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

LEVELS = ["Low", "Moderate", "High", "Very High"]
# Quantiles of validation-period probabilities that separate the levels.
LEVEL_QUANTILES = {"Moderate": 0.50, "High": 0.75, "Very High": 0.90}

ADVICE = {
    "Low": "Normal precautions. Do not leave fires unattended and dispose of cigarettes safely.",
    "Moderate": "Avoid burning leaves, crop residue or waste near forest edges. Keep water or sand "
                "at hand when using open flames.",
    "High": "Do not light any open fire in or near forests. Clear dry pine needles and litter "
            "around homes. Report smoke or fire immediately to the local forest range office "
            "or emergency number 112.",
    "Very High": "Extreme caution: no open fires or burning of any kind. Prepare to leave forest-edge "
                 "areas if a fire approaches; keep livestock and fodder safe. Report any fire "
                 "immediately to the forest department or 112 and follow official instructions.",
}
CAP_SEVERITY = {"Low": "Minor", "Moderate": "Moderate", "High": "Severe", "Very High": "Extreme"}


def compute_thresholds(val_probs: np.ndarray) -> dict[str, float]:
    """Probability cut-offs for each level from validation-period predictions."""
    return {lvl: round(float(np.quantile(val_probs, q)), 4) for lvl, q in LEVEL_QUANTILES.items()}


def assign_levels(probs: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    """Map probabilities to alert level names."""
    out = pd.Series("Low", index=probs.index)
    for lvl in ("Moderate", "High", "Very High"):
        out[probs >= thresholds[lvl]] = lvl
    return out


def district_bulletin(day: pd.DataFrame, thresholds: dict[str, float]) -> list[dict]:
    """Per-district summary: highest level, max probability and affected cells.

    ``day`` has ``cell_id``, ``prob`` and ``district`` for one prediction date.
    """
    d = day.assign(level=assign_levels(day["prob"], thresholds))
    d["rank"] = d["level"].map(LEVELS.index)
    rows = []
    for name, g in d.groupby("district"):
        top = g.sort_values("prob", ascending=False)
        lvl = LEVELS[int(g["rank"].max())]
        rows.append({
            "district": name,
            "level": lvl,
            "max_prob": round(float(g["prob"].max()), 4),
            "mean_prob": round(float(g["prob"].mean()), 4),
            "n_cells": int(len(g)),
            "n_elevated": int((g["rank"] >= 2).sum()),
            "cells": top["cell_id"].tolist(),
            "advice": ADVICE[lvl],
        })
    return sorted(rows, key=lambda r: (-LEVELS.index(r["level"]), -r["max_prob"]))


def to_cap(date: str, districts: list[dict], is_forecast: bool, sender: str = "agnidrishti") -> str:
    """Render a CAP 1.2 alert with one <info> block per district at High or above."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    start = f"{date}T00:00:00+05:30"
    end = (pd.Timestamp(date) + pd.Timedelta(days=7)).strftime("%Y-%m-%dT00:00:00+05:30")
    infos = []
    for d in districts:
        if LEVELS.index(d["level"]) < 2:
            continue
        infos.append(f"""  <info>
    <language>en-IN</language>
    <category>Fire</category>
    <event>Forest fire danger ({escape(d["level"])})</event>
    <urgency>{"Expected" if is_forecast else "Past"}</urgency>
    <severity>{CAP_SEVERITY[d["level"]]}</severity>
    <certainty>Possible</certainty>
    <effective>{start}</effective>
    <expires>{end}</expires>
    <senderName>AgniDrishti (research prototype)</senderName>
    <headline>{escape(d["level"])} forest fire danger in {escape(d["district"])}, week of {date}</headline>
    <description>Model probability of a satellite fire detection up to {d["max_prob"]:.0%} in {d["n_elevated"]} of {d["n_cells"]} grid cells.</description>
    <instruction>{escape(d["advice"])}</instruction>
    <area><areaDesc>{escape(d["district"])}, Uttarakhand</areaDesc></area>
  </info>""")
    status = "Actual" if is_forecast else "Exercise"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>agnidrishti-{date}</identifier>
  <sender>{sender}</sender>
  <sent>{now}</sent>
  <status>{status}</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <note>Research prototype, not an official warning. Follow instructions from the Uttarakhand Forest Department and district authorities.</note>
{chr(10).join(infos)}
</alert>
"""
