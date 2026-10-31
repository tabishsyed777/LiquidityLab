"""Synthetic intraday payment flow data and input validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 26
REQUIRED = ["date", "slot", "incoming_m", "outgoing_m"]


def make_demo_data(n_days: int = 180, seed: int = 37) -> pd.DataFrame:
    """Generate 15-minute aggregate payment flows in arbitrary millions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n_days, freq="D")
    slot = np.arange(SLOTS_PER_DAY)
    bell = np.sin(np.pi * (slot + 0.5) / SLOTS_PER_DAY)
    bell = bell / bell.max()
    rows = []
    day_factors = rng.lognormal(mean=0, sigma=0.12, size=n_days)
    for day_index, date in enumerate(dates):
        weekend_factor = 0.48 if date.dayofweek >= 5 else 1.0
        factor = day_factors[day_index] * weekend_factor
        incoming = np.maximum(
            0.05, (0.8 + 4.2 * bell) * factor + rng.normal(0, 0.38, SLOTS_PER_DAY)
        )
        outgoing = np.maximum(
            0.05, (0.7 + 4.5 * bell) * factor + rng.normal(0, 0.43, SLOTS_PER_DAY)
        )
        # A settlement-day surge creates realistic peaks and a visible stress case.
        if rng.random() < 0.09:
            first = int(rng.integers(4, 18))
            width = int(rng.integers(2, 6))
            outgoing[first:first + width] += rng.uniform(1.0, 3.5)
        for position in range(SLOTS_PER_DAY):
            rows.append({
                "date": date,
                "slot": position,
                "incoming_m": float(incoming[position]),
                "outgoing_m": float(outgoing[position]),
            })
    return pd.DataFrame(rows)


def validate_flows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        return frame, [f"Missing required columns: {', '.join(missing)}"]
    data = frame[REQUIRED].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["slot", "incoming_m", "outgoing_m"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    data = data[(data["slot"] >= 0) & (data["incoming_m"] >= 0) & (data["outgoing_m"] >= 0)]
    data["slot"] = data["slot"].astype(int)
    data = data.sort_values(["date", "slot"]).reset_index(drop=True)
    issues = []
    if data.empty:
        issues.append("No valid non-negative flow rows remain after cleaning.")
    elif data.duplicated(["date", "slot"]).any():
        issues.append("Each date and slot pair must be unique.")
    elif data["date"].nunique() < 6:
        issues.append("Provide at least six distinct dates for the forecasting history.")
    return data, issues
