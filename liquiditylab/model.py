"""Chronological flow forecasting and buffer-policy simulation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from liquiditylab.data import SLOTS_PER_DAY, make_demo_data

FEATURES = [
    "slot_sin", "slot_cos", "weekday_sin", "weekday_cos",
    "net_lag1", "net_mean5", "incoming_lag1", "outgoing_lag1",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values(["date", "slot"]).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["net_m"] = data["outgoing_m"] - data["incoming_m"]
    slot_angle = 2 * np.pi * data["slot"] / SLOTS_PER_DAY
    day_angle = 2 * np.pi * data["date"].dt.dayofweek / 7
    data["slot_sin"] = np.sin(slot_angle)
    data["slot_cos"] = np.cos(slot_angle)
    data["weekday_sin"] = np.sin(day_angle)
    data["weekday_cos"] = np.cos(day_angle)
    grouped = data.groupby("slot", sort=False)
    data["net_lag1"] = grouped["net_m"].shift(1)
    data["net_mean5"] = grouped["net_m"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=3).mean()
    )
    data["incoming_lag1"] = grouped["incoming_m"].shift(1)
    data["outgoing_lag1"] = grouped["outgoing_m"].shift(1)
    return data


@dataclass
class ForecastModel:
    estimator: HistGradientBoostingRegressor
    evaluation: dict[str, float]
    uncertainty_buffer_m: float
    train_days: list[pd.Timestamp]
    calibration_days: list[pd.Timestamp]
    test_days: list[pd.Timestamp]
    training_frame: pd.DataFrame
    scored_test: pd.DataFrame


def fit_demo_model(seed: int = 13) -> tuple[ForecastModel, pd.DataFrame]:
    data = add_features(make_demo_data())
    dates = sorted(data["date"].unique())
    train_end = int(len(dates) * 0.70)
    calibration_end = int(len(dates) * 0.80)
    train_days = list(dates[:train_end])
    calibration_days = list(dates[train_end:calibration_end])
    test_days = list(dates[calibration_end:])

    train = data[data["date"].isin(train_days)].dropna(subset=FEATURES)
    calibration = data[data["date"].isin(calibration_days)].dropna(subset=FEATURES)
    test = data[data["date"].isin(test_days)].dropna(subset=FEATURES)
    estimator = HistGradientBoostingRegressor(
        learning_rate=0.07, max_iter=120, max_leaf_nodes=15,
        l2_regularization=1.2, random_state=seed,
    )
    estimator.fit(train[FEATURES], train["net_m"])

    calibration = calibration.copy()
    calibration["predicted_net_m"] = estimator.predict(calibration[FEATURES])
    underforecast = []
    for _, daily in calibration.groupby("date", sort=True):
        actual_peak = float(np.maximum(daily["net_m"].cumsum(), 0).max())
        predicted_peak = float(np.maximum(daily["predicted_net_m"].cumsum(), 0).max())
        underforecast.append(max(0.0, actual_peak - predicted_peak))
    uncertainty = float(np.quantile(underforecast, 0.90, method="higher"))

    test = test.copy()
    test["predicted_net_m"] = estimator.predict(test[FEATURES])
    mae = float(mean_absolute_error(test["net_m"], test["predicted_net_m"]))
    rmse = float(np.sqrt(mean_squared_error(test["net_m"], test["predicted_net_m"])))
    peak_errors = []
    for _, daily in test.groupby("date", sort=True):
        actual_peak = float(np.maximum(daily["net_m"].cumsum(), 0).max())
        predicted_peak = float(np.maximum(daily["predicted_net_m"].cumsum(), 0).max())
        peak_errors.append(actual_peak - predicted_peak)
    coverage = float(np.mean(np.asarray(peak_errors) <= uncertainty))

    model = ForecastModel(
        estimator=estimator,
        evaluation={
            "window_mae_m": mae,
            "window_rmse_m": rmse,
            "peak_error_90_coverage": coverage,
        },
        uncertainty_buffer_m=uncertainty,
        train_days=[pd.Timestamp(day) for day in train_days],
        calibration_days=[pd.Timestamp(day) for day in calibration_days],
        test_days=[pd.Timestamp(day) for day in test_days],
        training_frame=data[data["date"].isin(train_days)].copy(),
        scored_test=test,
    )
    return model, data


def forecast_day(frame: pd.DataFrame, model: ForecastModel, target_date) -> pd.DataFrame:
    """Forecast each slot for a new date using histories from prior dates only."""
    target = pd.Timestamp(target_date).normalize()
    template = pd.DataFrame({
        "date": [target] * SLOTS_PER_DAY,
        "slot": np.arange(SLOTS_PER_DAY),
        "incoming_m": [np.nan] * SLOTS_PER_DAY,
        "outgoing_m": [np.nan] * SLOTS_PER_DAY,
    })
    combined = pd.concat([frame, template], ignore_index=True, sort=False)
    featured = add_features(combined)
    today = featured[featured["date"] == target].copy().sort_values("slot")
    today["predicted_net_m"] = model.estimator.predict(today[FEATURES])
    return today


def _stress_net(net: np.ndarray, outflow_multiplier: float, inflow_multiplier: float) -> np.ndarray:
    return np.where(net >= 0, net * outflow_multiplier, net * inflow_multiplier)


def compare_buffers(
    model: ForecastModel,
    stress_outflow: float = 1.0,
    stress_inflow: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Replay a fixed quantile buffer and forecast-led buffer on the test period."""
    if stress_outflow <= 0 or stress_inflow < 0:
        raise ValueError("Stress multipliers must be positive for outflows and non-negative for inflows.")
    training = model.training_frame.copy()
    training["stress_net_m"] = _stress_net(
        training["net_m"].to_numpy(), stress_outflow, stress_inflow
    )
    daily_peaks = training.groupby("date", sort=True)["stress_net_m"].apply(
        lambda values: float(np.maximum(values.cumsum(), 0).max())
    )
    fixed_buffer = float(daily_peaks.quantile(0.90))

    test = model.scored_test.copy()
    test["stress_net_m"] = _stress_net(
        test["net_m"].to_numpy(), stress_outflow, stress_inflow
    )
    test["stress_prediction_m"] = _stress_net(
        test["predicted_net_m"].to_numpy(), stress_outflow, stress_inflow
    )

    records = []
    uncertainty = model.uncertainty_buffer_m * max(stress_outflow, stress_inflow)
    for date, daily in test.groupby("date", sort=True):
        actual_cumulative = daily["stress_net_m"].cumsum().to_numpy()
        predicted_cumulative = daily["stress_prediction_m"].cumsum().to_numpy()
        fixed_shortfall = np.maximum(actual_cumulative - fixed_buffer, 0)
        forecast_buffer = max(0.0, float(np.maximum(predicted_cumulative, 0).max()) + uncertainty)
        forecast_shortfall = np.maximum(actual_cumulative - forecast_buffer, 0)
        records.append({
            "date": date,
            "actual_peak_need_m": float(np.maximum(actual_cumulative, 0).max()),
            "fixed_buffer_m": fixed_buffer,
            "forecast_buffer_m": forecast_buffer,
            "fixed_shortfall_m": float(fixed_shortfall.max()),
            "forecast_shortfall_m": float(forecast_shortfall.max()),
            "fixed_shortfall_slots": int(np.sum(fixed_shortfall > 0)),
            "forecast_shortfall_slots": int(np.sum(forecast_shortfall > 0)),
        })
    replay = pd.DataFrame(records)
    summary = {
        "Fixed reserve": {
            "average_buffer_m": float(replay["fixed_buffer_m"].mean()),
            "days_with_shortfall": float((replay["fixed_shortfall_m"] > 0).sum()),
            "mean_daily_peak_shortfall_m": float(replay["fixed_shortfall_m"].mean()),
            "shortfall_slots": float(replay["fixed_shortfall_slots"].sum()),
        },
        "Forecast reserve": {
            "average_buffer_m": float(replay["forecast_buffer_m"].mean()),
            "days_with_shortfall": float((replay["forecast_shortfall_m"] > 0).sum()),
            "mean_daily_peak_shortfall_m": float(replay["forecast_shortfall_m"].mean()),
            "shortfall_slots": float(replay["forecast_shortfall_slots"].sum()),
        },
    }
    return replay, summary
