from __future__ import annotations

import pandas as pd
import streamlit as st

from liquiditylab.data import SLOTS_PER_DAY, make_demo_data, validate_flows
from liquiditylab.model import FEATURES, add_features, compare_buffers, fit_demo_model, forecast_day

st.set_page_config(page_title="LiquidityLab", page_icon="◌", layout="wide")


@st.cache_resource
def load_demo():
    return fit_demo_model()


model, training_scored = load_demo()
st.title("◌ LiquidityLab")
st.caption("Intraday payment-flow forecasting · reserve simulation · stress scenarios")
st.info(
    "Portfolio simulation using synthetic flows. Reserve figures are illustrative outputs, "
    "not treasury guidance or recommendations for a real institution."
)

with st.sidebar:
    st.header("Scenario")
    stress_outflow = st.slider("Outgoing flow multiplier", 1.0, 2.0, 1.0, 0.05)
    stress_inflow = st.slider("Incoming flow multiplier", 0.5, 1.2, 1.0, 0.05)
    uploaded = st.file_uploader("Optional daily-flow CSV", type=["csv"])
    st.caption("Required: date, slot, incoming_m, outgoing_m.")

if uploaded is None:
    flows = make_demo_data()
    is_demo = True
    st.caption("180 synthetic days · 26 settlement windows per day · values in arbitrary millions.")
else:
    try:
        raw = pd.read_csv(uploaded)
        flows, issues = validate_flows(raw)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()
    is_demo = False
    st.caption(f"Loaded {len(flows):,} flow windows across {flows['date'].nunique()} dates.")

if is_demo:
    forecast_source = flows.copy()
    next_date = pd.Timestamp(flows["date"].max()) + pd.Timedelta(days=1)
else:
    forecast_source = add_features(flows)
    next_date = pd.Timestamp(flows["date"].max()) + pd.Timedelta(days=1)

try:
    next_forecast = forecast_day(flows, model, next_date)
except Exception as exc:
    st.error(f"Could not forecast the next day: {exc}")
    st.stop()

try:
    replay, summary = compare_buffers(model, stress_outflow, stress_inflow)
except Exception as exc:
    st.error(f"Could not run the stress replay: {exc}")
    st.stop()

ev = model.evaluation
m1, m2, m3, m4 = st.columns(4)
m1.metric("Holdout window MAE", f"{ev['window_mae_m']:.2f} M")
m2.metric("Holdout RMSE", f"{ev['window_rmse_m']:.2f} M")
m3.metric("90% peak-error coverage", f"{ev['peak_error_90_coverage']:.0%}")
ml_reserve = summary["Forecast reserve"]
fixed_reserve = summary["Fixed reserve"]
m4.metric("Forecast reserve, average", f"{ml_reserve['average_buffer_m']:.1f} M")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Next-day net-flow forecast")
    chart = next_forecast.set_index("slot")[["predicted_net_m"]].rename(
        columns={"predicted_net_m": "Predicted net outflow"}
    )
    st.line_chart(chart, height=300)
    st.caption(f"Forecast date: {next_date.date()} · positive values use liquidity; negative values add liquidity.")
with right:
    st.subheader("Reserve policy replay")
    comparison = pd.DataFrame(summary).T[
        ["average_buffer_m", "days_with_shortfall", "mean_daily_peak_shortfall_m", "shortfall_slots"]
    ].rename(columns={
        "average_buffer_m": "Average reserve (M)",
        "days_with_shortfall": "Shortfall days",
        "mean_daily_peak_shortfall_m": "Mean peak shortfall (M)",
        "shortfall_slots": "Shortfall windows",
    })
    st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)
    st.caption("Replay uses a chronological held-out period; each stress setting is applied to the same days.")

st.subheader("Stress-test comparison")
daily_view = replay.set_index("date")[[
    "actual_peak_need_m", "fixed_buffer_m", "forecast_buffer_m"
]].rename(columns={
    "actual_peak_need_m": "Realized peak need",
    "fixed_buffer_m": "Fixed reserve",
    "forecast_buffer_m": "Forecast reserve",
})
st.line_chart(daily_view, height=300)
st.caption(
    f"Stress setting: outgoing ×{stress_outflow:.2f}, incoming ×{stress_inflow:.2f}. "
    "A reserve shortfall indicates modeled cumulative net outflow exceeded the buffer."
)

with st.expander("Forecast details and limits"):
    st.write("Model features:", ", ".join(FEATURES))
    st.write(
        "The gradient-boosting model forecasts net flow per window using prior observations "
        "for the same window and calendar features. Data is split chronologically into fitting, "
        "calibration, and held-out replay periods."
    )
    st.write(
        "Uploaded data is scored by a model trained on synthetic flows. The demo omits real "
        "settlement rules, collateral, credit lines, funding costs, and institution-specific policy."
    )
