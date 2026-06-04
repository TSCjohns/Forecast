"""
Fulfillment Forecasting Tool
----------------------------
A Streamlit app that uses Facebook Prophet to forecast monthly order/sales
demand from wide-format spreadsheet data (one row per SKU x channel, one
column per month).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from prophet import Prophet

# ----------------------------- Page config ---------------------------------

st.set_page_config(
    page_title="internal forecast tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "sample_sales_data.xlsx"

# ----------------------------- Data loading --------------------------------


@st.cache_data(show_spinner=False)
def load_wide_data(file_source) -> pd.DataFrame:
    """Read the uploaded (or default) wide-format Excel file."""
    return pd.read_excel(file_source)


def reshape_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format (one column per month) to long-format with columns:
    part_number, bottle_size, org, function, ds (date), y (value).
    """
    meta_cols = ["Part # bbSE", "Bottle Size", "ORG", "Function"]
    date_cols = [c for c in df_wide.columns if c not in meta_cols]

    long = df_wide.melt(
        id_vars=meta_cols,
        value_vars=date_cols,
        var_name="ds",
        value_name="y",
    )

    long = long.rename(
        columns={
            "Part # bbSE": "part_number",
            "Bottle Size": "bottle_size",
            "ORG": "org",
            "Function": "function",
        }
    )

    long["ds"] = pd.to_datetime(long["ds"])
    long["y"] = pd.to_numeric(long["y"], errors="coerce")
    long["org"] = long["org"].str.strip().str.rstrip("-")
    long = long.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    return long


def aggregate_series(
    df_long: pd.DataFrame,
    parts: list[str],
    orgs: list[str],
) -> pd.DataFrame:
    """Filter by selected parts/orgs and aggregate monthly totals."""
    mask = df_long["part_number"].isin(parts) & df_long["org"].isin(orgs)
    sub = df_long.loc[mask].copy()
    if sub.empty:
        return sub
    agg = sub.groupby("ds", as_index=False)["y"].sum()
    return agg.sort_values("ds").reset_index(drop=True)


# ----------------------------- Forecasting ---------------------------------


@st.cache_resource(show_spinner=False)
def fit_prophet(
    series: pd.DataFrame,
    yearly: bool,
    weekly: bool,
    growth: str,
    interval_width: float,
    changepoint_prior: float,
    seasonality_prior: float,
    holidays_df: pd.DataFrame | None,
):
    """Fit a Prophet model and cache it for the current parameter set."""
    model = Prophet(
        growth=growth,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior,
        holidays=holidays_df,
    )
    model.fit(series[["ds", "y"]])
    return model


def make_forecast(model: Prophet, periods: int, freq: str = "MS") -> pd.DataFrame:
    """Build the future frame and produce a forecast."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


# ----------------------------- Visualization -------------------------------


def build_forecast_figure(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Plotly chart with history, forecast, and confidence band."""
    split = history["ds"].max()
    fc_future = forecast[forecast["ds"] > split]
    fc_fitted = forecast[forecast["ds"] <= split]

    fig = go.Figure()

    # Confidence band (future only)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([fc_future["ds"], fc_future["ds"][::-1]]),
            y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(15,110,86,0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            name="Confidence interval",
            showlegend=True,
        )
    )

    # Historical actuals
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines+markers",
            line=dict(color="#185FA5", width=2.5),
            marker=dict(size=5),
            name="Actual",
        )
    )

    # Fitted line (history portion of forecast)
    fig.add_trace(
        go.Scatter(
            x=fc_fitted["ds"],
            y=fc_fitted["yhat"],
            mode="lines",
            line=dict(color="#185FA5", width=1, dash="dot"),
            name="Fitted",
            opacity=0.5,
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=fc_future["ds"],
            y=fc_future["yhat"],
            mode="lines+markers",
            line=dict(color="#0F6E56", width=2.5, dash="dash"),
            marker=dict(size=5),
            name="Forecast",
        )
    )

    # Vertical separator at "today"
    fig.add_vline(
        x=split,
        line_dash="dot",
        line_color="rgba(128,128,128,0.4)",
        annotation_text="forecast start",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_components_figure(forecast: pd.DataFrame, has_yearly: bool) -> go.Figure:
    """Trend + yearly seasonality side-by-side."""
    cols = 2 if has_yearly and "yearly" in forecast.columns else 1
    fig = make_subplots(
        rows=1,
        cols=cols,
        subplot_titles=(
            ["Trend", "Yearly seasonality"] if cols == 2 else ["Trend"]
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["trend"],
            mode="lines",
            line=dict(color="#185FA5", width=2),
            name="Trend",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if cols == 2:
        # Show one year of seasonality
        yearly = (
            forecast[["ds", "yearly"]]
            .assign(month=forecast["ds"].dt.month)
            .groupby("month", as_index=False)["yearly"]
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=yearly["month"],
                y=yearly["yearly"],
                mode="lines",
                line=dict(color="#0F6E56", width=2),
                fill="tozeroy",
                fillcolor="rgba(15,110,86,0.1)",
                name="Yearly",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ----------------------------- Sidebar -------------------------------------

st.sidebar.title("internal forecast tool")
st.sidebar.caption("prophet library")

st.sidebar.subheader("1. Data source")
uploaded = st.sidebar.file_uploader(
    "Upload your sales workbook (.xlsx)",
    type=["xlsx"],
    help="Expected: meta columns Part #, Bottle Size, ORG, Function then one column per month.",
)

if uploaded is not None:
    source = uploaded
    source_label = uploaded.name
elif DEFAULT_DATA_PATH.exists():
    source = DEFAULT_DATA_PATH
    source_label = "sample_sales_data.xlsx (bundled)"
else:
    st.sidebar.error("No file uploaded and bundled sample not found.")
    st.stop()

try:
    df_wide = load_wide_data(source)
    df_long = reshape_to_long(df_wide)
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Could not read file: {e}")
    st.stop()

st.sidebar.success(f"Loaded: {source_label}")
st.sidebar.caption(
    f"{df_long['part_number'].nunique()} SKUs · "
    f"{df_long['org'].nunique()} channels · "
    f"{df_long['ds'].min():%b %Y} → {df_long['ds'].max():%b %Y}"
)

# ----- Filters

st.sidebar.subheader("2. What to forecast")

all_parts = sorted(df_long["part_number"].unique().tolist())
all_orgs = sorted(df_long["org"].unique().tolist())
all_sizes = sorted(df_long["bottle_size"].unique().tolist())

view_mode = st.sidebar.radio(
    "View",
    ["Single SKU", "By bottle size", "Total demand"],
    index=0,
    help=(
        "Single SKU: one product at a time. "
        "By bottle size: aggregate every SKU of one bottle size. "
        "Total demand: everything combined."
    ),
)

if view_mode == "Single SKU":
    sel_part = st.sidebar.selectbox("SKU", all_parts, index=0)
    sel_parts = [sel_part]
    scope_descriptor = sel_part
elif view_mode == "By bottle size":
    sel_size = st.sidebar.selectbox("Bottle size", all_sizes, index=0)
    sel_parts = sorted(
        df_long.loc[df_long["bottle_size"] == sel_size, "part_number"]
        .unique()
        .tolist()
    )
    scope_descriptor = f"all {sel_size} SKUs ({len(sel_parts)})"
else:  # Total demand
    sel_parts = all_parts
    scope_descriptor = f"all {len(all_parts)} SKUs"

channel_choice = st.sidebar.radio(
    "Channel",
    ["All channels"] + all_orgs,
    index=0,
)
sel_orgs = all_orgs if channel_choice == "All channels" else [channel_choice]
channel_descriptor = "all channels" if channel_choice == "All channels" else channel_choice

# ----- Model knobs

st.sidebar.subheader("3. Forecast settings")

horizon = st.sidebar.slider("Forecast horizon (months)", 3, 36, 12, step=1)
ci = st.sidebar.slider("Confidence interval", 0.50, 0.95, 0.80, step=0.05)

with st.sidebar.expander("Advanced model controls"):
    growth = st.selectbox("Trend type", ["linear", "flat"], index=0)
    yearly_on = st.checkbox("Yearly seasonality", value=True)
    weekly_on = st.checkbox(
        "Weekly seasonality",
        value=False,
        help="Off by default — your data is monthly.",
    )
    changepoint_prior = st.slider(
        "Trend flexibility",
        0.001,
        0.5,
        0.05,
        step=0.001,
        format="%.3f",
        help="Higher = trend bends more readily.",
    )
    seasonality_prior = st.slider(
        "Seasonality strength",
        0.01,
        20.0,
        10.0,
        step=0.5,
        help="Higher = bigger seasonal swings.",
    )

with st.sidebar.expander("Custom events / promos"):
    st.caption(
        "Add known one-off spikes (e.g. promos, channel launches). "
        "These are treated as 'holidays' by Prophet."
    )
    events_text = st.text_area(
        "One per line: name, YYYY-MM-DD",
        value="",
        placeholder="amazon_prime_day, 2024-07-16\nblack_friday, 2024-11-29",
        height=100,
    )

# Parse events
holidays_df = None
if events_text.strip():
    rows = []
    for line in events_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                rows.append({"holiday": parts[0], "ds": pd.to_datetime(parts[1])})
            except Exception:  # noqa: BLE001
                pass
    if rows:
        holidays_df = pd.DataFrame(rows)
        holidays_df["lower_window"] = 0
        holidays_df["upper_window"] = 1

# ----------------------------- Main page -----------------------------------

st.title("Monthly demand forecast")
st.caption(
    "Historical sales drive a Prophet model that projects forward by your "
    "chosen horizon. Adjust the controls in the sidebar to explore scenarios."
)

if not sel_parts or not sel_orgs:
    st.warning("No SKUs match the current selection. Try a different bottle size or view.")
    st.stop()

series = aggregate_series(df_long, sel_parts, sel_orgs)

if len(series) < 12:
    st.error(
        f"Only {len(series)} months of data after filtering — need at least 12. "
        "Broaden your selection."
    )
    st.stop()

# Fit + forecast
with st.spinner("Fitting Prophet model…"):
    model = fit_prophet(
        series,
        yearly=yearly_on,
        weekly=weekly_on,
        growth=growth,
        interval_width=ci,
        changepoint_prior=changepoint_prior,
        seasonality_prior=seasonality_prior,
        holidays_df=holidays_df,
    )
    forecast = make_forecast(model, periods=horizon, freq="MS")

# ----- KPI strip

hist_avg = series["y"].mean()
hist_total = series["y"].sum()
fc_future = forecast[forecast["ds"] > series["ds"].max()].copy()
fc_avg = fc_future["yhat"].mean()
fc_total = fc_future["yhat"].sum()
growth_pct = (fc_avg - hist_avg) / hist_avg * 100 if hist_avg else 0

scope_label = f"{scope_descriptor} · {channel_descriptor}"
st.markdown(f"**Scope:** {scope_label}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg monthly (hist)", f"{hist_avg:,.0f}")
k2.metric("Total (hist)", f"{hist_total:,.0f}")
k3.metric("Avg monthly (forecast)", f"{fc_avg:,.0f}", f"{growth_pct:+.1f}%")
k4.metric(f"Total next {horizon} mo", f"{fc_total:,.0f}")
k5.metric(
    "Forecast range",
    f"{fc_future['yhat_lower'].sum():,.0f} – {fc_future['yhat_upper'].sum():,.0f}",
    help=f"{int(ci*100)}% confidence interval, summed over the horizon.",
)

# ----- Main chart

fig = build_forecast_figure(
    series, forecast, title=f"{scope_descriptor} ({channel_descriptor}) — monthly units"
)
st.plotly_chart(fig, use_container_width=True)

# ----- Components

st.subheader("What's driving the forecast")
comp_fig = build_components_figure(forecast, has_yearly=yearly_on)
st.plotly_chart(comp_fig, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        """
- **Trend** — the underlying growth or decline once seasonality is stripped out.
- **Yearly seasonality** — the typical month-by-month pattern. Peaks tell you
  which months historically over-perform the trend; troughs under-perform.
- **Confidence interval** widens with horizon: short-term forecasts are tight,
  longer-term ones carry more uncertainty.
"""
    )

# ----- Forecast table + download

st.subheader("Forecast detail")
table = fc_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
table.columns = ["Month", "Forecast", "Lower bound", "Upper bound"]
table["Month"] = table["Month"].dt.strftime("%b %Y")
for c in ["Forecast", "Lower bound", "Upper bound"]:
    table[c] = table[c].round(0).astype(int)

st.dataframe(table, use_container_width=True, hide_index=True)

# Build Excel download
@st.cache_data
def to_excel_bytes(history: pd.DataFrame, forecast_table: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        history.rename(columns={"ds": "Month", "y": "Actual"}).to_excel(
            writer, sheet_name="History", index=False
        )
        forecast_table.to_excel(writer, sheet_name="Forecast", index=False)
    return buf.getvalue()


csv_bytes = table.to_csv(index=False).encode("utf-8")
xlsx_bytes = to_excel_bytes(series, table)

d1, d2 = st.columns(2)
d1.download_button(
    "⬇ Download forecast (CSV)",
    data=csv_bytes,
    file_name=f"forecast_{datetime.now():%Y%m%d}.csv",
    mime="text/csv",
    use_container_width=True,
)
d2.download_button(
    "⬇ Download forecast + history (XLSX)",
    data=xlsx_bytes,
    file_name=f"forecast_{datetime.now():%Y%m%d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# ----- Footer

st.divider()
with st.expander("about prophet"):
    st.markdown(
        """
This forecasting tool uses Prophet

Prophet is a procedure for forecasting time series data based on an additive model 
where non-linear trends are fit with yearly, weekly, and daily seasonality, 
plus holiday effects. It works best with time series that have strong seasonal 
effects and several seasons of historical data. Prophet is robust to missing 
data and shifts in the trend, and typically handles outliers well.

Prophet is open source software released by Facebook's Core Data Science team .

Full documentation and examples available at the homepage: https://facebook.github.io/prophet/

Caveats:
- The model assumes the future looks roughly like the past. Step-changes
  (new channel, new product line, major supply disruption) need to be
  reflected manually via the events panel or by re-fitting on relevant data.
- Confidence intervals widen quickly past ~12 months
"""
    )
