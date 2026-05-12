"""
Streamlit dashboard: Infrabel train-delay analysis on the Belgian network.

Tabs:
  - Map         : current-state map (stations + lines) + ranking + hour evolution
  - Patterns    : recurring-delay heatmap (weekday x hour)
  - Animation   : animated 24h map (Plotly)
  - Trains      : top recurring late train numbers

Sidebar:
  - Date range (year/month-agnostic, multi-month supported)
  - Weekday filter (multi-select)
  - Exclude Belgian public holidays
  - Hour-of-day slider (for map + ranking)
  - Display toggles (stations / lines, ranking mode)
  - Cache info + force-refresh

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from mobilitytwin.api import (
    cached_days,
    fetch_punctuality_range,
)
from mobilitytwin.pipeline import (
    WEEKDAY_ORDER,
    aggregate_line_hour,
    aggregate_station_hour,
    aggregate_weekday_hour,
    filter_by_weekdays,
    filter_holidays,
    join_lines,
    join_stations,
    load_line_sections,
    load_operational_points,
    records_to_dataframe,
    top_recurring_late_trains,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_VIEW = pdk.ViewState(latitude=50.85, longitude=4.55, zoom=7.5, pitch=0)

COLOR_LOW = np.array([46, 204, 113])
COLOR_MID = np.array([241, 196, 15])
COLOR_HIGH = np.array([192, 57, 43])
DELAY_SATURATION_MIN = 5.0


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading station geometries...")
def cached_operational_points():
    return load_operational_points()


@st.cache_resource(show_spinner="Loading line geometries (dissolve)...")
def cached_line_sections():
    return load_line_sections()


@st.cache_data(show_spinner=False)
def load_range_dataframe(
    start: date, end: date, refresh_token: int = 0
) -> tuple[pd.DataFrame, list[date]]:
    """Fetch + clean a date range. refresh_token busts the cache on demand."""
    progress = st.progress(0.0, text=f"Fetching {start} → {end} ...")

    def _on_progress(done: int, total: int, day: date) -> None:
        progress.progress(done / total, text=f"Fetched {day} ({done}/{total})")

    records, failed = fetch_punctuality_range(
        start, end, on_progress=_on_progress, force=bool(refresh_token < 0)
    )
    progress.empty()
    return records_to_dataframe(records), failed


# ---------------------------------------------------------------------------
# Color & geometry helpers
# ---------------------------------------------------------------------------
def delay_to_rgb(delay_min: float) -> list[int]:
    if pd.isna(delay_min):
        return [180, 180, 180]
    d = max(0.0, float(delay_min))
    t = min(d / DELAY_SATURATION_MIN, 1.0)
    if t < 0.5:
        rgb = COLOR_LOW + (COLOR_MID - COLOR_LOW) * (t / 0.5)
    else:
        rgb = COLOR_MID + (COLOR_HIGH - COLOR_MID) * ((t - 0.5) / 0.5)
    return [int(round(x)) for x in rgb]


def geometry_to_paths(geom) -> list[list[list[float]]]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[[x, y] for x, y, *_ in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[[x, y] for x, y, *_ in line.coords] for line in geom.geoms]
    return []


# ---------------------------------------------------------------------------
# Map layers
# ---------------------------------------------------------------------------
def build_segment_layer(segments: gpd.GeoDataFrame, hour: int) -> pdk.Layer:
    snap = segments[segments["hour"] == hour].copy()
    snap["color"] = snap["mean_delay_min"].apply(delay_to_rgb)
    rows = []
    for _, r in snap.iterrows():
        for path in geometry_to_paths(r.geometry):
            rows.append({
                "path": path,
                "color": r["color"],
                "line_no": str(r["line_no"]),
                "mean_delay_min": float(r["mean_delay_min"]),
                "n_observations": int(r["n_observations"]),
            })
    return pdk.Layer(
        "PathLayer",
        data=pd.DataFrame(rows),
        get_path="path",
        get_color="color",
        get_width=40,
        width_min_pixels=2,
        pickable=True,
    )


def build_station_layer(stations: gpd.GeoDataFrame, hour: int) -> pdk.Layer:
    snap = stations[stations["hour"] == hour].copy()
    snap["color"] = snap["mean_delay_min"].apply(delay_to_rgb)
    snap["radius"] = (snap["n_observations"].clip(lower=1)).pow(0.5) * 25
    df = snap[[
        "lon", "lat", "color", "radius",
        "station_label", "mean_delay_min", "n_observations",
    ]]
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=20,
        pickable=True,
        opacity=0.85,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_top_worst(
    stations: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    hour: int,
    mode: str,
    min_observations: int,
):
    if mode == "Stations":
        snap = stations[stations["hour"] == hour]
        snap = snap[snap["n_observations"] >= min_observations]
        snap = snap.sort_values("mean_delay_min", ascending=False).head(10).sort_values("mean_delay_min")
        y = "station_label"
        title = f"10 worst stations at {hour:02d}:00  (≥ {min_observations} obs.)"
    else:
        snap = segments[segments["hour"] == hour]
        snap = snap[snap["n_observations"] >= min_observations]
        snap = (
            snap.assign(line_label=lambda d: "Line " + d["line_no"].astype(str))
            .sort_values("mean_delay_min", ascending=False).head(10)
            .sort_values("mean_delay_min")
        )
        y = "line_label"
        title = f"10 worst lines at {hour:02d}:00  (≥ {min_observations} obs.)"

    if snap.empty:
        return px.bar(title=f"No data at {hour:02d}:00")

    fig = px.bar(
        snap, x="mean_delay_min", y=y, orientation="h",
        color="mean_delay_min",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        labels={"mean_delay_min": "Mean delay (min)", y: ""},
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def chart_hourly_evolution(stations: gpd.GeoDataFrame, hour: int):
    if stations.empty:
        return px.line(title="No data")
    hourly = (
        stations.groupby("hour")
        .apply(
            lambda g: np.average(g["mean_delay_min"], weights=g["n_observations"])
        )
        .reset_index(name="mean_delay_min")
    )
    fig = px.line(
        hourly, x="hour", y="mean_delay_min", markers=True,
        title="Global mean delay across the day",
        labels={"hour": "Hour", "mean_delay_min": "Mean delay (min)"},
    )
    fig.add_vline(x=hour, line_dash="dash", line_color="#c0392b")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def chart_weekday_heatmap(weekday_hour: pd.DataFrame):
    if weekday_hour.empty:
        return px.imshow([[0]], title="No data")
    pivot = (
        weekday_hour.pivot(index="weekday", columns="hour", values="mean_delay_min")
        .reindex(WEEKDAY_ORDER)
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        labels=dict(x="Hour of day", y="Weekday", color="Mean delay (min)"),
        title="Recurring delay patterns — mean arrival delay (min) by weekday × hour",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def chart_animated_map(stations: gpd.GeoDataFrame):
    if stations.empty:
        return px.scatter_mapbox(title="No data")
    df = stations.copy()
    df["mean_delay_min_clipped"] = df["mean_delay_min"].clip(
        lower=-2, upper=DELAY_SATURATION_MIN
    )
    df["size"] = df["n_observations"].clip(lower=1).pow(0.5)
    fig = px.scatter_mapbox(
        df.sort_values("hour"),
        lat="lat", lon="lon",
        color="mean_delay_min_clipped",
        size="size",
        hover_name="station_label",
        hover_data={
            "mean_delay_min": ":.2f",
            "n_observations": True,
            "size": False,
            "mean_delay_min_clipped": False,
            "lat": False, "lon": False,
        },
        animation_frame="hour",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        range_color=(-2, DELAY_SATURATION_MIN),
        zoom=7, height=650,
        mapbox_style="open-street-map",
        title="24-hour propagation of delays across the network",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PRESETS: list[tuple[str, callable]] = [
    ("Today",         lambda t: (t, t)),
    ("Last 7 days",   lambda t: (t - timedelta(days=6), t)),
    ("Last 30 days",  lambda t: (t - timedelta(days=29), t)),
    ("This month",    lambda t: (t.replace(day=1), t)),
    ("Last 3 months", lambda t: (t - timedelta(days=89), t)),
    ("Year-to-date",  lambda t: (date(t.year, 1, 1), t)),
]


def _apply_preset(start_val: date, end_val: date) -> None:
    """Streamlit button callback — runs BEFORE date_input widgets render."""
    st.session_state.start_date = start_val
    st.session_state.end_date = end_val


def render_sidebar() -> dict:
    today = date.today()
    if "start_date" not in st.session_state:
        st.session_state.start_date = today.replace(day=1)
    if "end_date" not in st.session_state:
        st.session_state.end_date = today

    with st.sidebar:
        st.header("📅 Period")

        st.markdown("**Quick presets**")
        cols = st.columns(2)
        for i, (label, fn) in enumerate(PRESETS):
            s, e = fn(today)
            cols[i % 2].button(
                label,
                key=f"preset_{i}",
                use_container_width=True,
                on_click=_apply_preset,
                args=(s, e),
            )

        st.markdown("**Custom range**")
        start = st.date_input(
            "Start date",
            key="start_date",
            min_value=date(2023, 1, 1),
            max_value=today,
            format="YYYY-MM-DD",
        )
        end = st.date_input(
            "End date",
            key="end_date",
            min_value=date(2023, 1, 1),
            max_value=today,
            format="YYYY-MM-DD",
        )

        if start > end:
            st.error("⚠️ Start date must be before end date.")
            st.stop()

        days_total = (end - start).days + 1
        cached_set = set(cached_days())
        cached_in_range = sum(
            1 for i in range(days_total) if (start + timedelta(days=i)) in cached_set
        )
        to_fetch = days_total - cached_in_range
        if to_fetch == 0:
            st.success(f"✅ {days_total} day(s) — all cached, instant load")
        elif to_fetch <= 7:
            st.info(
                f"📅 {days_total} day(s) · {cached_in_range} cached · "
                f"{to_fetch} to fetch (~{to_fetch}s)"
            )
        else:
            st.warning(
                f"📅 **{days_total} day(s)** · {cached_in_range} cached · "
                f"**{to_fetch} to fetch** (~{to_fetch}s, please wait)"
            )

        st.divider()
        st.header("🔎 Filters")
        weekdays = st.multiselect(
            "Weekdays", WEEKDAY_ORDER, default=WEEKDAY_ORDER,
        )
        exclude_holidays = st.checkbox(
            "Exclude Belgian public holidays",
            value=False,
            help="Use the `holidays` package (country=BE).",
        )
        hour = st.slider("Hour of day (map / ranking)", 0, 23, 8)
        min_obs = st.number_input(
            "Min observations (ranking)", min_value=1, max_value=500, value=10,
            help="Hide stations/lines with too few data points in the top-10.",
        )

        st.divider()
        st.header("🎨 Display")
        show_stations = st.checkbox("Show stations", value=True)
        show_lines = st.checkbox("Show lines", value=True)
        mode = st.radio("Ranking", ["Stations", "Lines"], horizontal=True)

        st.divider()
        with st.expander("💾 Cache", expanded=False):
            n_cached = len(cached_days())
            st.caption(f"{n_cached} day(s) cached on disk in `data/raw/api_cache/`.")
            if st.button("🔄 Force refresh (re-download)", use_container_width=True):
                st.session_state.refresh_token = (
                    st.session_state.get("refresh_token", 0) + 1
                )
                st.cache_data.clear()
                st.rerun()

    return dict(
        start=start, end=end,
        weekdays=weekdays,
        exclude_holidays=exclude_holidays,
        hour=hour, min_obs=int(min_obs),
        show_stations=show_stations, show_lines=show_lines,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_map_tab(stations, segments, ctx: dict, df_filtered: pd.DataFrame):
    hour = ctx["hour"]

    st_snap = stations[stations["hour"] == hour]
    sg_snap = segments[segments["hour"] == hour]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records (filtered)", f"{len(df_filtered):,}")
    c2.metric("Stations @ hour", f"{len(st_snap):,}")
    c3.metric("Lines @ hour", f"{sg_snap['line_no'].nunique():,}")
    avg = st_snap["mean_delay_min"].mean() if len(st_snap) else float("nan")
    c4.metric(
        f"Avg delay @ {hour:02d}:00",
        "—" if pd.isna(avg) else f"{avg:.2f} min",
    )

    layers = []
    if ctx["show_lines"]:
        layers.append(build_segment_layer(segments, hour))
    if ctx["show_stations"]:
        layers.append(build_station_layer(stations, hour))

    tooltip = {
        "html": (
            "<b>{station_label}{line_no}</b><br/>"
            "Mean delay: {mean_delay_min} min<br/>"
            "Observations: {n_observations}"
        ),
        "style": {"backgroundColor": "#1f1f1f", "color": "white"},
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=DEFAULT_VIEW,
            map_style="light",
            tooltip=tooltip,
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            chart_top_worst(stations, segments, hour, ctx["mode"], ctx["min_obs"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            chart_hourly_evolution(stations, hour),
            use_container_width=True,
        )


def render_patterns_tab(df_filtered: pd.DataFrame):
    st.markdown(
        "Aggregate view across the whole selected period — useful to spot "
        "**recurring** weekly patterns (rush hour vs night, weekday vs Sunday, etc.)."
    )
    wh = aggregate_weekday_hour(df_filtered)
    st.plotly_chart(chart_weekday_heatmap(wh), use_container_width=True)

    if not df_filtered.empty:
        late = df_filtered.dropna(subset=["delay_arr_min"]).copy()
        late["is_late_5"] = late["delay_arr_min"] >= 5
        st.markdown("**Daily late-arrival count (≥ 5 min)**")
        daily = (
            late.groupby(late["date"].dt.date)["is_late_5"]
            .sum()
            .reset_index(name="late_arrivals")
        )
        fig = px.bar(
            daily, x="date", y="late_arrivals",
            labels={"date": "Day", "late_arrivals": "Late arrivals (≥5 min)"},
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_animation_tab(stations: gpd.GeoDataFrame):
    st.markdown(
        "Press ▶ to watch how delays build up and unwind across the 24 hours "
        "(averaged over the selected period)."
    )
    st.plotly_chart(chart_animated_map(stations), use_container_width=True)


def render_trains_tab(df_filtered: pd.DataFrame):
    st.markdown(
        "Trains whose arrivals are recurringly late on the selected period. "
        "Useful to flag timetable inputs that may need adjustment."
    )
    threshold = st.slider(
        "Late threshold (min)", min_value=1, max_value=15, value=5,
    )
    top_n = st.slider("Top N", min_value=5, max_value=50, value=20)
    table = top_recurring_late_trains(df_filtered, threshold_min=threshold, top_n=top_n)
    if table.empty:
        st.info("No data.")
        return
    st.dataframe(table, use_container_width=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Infrabel — Delay analysis", layout="wide")
    st.title("Infrabel — Belgian rail delay dashboard")
    st.caption(
        "Live data from mobilitytwin.brussels — filter by date range, weekday, "
        "holiday, and hour of day."
    )

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0

    ctx = render_sidebar()

    # ----- Fetch + clean -----
    df_month, failed_days = load_range_dataframe(
        ctx["start"], ctx["end"], st.session_state.refresh_token
    )

    if failed_days:
        st.warning(
            f"⚠️ {len(failed_days)} day(s) could not be fetched after retries: "
            f"{[d.isoformat() for d in failed_days[:5]]}"
            f"{' ...' if len(failed_days) > 5 else ''}. "
            f"Click **🔄 Force refresh** in the sidebar to retry."
        )

    # ----- Apply in-memory filters -----
    df = filter_by_weekdays(df_month, ctx["weekdays"])
    df = filter_holidays(df, ctx["exclude_holidays"])

    if df.empty:
        st.warning(
            "No records left after weekday/holiday filters. Loosen the filters."
        )
        st.stop()

    stations_agg = aggregate_station_hour(df)
    lines_agg = aggregate_line_hour(df)
    stations = join_stations(stations_agg, cached_operational_points())
    segments = join_lines(lines_agg, cached_line_sections())

    # ----- Tabs -----
    tab_map, tab_patterns, tab_anim, tab_trains = st.tabs(
        ["🗺️ Map", "📊 Patterns", "🎬 Animation", "🚆 Trains"]
    )
    with tab_map:
        render_map_tab(stations, segments, ctx, df)
    with tab_patterns:
        render_patterns_tab(df)
    with tab_anim:
        render_animation_tab(stations)
    with tab_trains:
        render_trains_tab(df)


if __name__ == "__main__":
    main()
