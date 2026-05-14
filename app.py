"""
Streamlit dashboard: Infrabel train-delay analysis on the Belgian network.

Tabs
  - Map        : geographic state (stations + lines) at one hour + ranking
  - Patterns   : recurring-delay heatmap (weekday x hour) + daily late counts
  - Animation  : animated 24h map
  - Drill-down : single-entity detail — pick a station, a train, or a relation
  - Trains     : top recurring late train numbers

Sidebar
  - Date range (presets + custom start/end)
  - Weekday filter, holiday exclusion
  - Stations / Relations multiselect (dynamic from fetched data)
  - Hour-of-day slider, min observations
  - Display toggles + ranking mode (Stations / Lines / Relations)

Run
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datetime import date, timedelta

import geopandas as gpd
import holidays
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

_BE_HOLIDAYS = holidays.country_holidays("BE")

from mobilitytwin.api import (
    cached_days,
    fetch_punctuality_range,
)
from mobilitytwin.pipeline import (
    WEEKDAY_ORDER,
    aggregate_line_hour,
    aggregate_relation_hour,
    aggregate_station_hour,
    aggregate_weekday_hour,
    available_relations,
    available_stations,
    available_train_nos,
    filter_by_relations,
    filter_by_stations,
    filter_by_weekdays,
    filter_holidays,
    join_lines,
    join_stations,
    load_line_sections,
    load_operational_points,
    records_to_dataframe,
    relation_hourly_profile,
    station_hourly_profile,
    station_top_trains,
    top_recurring_late_trains,
    train_journey_profile,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_VIEW = pdk.ViewState(latitude=50.85, longitude=4.55, zoom=7.5, pitch=0)

COLOR_LOW = np.array([46, 204, 113])
COLOR_MID = np.array([241, 196, 15])
COLOR_HIGH = np.array([192, 57, 43])
DELAY_SATURATION_MIN = 5.0

PRESETS: list[tuple[str, callable]] = [
    ("Today",         lambda t: (t, t)),
    ("Last 7 days",   lambda t: (t - timedelta(days=6), t)),
    ("Last 30 days",  lambda t: (t - timedelta(days=29), t)),
    ("This month",    lambda t: (t.replace(day=1), t)),
    ("Last 3 months", lambda t: (t - timedelta(days=89), t)),
    ("Year-to-date",  lambda t: (date(t.year, 1, 1), t)),
]


def _apply_preset(start_val: date, end_val: date) -> None:
    st.session_state.start_date = start_val
    st.session_state.end_date = end_val


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
) -> tuple[pd.DataFrame, list[date], dict[date, str]]:
    progress = st.progress(0.0, text=f"Fetching {start} → {end} ...")

    def _on_progress(done: int, total: int, day: date) -> None:
        progress.progress(done / total, text=f"Fetched {day} ({done}/{total})")

    fetch_errors: dict[date, str] = {}
    records, failed = fetch_punctuality_range(
        start,
        end,
        on_progress=_on_progress,
        force=bool(refresh_token < 0),
        failed_errors=fetch_errors,
    )
    progress.empty()
    return records_to_dataframe(records), failed, fetch_errors


# ---------------------------------------------------------------------------
# Period description helpers
# ---------------------------------------------------------------------------
def period_caption(ctx: dict) -> str:
    """Compact one-line caption, used right above each chart."""
    days = (ctx["end"] - ctx["start"]).days + 1
    bits = [
        f"📅 {ctx['start'].strftime('%b %d, %Y')} → {ctx['end'].strftime('%b %d, %Y')}",
        f"{days} day{'s' if days != 1 else ''}",
    ]
    if 0 < len(ctx["weekdays"]) < 7:
        bits.append(", ".join(d[:3] for d in ctx["weekdays"]))
    if ctx["exclude_holidays"]:
        bits.append("BE holidays excluded")
    if ctx.get("relations_filter"):
        rels = ctx["relations_filter"]
        head = ", ".join(rels[:3])
        more = f" +{len(rels) - 3}" if len(rels) > 3 else ""
        bits.append(f"relations: {head}{more}")
    if ctx.get("stations_filter"):
        bits.append(f"{len(ctx['stations_filter'])} station(s) selected")
    return " · ".join(bits)


def describe_period_block(ctx: dict, n_records: int) -> None:
    """Big descriptive header at the top of the app — sets the scene."""
    days = (ctx["end"] - ctx["start"]).days + 1
    parts = [
        f"**{ctx['start'].strftime('%b %d, %Y')} → "
        f"{ctx['end'].strftime('%b %d, %Y')}** "
        f"({days} day{'s' if days != 1 else ''})",
    ]
    wd = ctx["weekdays"]
    if 0 < len(wd) < 7:
        parts.append(f"weekdays: {', '.join(d[:3] for d in wd)}")
    else:
        parts.append("all weekdays")
    parts.append(
        "🎉 holidays excluded" if ctx["exclude_holidays"] else "holidays included"
    )
    if ctx["stations_filter"]:
        parts.append(f"📍 {len(ctx['stations_filter'])} station(s)")
    if ctx["relations_filter"]:
        parts.append(f"🚆 {len(ctx['relations_filter'])} relation(s)")
    parts.append(f"**{n_records:,}** train arrivals analyzed")
    st.info(" · ".join(parts))


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
    relations_hour: pd.DataFrame,
    hour: int,
    mode: str,
    min_observations: int,
    n_days: int,
):
    avg_suffix = (
        f"averaged over {n_days} day{'s' if n_days != 1 else ''}"
    )
    if mode == "Stations":
        snap = stations[stations["hour"] == hour]
        snap = snap[snap["n_observations"] >= min_observations]
        snap = (
            snap.sort_values("mean_delay_min", ascending=False).head(10)
            .sort_values("mean_delay_min")
        )
        y, label = "station_label", "Station"
        title = (
            f"10 worst {label.lower()}s at {hour:02d}:00–{hour:02d}:59"
            f" — {avg_suffix}"
        )
    elif mode == "Lines":
        snap = segments[segments["hour"] == hour]
        snap = snap[snap["n_observations"] >= min_observations]
        snap = (
            snap.assign(line_label=lambda d: "Line " + d["line_no"].astype(str))
            .sort_values("mean_delay_min", ascending=False).head(10)
            .sort_values("mean_delay_min")
        )
        y, label = "line_label", "Rail line"
        title = (
            f"10 worst rail lines at {hour:02d}:00–{hour:02d}:59"
            f" — {avg_suffix}"
        )
    else:  # Relations
        snap = relations_hour[relations_hour["hour"] == hour]
        snap = snap[snap["n_observations"] >= min_observations]
        snap = (
            snap.sort_values("mean_delay_min", ascending=False).head(10)
            .sort_values("mean_delay_min")
        )
        y, label = "relation", "Relation"
        title = (
            f"10 worst IC/S/L relations at {hour:02d}:00–{hour:02d}:59"
            f" — {avg_suffix}"
        )

    if snap.empty:
        return px.bar(title=f"{title} — no data (try lowering 'min observations')")

    fig = px.bar(
        snap, x="mean_delay_min", y=y, orientation="h",
        color="mean_delay_min",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        labels={"mean_delay_min": "Mean delay (min)", y: ""},
        title=title,
    )
    fig.update_layout(
        coloraxis_showscale=False, height=420, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def chart_hourly_evolution(stations: gpd.GeoDataFrame, hour: int, n_days: int):
    if stations.empty:
        return px.line(title="No data")
    hourly = (
        stations.groupby("hour")
        .apply(
            lambda g: np.average(g["mean_delay_min"], weights=g["n_observations"])
        )
        .reset_index(name="mean_delay_min")
    )
    avg_suffix = f"averaged over {n_days} day{'s' if n_days != 1 else ''}"
    fig = px.line(
        hourly, x="hour", y="mean_delay_min", markers=True,
        title=f"Network-wide average delay by hour — {avg_suffix}",
        labels={"hour": "Hour of day", "mean_delay_min": "Mean delay (min)"},
    )
    fig.add_vline(x=hour, line_dash="dash", line_color="#c0392b",
                  annotation_text=f"Selected: {hour:02d}:00", annotation_position="top")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def chart_weekday_heatmap(weekday_hour: pd.DataFrame):
    """Heatmap weekday x hour. NaN cells (no records) are drawn light grey.

    Hover exposes both the mean delay AND the number of observations,
    so users can tell apart 'green = on time' vs 'grey = no data at all'.
    """
    if weekday_hour.empty:
        return px.imshow([[0]], title="No data")

    pivot_delay = (
        weekday_hour.pivot(index="weekday", columns="hour", values="mean_delay_min")
        .reindex(WEEKDAY_ORDER)
        .reindex(columns=range(24))
    )
    pivot_obs = (
        weekday_hour.pivot(index="weekday", columns="hour", values="n_observations")
        .reindex(WEEKDAY_ORDER)
        .reindex(columns=range(24))
        .fillna(0)
        .astype(int)
    )

    fig = go.Figure(
        go.Heatmap(
            x=list(range(24)),
            y=WEEKDAY_ORDER,
            z=pivot_delay.values,
            customdata=pivot_obs.values,
            colorscale=[[0, "#2ecc71"], [0.5, "#f1c40f"], [1, "#c0392b"]],
            zmin=0,
            zmax=10,
            colorbar=dict(title="Mean delay<br>(min)"),
            hovertemplate=(
                "Hour: %{x}:00<br>"
                "Weekday: %{y}<br>"
                "Mean delay: %{z:.2f} min<br>"
                "Observations: %{customdata}<extra></extra>"
            ),
            hoverongaps=False,  # don't show hover for NaN cells
        )
    )
    fig.update_layout(
        title="Recurring delay patterns — weekday × hour "
              "(grey = no train arrivals in that bucket)",
        xaxis_title="Hour of day",
        yaxis_title="Weekday",
        xaxis=dict(dtick=1),
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="#d0d0d0",  # NaN cells fall through to this grey
    )
    return fig


def chart_animated_map(stations: gpd.GeoDataFrame, n_days: int):
    if stations.empty:
        return px.scatter_mapbox(title="No data")
    df = stations.copy()
    df["mean_delay_min_clipped"] = df["mean_delay_min"].clip(
        lower=-2, upper=DELAY_SATURATION_MIN
    )
    df["size"] = df["n_observations"].clip(lower=1).pow(0.5)
    avg_suffix = f"averaged over {n_days} day{'s' if n_days != 1 else ''}"
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
        title=f"24-hour propagation of delays — {avg_suffix}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_period_sidebar() -> tuple[date, date]:
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
                label, key=f"preset_{i}", use_container_width=True,
                on_click=_apply_preset, args=(s, e),
            )

        st.markdown("**Custom range**")
        start = st.date_input(
            "Start date", key="start_date",
            min_value=date(2023, 1, 1), max_value=today,
            format="YYYY-MM-DD",
        )
        end = st.date_input(
            "End date", key="end_date",
            min_value=date(2023, 1, 1), max_value=today,
            format="YYYY-MM-DD",
        )

        if start > end:
            st.error("⚠️ Start date must be before end date.")
            st.stop()

        days_total = (end - start).days + 1
        cached_set = set(cached_days())
        cached_in_range = sum(
            1 for i in range(days_total)
            if (start + timedelta(days=i)) in cached_set
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
    return start, end


def render_filters_sidebar(df_raw: pd.DataFrame) -> dict:
    """Render the filter section AFTER data fetch, with dynamic options."""
    stations_opts = available_stations(df_raw)
    relations_opts = available_relations(df_raw)

    with st.sidebar:
        st.divider()
        st.header("🔎 Filters")

        weekdays = st.multiselect("Weekdays", WEEKDAY_ORDER, default=WEEKDAY_ORDER)
        exclude_holidays = st.checkbox(
            "Exclude Belgian public holidays", value=False,
        )
        stations_filter = st.multiselect(
            f"Stations ({len(stations_opts)} available)",
            options=stations_opts, default=[],
            help="Empty = all stations.",
        )
        relations_filter = st.multiselect(
            f"Relations / IC routes ({len(relations_opts)} available)",
            options=relations_opts, default=[],
            help="Empty = all relations. Examples: IC 14-1, IC 18, S5-1, L.",
        )
        hour = st.slider("Hour of day (map / ranking)", 0, 23, 8)
        min_obs = st.number_input(
            "Min observations (ranking)",
            min_value=1, max_value=500, value=10,
            help="Hide entities with too few data points in the top-10.",
        )

        st.divider()
        st.header("🎨 Display")
        show_stations = st.checkbox("Show stations", value=True)
        show_lines = st.checkbox("Show lines", value=True)
        mode = st.radio(
            "Ranking by",
            ["Stations", "Lines", "Relations"],
            horizontal=True,
            help="Lines = physical Infrabel track segments. "
                 "Relations = commercial routes (IC, S, L).",
        )

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
        weekdays=weekdays,
        exclude_holidays=exclude_holidays,
        stations_filter=stations_filter,
        relations_filter=relations_filter,
        hour=hour, min_obs=int(min_obs),
        show_stations=show_stations, show_lines=show_lines,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_map_tab(
    stations: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    relations_hour: pd.DataFrame,
    ctx: dict,
    df_filtered: pd.DataFrame,
) -> None:
    hour = ctx["hour"]
    n_days = (ctx["end"] - ctx["start"]).days + 1

    st.subheader(
        f"🗺️ Network state at {hour:02d}:00–{hour:02d}:59 "
        f"— averaged over {n_days} day{'s' if n_days != 1 else ''}"
    )
    st.caption(period_caption(ctx))
    st.caption(
        "ℹ️ Each station/line is colored by the **mean arrival delay across "
        f"every {hour:02d}:XX arrival** within the selected period — not a "
        "single-day snapshot."
    )

    st_snap = stations[stations["hour"] == hour]
    sg_snap = segments[segments["hour"] == hour]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arrivals analyzed", f"{len(df_filtered):,}")
    c2.metric(f"Stations @ {hour:02d}:00", f"{len(st_snap):,}")
    c3.metric(f"Lines @ {hour:02d}:00", f"{sg_snap['line_no'].nunique():,}")
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
        st.markdown(f"##### 🏆 Top 10 — ranking by `{ctx['mode']}`")
        st.caption(period_caption(ctx))
        st.caption(
            f"Each bar = mean delay during the **{hour:02d}:00–{hour:02d}:59** "
            f"hour window, averaged across the **{n_days} day"
            f"{'s' if n_days != 1 else ''}** of the selected period."
        )
        st.plotly_chart(
            chart_top_worst(
                stations, segments, relations_hour,
                hour, ctx["mode"], ctx["min_obs"], n_days,
            ),
            use_container_width=True,
        )
    with right:
        st.markdown("##### 🕐 Average delay across the 24 hours")
        st.caption(period_caption(ctx))
        st.caption(
            f"Each point = mean delay during one hour window, averaged across "
            f"the **{n_days} day{'s' if n_days != 1 else ''}** of the period."
        )
        st.plotly_chart(
            chart_hourly_evolution(stations, hour, n_days),
            use_container_width=True,
        )


def render_patterns_tab(df_filtered: pd.DataFrame, ctx: dict) -> None:
    n_days = (ctx["end"] - ctx["start"]).days + 1
    st.subheader(
        f"📊 Recurring weekly patterns — pooled across {n_days} "
        f"day{'s' if n_days != 1 else ''}"
    )
    st.caption(period_caption(ctx))
    st.markdown(
        f"Each cell of the heatmap pools every arrival of that (weekday, hour) "
        f"combination found within the **{n_days} day"
        f"{'s' if n_days != 1 else ''}** of the selected period — useful to "
        "spot **recurring** weekly patterns (rush hour vs night, "
        "weekday vs Sunday, etc.)."
    )

    wh = aggregate_weekday_hour(df_filtered)
    st.plotly_chart(chart_weekday_heatmap(wh), use_container_width=True)

    if df_filtered.empty:
        return

    st.markdown("##### 📉 Daily late-arrival count (≥ 5 min)")
    st.caption(period_caption(ctx))

    # Trains that depart late at night spill their `datdep` onto the previous
    # calendar day — clip to the user's period so the bar chart starts at the
    # selected start date.
    late = df_filtered.dropna(subset=["delay_arr_min", "date"]).copy()
    late["date_only"] = late["date"].dt.date
    late = late[
        (late["date_only"] >= ctx["start"]) & (late["date_only"] <= ctx["end"])
    ]
    late["is_late_5"] = late["delay_arr_min"] >= 5

    # Reindex with every day in the period so missing days are still visible
    # as zero-height bars, and color-code WHY a day is zero:
    #   - "in selection"      : day passes all filters AND was fetched
    #   - "weekday filtered"  : day's weekday excluded by sidebar
    #   - "holiday filtered"  : Belgian public holiday + 'Exclude holidays' on
    #   - "not fetched"       : API fetch failed (or day never cached)
    full_range = pd.date_range(ctx["start"], ctx["end"], freq="D").date
    counts = late.groupby("date_only")["is_late_5"].sum()
    cached_set = set(cached_days())

    def _status(d) -> str:
        if d not in cached_set:
            return "⚠️ not fetched"
        if d.strftime("%A") not in ctx["weekdays"]:
            return "🚫 weekday filtered"
        if ctx["exclude_holidays"] and d in _BE_HOLIDAYS:
            return "🎉 holiday filtered"
        return "✅ in selection"

    daily = pd.DataFrame({
        "date": full_range,
        "late_arrivals": [int(counts.get(d, 0)) for d in full_range],
        "status": [_status(d) for d in full_range],
    })

    fig = px.bar(
        daily, x="date", y="late_arrivals", color="status",
        color_discrete_map={
            "✅ in selection":     "#3498db",
            "🚫 weekday filtered": "#dcdcdc",
            "🎉 holiday filtered": "#f5d76e",
            "⚠️ not fetched":      "#7f8c8d",
        },
        labels={"date": "Day", "late_arrivals": "Late arrivals (≥5 min)"},
        hover_data={"status": True},
    )
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_animation_tab(stations: gpd.GeoDataFrame, ctx: dict) -> None:
    n_days = (ctx["end"] - ctx["start"]).days + 1
    st.subheader(
        f"🎬 24-hour delay propagation — averaged over {n_days} "
        f"day{'s' if n_days != 1 else ''}"
    )
    st.caption(period_caption(ctx))
    st.markdown(
        "Press ▶ to watch how delays build up and unwind across the 24 hours. "
        "Each frame shows the **mean delay per station during one hour of the "
        f"day**, averaged across the **{n_days} day"
        f"{'s' if n_days != 1 else ''}** in the selected period — not a "
        "single-day timelapse."
    )
    st.plotly_chart(chart_animated_map(stations, n_days), use_container_width=True)


def render_trains_tab(df_filtered: pd.DataFrame, ctx: dict) -> None:
    st.subheader("🚆 Top recurring late trains")
    st.caption(period_caption(ctx))
    st.markdown(
        "Train numbers (Infrabel `train_no`) whose arrivals are **recurringly** "
        "late on the selected period. Useful to flag timetable inputs that may "
        "need adjustment."
    )
    threshold = st.slider("Late threshold (min)", 1, 15, 5)
    top_n = st.slider("Top N", 5, 50, 20)
    table = top_recurring_late_trains(
        df_filtered, threshold_min=threshold, top_n=top_n
    )
    if table.empty:
        st.info("No data.")
        return
    st.dataframe(table, use_container_width=True)


def render_drilldown_tab(df_filtered: pd.DataFrame, ctx: dict) -> None:
    st.subheader("🔍 Drill-down — inspect a single station, train, or relation")
    st.caption(period_caption(ctx))

    target = st.radio(
        "What do you want to inspect?",
        ["Station", "Train", "Relation (IC / S / L)"],
        horizontal=True,
    )
    st.divider()

    if target == "Station":
        _render_station_detail(df_filtered, ctx)
    elif target == "Train":
        _render_train_detail(df_filtered, ctx)
    else:
        _render_relation_detail(df_filtered, ctx)


def _render_station_detail(df: pd.DataFrame, ctx: dict) -> None:
    stations_opts = available_stations(df)
    if not stations_opts:
        st.info("No stations in the filtered data.")
        return
    station = st.selectbox(
        "Pick a station (type to search)",
        options=stations_opts,
        index=0,
    )

    profile = station_hourly_profile(df, station)
    top_trains = station_top_trains(df, station, top_n=15)
    if profile.empty:
        st.info(f"No data for station {station}.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Arrivals", f"{int(profile['n_observations'].sum()):,}")
    weighted_mean = np.average(
        profile["mean_delay_min"], weights=profile["n_observations"]
    )
    col2.metric("Mean delay", f"{weighted_mean:.2f} min")
    col3.metric("Worst hour delay", f"{profile['mean_delay_min'].max():.2f} min")

    st.markdown(f"##### Hourly delay profile — **{station}**")
    st.caption(period_caption(ctx))
    fig = px.bar(
        profile, x="hour", y="mean_delay_min",
        color="mean_delay_min",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        labels={"hour": "Hour of day", "mean_delay_min": "Mean delay (min)"},
        hover_data={"n_observations": True},
    )
    fig.update_layout(coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### Trains stopping at **{station}** — worst by mean delay")
    st.caption(period_caption(ctx))
    if top_trains.empty:
        st.info("No train data.")
    else:
        st.dataframe(top_trains, use_container_width=True)


def _render_train_detail(df: pd.DataFrame, ctx: dict) -> None:
    trains_opts = available_train_nos(df)
    if not trains_opts:
        st.info("No trains in the filtered data.")
        return
    train_no = st.selectbox(
        "Pick a train number (type to search)",
        options=trains_opts,
        index=0,
    )

    profile = train_journey_profile(df, train_no)
    if profile.empty:
        st.info(f"No data for train {train_no}.")
        return

    relation = profile["relation"].iloc[0] if "relation" in profile.columns else "—"
    col1, col2, col3 = st.columns(3)
    col1.metric("Relation", str(relation))
    col2.metric("Stops in route", f"{len(profile)}")
    col3.metric(
        "Mean delay (worst stop)",
        f"{profile['mean_delay_min'].max():.2f} min",
    )

    st.markdown(
        f"##### Delay along the route — **train {train_no}** ({relation})"
    )
    st.caption(
        period_caption(ctx)
        + " · stops in chronological route order (median planned-arrival time)"
    )
    fig = px.line(
        profile, x="station_name", y="mean_delay_min",
        markers=True,
        hover_data={"n_days": True, "max_delay_min": True},
        labels={
            "station_name": "Stop (in route order)",
            "mean_delay_min": "Mean delay (min)",
        },
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=80))
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw stops table"):
        st.dataframe(profile, use_container_width=True)


def _render_relation_detail(df: pd.DataFrame, ctx: dict) -> None:
    relations_opts = available_relations(df)
    if not relations_opts:
        st.info("No relations in the filtered data.")
        return
    relation = st.selectbox(
        "Pick a relation (IC / S / L)",
        options=relations_opts,
        index=0,
        help="Example: 'IC 14-1' is IC 14 in direction 1.",
    )

    profile = relation_hourly_profile(df, relation)
    if profile.empty:
        st.info(f"No data for relation {relation}.")
        return

    sub = df[df["relation"].astype("string") == relation]
    n_trains = sub["train_no"].nunique() if not sub.empty else 0
    n_stations = sub["station_name"].nunique() if not sub.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Distinct train numbers", f"{n_trains}")
    col2.metric("Stations served", f"{n_stations}")
    col3.metric(
        "Mean delay",
        f"{np.average(profile['mean_delay_min'], weights=profile['n_observations']):.2f} min",
    )

    st.markdown(f"##### Hourly delay profile — **{relation}**")
    st.caption(period_caption(ctx))
    fig = px.bar(
        profile, x="hour", y="mean_delay_min",
        color="mean_delay_min",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#c0392b"],
        labels={"hour": "Hour of day", "mean_delay_min": "Mean delay (min)"},
        hover_data={"n_observations": True},
    )
    fig.update_layout(coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### Stations served by **{relation}** — sorted by mean delay")
    st.caption(period_caption(ctx))
    by_station = (
        sub.dropna(subset=["station_name", "delay_arr_min"])
        .groupby("station_name", as_index=False)
        .agg(
            arrivals=("delay_arr_min", "size"),
            mean_delay_min=("delay_arr_min", "mean"),
            max_delay_min=("delay_arr_min", "max"),
        )
        .round({"mean_delay_min": 2, "max_delay_min": 2})
        .sort_values("mean_delay_min", ascending=False)
    )
    st.dataframe(by_station, use_container_width=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Infrabel — Delay analysis", layout="wide")
    st.title("Infrabel — Belgian rail delay dashboard")
    st.caption(
        "Live data from mobilitytwin.brussels — explore train arrival delays "
        "by date range, weekday, holiday, station, and IC relation."
    )

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0

    # ----- Sidebar (period section) -----
    start, end = render_period_sidebar()

    # ----- Fetch + clean -----
    df_raw, failed_days, fetch_errors = load_range_dataframe(
        start, end, st.session_state.refresh_token
    )

    if failed_days:
        st.warning(
            f"⚠️ {len(failed_days)} day(s) could not be fetched after retries: "
            f"{[d.isoformat() for d in failed_days[:5]]}"
            f"{' ...' if len(failed_days) > 5 else ''}. "
            f"Click **🔄 Force refresh** in the sidebar to retry."
        )
        with st.expander("Fetch error details", expanded=False):
            for d in failed_days[:10]:
                st.code(f"{d.isoformat()}: {fetch_errors.get(d, 'unknown error')}")
            if len(failed_days) > 10:
                st.caption(f"{len(failed_days) - 10} more day(s) omitted.")

    # ----- Sidebar (filters section, dynamic options from df_raw) -----
    filt_ctx = render_filters_sidebar(df_raw)
    ctx = {"start": start, "end": end, **filt_ctx}

    # ----- Apply filters -----
    df = filter_by_weekdays(df_raw, ctx["weekdays"])
    df = filter_holidays(df, ctx["exclude_holidays"])
    df = filter_by_stations(df, ctx["stations_filter"])
    df = filter_by_relations(df, ctx["relations_filter"])

    # ----- Header description -----
    describe_period_block(ctx, n_records=len(df))

    if df.empty:
        st.warning(
            "No records left after the current filters. Loosen weekdays / "
            "stations / relations / holiday filters."
        )
        st.stop()

    # ----- Aggregations + geo joins -----
    stations_agg = aggregate_station_hour(df)
    lines_agg = aggregate_line_hour(df)
    relations_hour = aggregate_relation_hour(df)
    stations = join_stations(stations_agg, cached_operational_points())
    segments = join_lines(lines_agg, cached_line_sections())

    # ----- Tabs -----
    tab_map, tab_pat, tab_anim, tab_drill, tab_trains = st.tabs([
        "🗺️ Map", "📊 Patterns", "🎬 Animation",
        "🔍 Drill-down", "🚆 Trains",
    ])
    with tab_map:
        render_map_tab(stations, segments, relations_hour, ctx, df)
    with tab_pat:
        render_patterns_tab(df, ctx)
    with tab_anim:
        render_animation_tab(stations, ctx)
    with tab_drill:
        render_drilldown_tab(df, ctx)
    with tab_trains:
        render_trains_tab(df, ctx)


if __name__ == "__main__":
    main()
