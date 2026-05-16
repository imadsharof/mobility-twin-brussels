"""
trends_tab.py — Drop this file next to app.py.

Provides render_trends_tab(df_filtered, ctx) which covers the three
longitudinal analyses missing from the original dashboard:

  1. Week-over-week punctuality trend  (structural degradation + anomalies)
  2. Recurring offenders               (train × weekday that's always late)
  3. Delay propagation along a route   (which stop initiates the problem)

Requires the three new functions added to src/mobilitytwin/pipeline.py:
  aggregate_weekly_trend, recurring_offenders, propagation_delta

Integration (3 edits in app.py):
  # 1 — import at top of app.py:
  from trends_tab import render_trends_tab

  # 2 — add "📈 Trends" to the st.tabs() call:
  tab_map, tab_pat, tab_anim, tab_drill, tab_trains, tab_trends = st.tabs([
      "🗺️ Map", "📊 Patterns", "🎬 Animation",
      "🔍 Drill-down", "🚆 Trains", "📈 Trends",
  ])

  # 3 — add at the end of main():
  with tab_trends:
      render_trends_tab(df, ctx)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── import the three new pipeline functions ──────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mobilitytwin.pipeline import (
    aggregate_weekly_trend,
    available_train_nos,
    propagation_delta,
    recurring_offenders,
)

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

# ── colour palette shared with app.py ────────────────────────────────────────
_SCALE = ["#2ecc71", "#f1c40f", "#c0392b"]


# ---------------------------------------------------------------------------
# Section 1 — Week-over-week trend
# ---------------------------------------------------------------------------
def _render_weekly_trend(df: pd.DataFrame, ctx: dict) -> None:
    st.markdown("### 📅 Week-over-week punctuality trend")
    st.markdown(
        "Each point is one calendar week. The shaded band shows ±1 standard "
        "deviation from the overall mean. Weeks flagged with a red ✕ are "
        "statistical outliers — likely strikes, severe weather, or major "
        "infrastructure works."
    )

    weekly = aggregate_weekly_trend(df)
    if weekly.empty:
        st.info("Not enough data to compute weekly trends. Select a longer period.")
        return

    n_weeks = len(weekly)
    if n_weeks < 2:
        st.info("Need at least 2 weeks of data for a trend chart.")
        return

    metric = st.radio(
        "Show metric as",
        ["Mean delay (min)", "% late arrivals (≥5 min)"],
        horizontal=True,
        key="trend_metric",
    )
    z_thresh = st.slider(
        "Anomaly threshold (z-score)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.1,
        key="trend_z",
        help="Weeks this many standard deviations above the mean are flagged.",
    )

    y_col = "mean_delay_min" if "Mean delay" in metric else "late_pct"
    y_label = "Mean delay (min)" if "Mean delay" in metric else "Late arrival rate"
    mu = weekly[y_col].mean()
    sigma = weekly[y_col].std()

    fig = go.Figure()

    # ± 1 SD band
    fig.add_trace(go.Scatter(
        x=list(weekly["week_label"]) + list(weekly["week_label"][::-1]),
        y=list((weekly[y_col] + sigma).clip(lower=0))
          + list((weekly[y_col] - sigma).clip(lower=0))[::-1],
        fill="toself",
        fillcolor="rgba(52,152,219,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 SD",
        showlegend=True,
    ))

    # Main trend line
    fig.add_trace(go.Scatter(
        x=weekly["week_label"],
        y=weekly[y_col],
        mode="lines+markers",
        name=y_label,
        line=dict(color="#3498db", width=2),
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{y_label}: %{{y:.3f}}<br>"
            "Observations: %{customdata}<extra></extra>"
        ),
        customdata=weekly["n_observations"],
    ))

    # Mean reference line
    fig.add_hline(y=mu, line_dash="dot", line_color="#7f8c8d",
                  annotation_text=f"Mean {mu:.2f}", annotation_position="right")

    # Anomaly markers
    anomalies = weekly[weekly["z_score"].abs() >= z_thresh]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["week_label"],
            y=anomalies[y_col],
            mode="markers",
            marker=dict(symbol="x", size=14, color="#c0392b", line=dict(width=2)),
            name=f"Anomaly (|z|≥{z_thresh})",
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{y_label}: %{{y:.3f}}<br>"
                "z-score: %{customdata:.2f}<extra></extra>"
            ),
            customdata=anomalies["z_score"],
        ))

    fig.update_layout(
        xaxis_title="ISO week",
        yaxis_title=y_label,
        height=420,
        margin=dict(l=0, r=0, t=10, b=60),
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weeks analysed", f"{n_weeks}")
    c2.metric("Overall mean delay", f"{weekly['mean_delay_min'].mean():.2f} min")
    c3.metric("Overall late rate", f"{weekly['late_pct'].mean()*100:.1f}%")
    c4.metric("Anomalous weeks", f"{len(anomalies)}")

    if not anomalies.empty:
        st.warning(
            f"**{len(anomalies)} anomalous week(s) detected** "
            f"(z ≥ {z_thresh}): "
            + ", ".join(anomalies["week_label"].tolist())
            + " — cross-check against Belgian rail strike calendar or severe weather events."
        )


# ---------------------------------------------------------------------------
# Section 2 — Recurring offenders
# ---------------------------------------------------------------------------
def _render_recurring_offenders(df: pd.DataFrame, ctx: dict) -> None:
    st.markdown("### 🚨 Recurring offenders — trains that are structurally late")
    st.markdown(
        "A **recurring offender** is a train that arrives late (≥ threshold) "
        "on the same weekday repeatedly. Unlike the 'top trains' tab which only "
        "counts total late events, this deduplicates by day so a train stopping "
        "at 10 stations still counts as **one run**. A high late-rate on a "
        "specific weekday usually points to a **timetable structuring problem**, "
        "not random bad luck."
    )

    col1, col2, col3 = st.columns(3)
    threshold = col1.slider("Late threshold (min)", 1, 15, 5, key="off_thresh")
    min_days = col2.slider("Min late days", 1, 20, 3, key="off_min")
    top_n = col3.slider("Show top N", 5, 50, 20, key="off_topn")

    offenders = recurring_offenders(df, threshold_min=threshold,
                                    min_late_days=min_days, top_n=top_n)

    if offenders.empty:
        st.info("No recurring offenders found with current settings. "
                "Try lowering the threshold or min late days.")
        return

    # Heatmap: train_no × weekday, coloured by late_rate
    pivot = (
        offenders.pivot_table(
            index="train_no", columns="weekday",
            values="late_rate", aggfunc="mean",
        )
        .reindex(columns=[d for d in WEEKDAY_ORDER if d in offenders["weekday"].unique()])
    )

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=_SCALE,
        zmin=0, zmax=1,
        labels=dict(x="Weekday", y="Train number", color="Late rate"),
        title="Late rate by train × weekday (darker = more often late)",
        aspect="auto",
    )
    fig_heat.update_layout(height=max(300, len(pivot) * 22 + 80),
                           margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Bar chart — top offenders by late_days
    top10 = offenders.head(10).sort_values("late_days")
    top10["label"] = top10["train_no"].astype(str) + " (" + top10["weekday"].str[:3] + ")"

    fig_bar = px.bar(
        top10, x="late_days", y="label", orientation="h",
        color="late_rate",
        color_continuous_scale=_SCALE,
        labels={"late_days": "Late days", "label": "", "late_rate": "Late rate"},
        title="Top 10 recurring offenders by number of late days",
        hover_data={"mean_delay_min": True, "max_delay_min": True,
                    "total_days": True, "relation": True},
    )
    fig_bar.update_layout(coloraxis_showscale=True, height=380,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Full recurring offenders table"):
        st.dataframe(offenders, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3 — Delay propagation
# ---------------------------------------------------------------------------
def _render_propagation(df: pd.DataFrame, ctx: dict) -> None:
    st.markdown("### 🔁 Delay propagation along a route")
    st.markdown(
        "For a chosen train, this shows **where** delay accumulates stop by stop. "
        "The **initiator** is the station where delay first jumps significantly — "
        "that's the root cause. **Amplifiers** make it worse; **absorbers** recover "
        "some time. Grey bars are neutral stops."
    )

    trains = available_train_nos(df)
    if not trains:
        st.info("No train data available with the current filters.")
        return

    train_no = st.selectbox(
        "Pick a train number",
        options=trains,
        key="prop_train",
        help="Type to search. Choose trains from the 'recurring offenders' table above for best results.",
    )

    profile = propagation_delta(df, train_no)
    if profile.empty:
        st.info(f"No route data for train {train_no}.")
        return

    relation = profile["relation"].iloc[0] if "relation" in profile.columns else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relation", str(relation))
    c2.metric("Stops", f"{len(profile)}")
    c3.metric("Max delay", f"{profile['mean_delay_min'].max():.2f} min")
    initiator_rows = profile[profile["role"] == "initiator"]
    c4.metric(
        "Initiator station",
        initiator_rows["station_name"].iloc[0] if not initiator_rows.empty else "—",
    )

    role_colors = {
        "initiator": "#c0392b",
        "amplifier": "#e67e22",
        "absorber":  "#2ecc71",
        "neutral":   "#bdc3c7",
    }

    # Dual-axis: mean delay line + delta bars
    fig = go.Figure()

    # Delta bars coloured by role
    bar_colors = [role_colors.get(r, "#bdc3c7") for r in profile["role"]]
    fig.add_trace(go.Bar(
        x=profile["station_name"],
        y=profile["delta_min"],
        marker_color=bar_colors,
        name="Delay change (min)",
        yaxis="y2",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Change: %{y:+.2f} min<br>"
            "Role: %{customdata}<extra></extra>"
        ),
        customdata=profile["role"],
        opacity=0.7,
    ))

    # Absolute delay line
    fig.add_trace(go.Scatter(
        x=profile["station_name"],
        y=profile["mean_delay_min"],
        mode="lines+markers",
        name="Mean delay (min)",
        line=dict(color="#2980b9", width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean delay: %{y:.2f} min<br>"
            "Runs: %{customdata}<extra></extra>"
        ),
        customdata=profile["n_days"],
    ))

    # Annotate initiator
    if not initiator_rows.empty:
        init = initiator_rows.iloc[0]
        fig.add_annotation(
            x=init["station_name"],
            y=init["mean_delay_min"],
            text="🔴 Initiator",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#c0392b",
            font=dict(color="#c0392b", size=12),
            yshift=20,
        )

    fig.update_layout(
        title=f"Delay propagation — train {train_no} ({relation})",
        xaxis=dict(title="Stop (route order)", tickangle=-45),
        yaxis=dict(title="Mean delay (min)", side="left"),
        yaxis2=dict(title="Change per stop (min)", side="right",
                    overlaying="y", zeroline=True,
                    zerolinecolor="#7f8c8d", zerolinewidth=1),
        height=480,
        margin=dict(l=0, r=0, t=50, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        barmode="relative",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Role legend
    role_col1, role_col2, role_col3, role_col4 = st.columns(4)
    role_col1.markdown("🔴 **Initiator** — first big delay jump")
    role_col2.markdown("🟠 **Amplifier** — makes it worse")
    role_col3.markdown("🟢 **Absorber** — recovers time")
    role_col4.markdown("⬜ **Neutral** — no significant change")

    with st.expander("Raw propagation table"):
        st.dataframe(profile, use_container_width=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def render_trends_tab(df: pd.DataFrame, ctx: dict) -> None:
    n_days = (ctx["end"] - ctx["start"]).days + 1
    st.subheader(
        f"📈 Long-term trends & structural analysis — "
        f"{n_days} day{'s' if n_days != 1 else ''}"
    )
    st.markdown(
        "These analyses go **beyond daily snapshots** to find patterns that "
        "repeat over weeks, by weekday, or along a route — the kind of "
        "structural issues that warrant timetable adjustments."
    )

    if df.empty:
        st.warning("No data available with current filters.")
        return

    if n_days < 7:
        st.warning(
            "⚠️ Select at least **7 days** (preferably 30+) to get meaningful "
            "trend and recurrence results."
        )

    st.divider()
    _render_weekly_trend(df, ctx)

    st.divider()
    _render_recurring_offenders(df, ctx)

    st.divider()
    _render_propagation(df, ctx)
