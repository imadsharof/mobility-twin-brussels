"""Cleaning + aggregation pipelines used by the Streamlit dashboard.

Pure functions: take API records (list of dicts) and produce the GeoDataFrames
the map / charts consume. No I/O except for loading the static reference
geometries (operational_points + line_sections), both already cached in
data/raw/.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import holidays
import pandas as pd

from .api import ensure_reference_data

_BE_HOLIDAYS = holidays.country_holidays("BE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

OPERATIONAL_POINTS_FILE = RAW_DIR / "operational_points.json"
LINE_SECTIONS_FILE = RAW_DIR / "line_sections.json"

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


# ---------------------------------------------------------------------------
# Records -> cleaned DataFrame
# ---------------------------------------------------------------------------
def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Apply the same cleaning as scripts/clean_data.py but in-memory."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["planned_datetime_arr"] = pd.to_datetime(
        df["planned_date_arr"].astype(str) + " " + df["planned_time_arr"].astype(str),
        errors="coerce",
    )
    df["planned_datetime_dep"] = pd.to_datetime(
        df["planned_date_dep"].astype(str) + " " + df["planned_time_dep"].astype(str),
        errors="coerce",
    )
    df["delay_arr_min"] = pd.to_numeric(df.get("delay_arr"), errors="coerce") / 60
    df["delay_dep_min"] = pd.to_numeric(df.get("delay_dep"), errors="coerce") / 60

    df["date"] = pd.to_datetime(df.get("datdep"), errors="coerce")
    df["hour_arr"] = df["planned_datetime_arr"].dt.hour
    df["hour_dep"] = df["planned_datetime_dep"].dt.hour
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.weekday >= 5
    df["is_holiday"] = df["date"].dt.date.map(
        lambda d: bool(d in _BE_HOLIDAYS) if pd.notna(d) else False
    )

    df = df.rename(columns={"ptcar_lg_nm_nl": "station_name"})
    return df


# ---------------------------------------------------------------------------
# Filtering helpers (used by the Streamlit sidebar)
# ---------------------------------------------------------------------------
def filter_by_weekdays(df: pd.DataFrame, weekdays: list[str]) -> pd.DataFrame:
    if not weekdays or df.empty:
        return df
    return df[df["weekday"].isin(weekdays)]


def filter_holidays(df: pd.DataFrame, exclude: bool) -> pd.DataFrame:
    if not exclude or df.empty or "is_holiday" not in df.columns:
        return df
    return df[~df["is_holiday"]]


def filter_by_stations(df: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    if not stations or df.empty:
        return df
    sel = {s.strip().upper() for s in stations}
    return df[df["station_name"].astype("string").str.strip().str.upper().isin(sel)]


def filter_by_relations(df: pd.DataFrame, relations: list[str]) -> pd.DataFrame:
    if not relations or df.empty or "relation" not in df.columns:
        return df
    return df[df["relation"].astype("string").isin(relations)]


def available_stations(df: pd.DataFrame) -> list[str]:
    if df.empty or "station_name" not in df.columns:
        return []
    return sorted(df["station_name"].dropna().astype(str).str.strip().unique())


def available_relations(df: pd.DataFrame) -> list[str]:
    if df.empty or "relation" not in df.columns:
        return []
    return sorted(df["relation"].dropna().astype(str).unique())


def available_train_nos(df: pd.DataFrame) -> list[str]:
    if df.empty or "train_no" not in df.columns:
        return []
    return sorted(df["train_no"].dropna().astype(str).unique(), key=lambda s: (len(s), s))


# ---------------------------------------------------------------------------
# Single-entity profiles (drill-down tab)
# ---------------------------------------------------------------------------
def station_hourly_profile(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """Mean delay per hour for a single station, plus sample counts."""
    if df.empty or not station_name:
        return pd.DataFrame()
    s = station_name.strip().upper()
    mask = df["station_name"].astype("string").str.strip().str.upper() == s
    sub = df[mask].dropna(subset=["hour_arr", "delay_arr_min"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["hour_arr"] = sub["hour_arr"].astype(int)
    return (
        sub.groupby("hour_arr", as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


def station_top_trains(
    df: pd.DataFrame, station_name: str, top_n: int = 15
) -> pd.DataFrame:
    """For one station, list the trains that stop there ranked by mean delay."""
    if df.empty or not station_name:
        return pd.DataFrame()
    s = station_name.strip().upper()
    mask = df["station_name"].astype("string").str.strip().str.upper() == s
    sub = df[mask].dropna(subset=["train_no", "delay_arr_min"]).copy()
    if sub.empty:
        return pd.DataFrame()
    grouped = (
        sub.groupby("train_no", as_index=False)
        .agg(
            relation=("relation", "first"),
            stops=("delay_arr_min", "size"),
            mean_delay_min=("delay_arr_min", "mean"),
            max_delay_min=("delay_arr_min", "max"),
        )
        .round({"mean_delay_min": 2, "max_delay_min": 2})
        .sort_values("mean_delay_min", ascending=False)
        .head(top_n)
    )
    return grouped


def train_journey_profile(df: pd.DataFrame, train_no: str) -> pd.DataFrame:
    """Average delay per station for one train, in route order.

    Stop order is inferred from the median planned-arrival minute-of-day,
    so stations come out in the order the train traverses them.
    """
    if df.empty or train_no is None or str(train_no) == "":
        return pd.DataFrame()
    sub = df[df["train_no"].astype(str) == str(train_no)].dropna(
        subset=["station_name", "delay_arr_min", "planned_datetime_arr"]
    ).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["minutes_of_day"] = (
        sub["planned_datetime_arr"].dt.hour * 60
        + sub["planned_datetime_arr"].dt.minute
    )
    profile = (
        sub.groupby("station_name", as_index=False)
        .agg(
            stop_order=("minutes_of_day", "median"),
            n_days=("delay_arr_min", "size"),
            mean_delay_min=("delay_arr_min", "mean"),
            max_delay_min=("delay_arr_min", "max"),
            relation=("relation", "first"),
        )
        .round({"mean_delay_min": 2, "max_delay_min": 2})
        .sort_values("stop_order")
        .reset_index(drop=True)
    )
    return profile


def relation_hourly_profile(df: pd.DataFrame, relation: str) -> pd.DataFrame:
    """Hourly profile of mean delay for one IC/S/L relation."""
    if df.empty or not relation:
        return pd.DataFrame()
    sub = df[df["relation"].astype("string") == relation].dropna(
        subset=["hour_arr", "delay_arr_min"]
    ).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["hour_arr"] = sub["hour_arr"].astype(int)
    return (
        sub.groupby("hour_arr", as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


def aggregate_weekday_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Mean delay per (weekday, hour) — drives the recurring-patterns heatmap."""
    cols = ["weekday", "hour", "mean_delay_min", "n_observations"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    sub = df.dropna(subset=["weekday", "hour_arr", "delay_arr_min"]).copy()
    sub["hour_arr"] = sub["hour_arr"].astype(int)
    return (
        sub.groupby(["weekday", "hour_arr"], as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


def top_recurring_late_trains(
    df: pd.DataFrame, threshold_min: float = 5.0, top_n: int = 20
) -> pd.DataFrame:
    """Trains with the most late-arrival events over the selected period."""
    if df.empty or "train_no" not in df.columns:
        return pd.DataFrame(columns=[
            "train_no", "relation", "late_events", "total_stops",
            "late_rate", "mean_delay_min", "max_delay_min",
        ])

    sub = df.dropna(subset=["train_no", "delay_arr_min"]).copy()
    sub["is_late"] = sub["delay_arr_min"] >= threshold_min
    grouped = sub.groupby("train_no").agg(
        late_events=("is_late", "sum"),
        total_stops=("is_late", "size"),
        mean_delay_min=("delay_arr_min", "mean"),
        max_delay_min=("delay_arr_min", "max"),
        relation=("relation", "first"),
    )
    grouped["late_rate"] = (grouped["late_events"] / grouped["total_stops"]).round(3)
    grouped = grouped.sort_values("late_events", ascending=False).head(top_n)
    return (
        grouped.reset_index()
        .round({"mean_delay_min": 2, "max_delay_min": 2})
        [["train_no", "relation", "late_events", "total_stops",
          "late_rate", "mean_delay_min", "max_delay_min"]]
    )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def aggregate_station_hour(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["_join_key", "hour", "mean_delay_min", "n_observations"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    sub = df.dropna(subset=["station_name", "hour_arr", "delay_arr_min"]).copy()
    sub["_join_key"] = sub["station_name"].astype("string").str.strip().str.upper()
    sub["hour_arr"] = sub["hour_arr"].astype(int)

    return (
        sub.groupby(["_join_key", "hour_arr"], as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


def aggregate_relation_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Mean delay per (relation, hour) — drives the IC route ranking."""
    cols = ["relation", "hour", "mean_delay_min", "n_observations"]
    if df.empty or "relation" not in df.columns:
        return pd.DataFrame(columns=cols)

    sub = df.dropna(subset=["relation", "hour_arr", "delay_arr_min"]).copy()
    sub["relation"] = sub["relation"].astype("string")
    sub["hour_arr"] = sub["hour_arr"].astype(int)
    return (
        sub.groupby(["relation", "hour_arr"], as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


def aggregate_line_hour(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["line_no", "hour", "mean_delay_min", "n_observations"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    sub = df.dropna(subset=["line_no_arr", "hour_arr", "delay_arr_min"]).copy()
    sub["line_no"] = sub["line_no_arr"].astype("string").str.strip().str.upper()
    sub["hour_arr"] = sub["hour_arr"].astype(int)

    return (
        sub.groupby(["line_no", "hour_arr"], as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .round({"mean_delay_min": 2})
        .rename(columns={"hour_arr": "hour"})
    )


# ---------------------------------------------------------------------------
# Geometry loaders (cached at the Streamlit layer)
# ---------------------------------------------------------------------------
def load_operational_points() -> gpd.GeoDataFrame:
    if not OPERATIONAL_POINTS_FILE.exists():
        ensure_reference_data()
    gdf = gpd.read_file(OPERATIONAL_POINTS_FILE).to_crs(4326)
    name_col = next(
        (c for c in ("shortnamefrench", "shortnamedutch") if c in gdf.columns),
        None,
    )
    if name_col is None:
        raise KeyError(
            "operational_points.json has no shortnamefrench/shortnamedutch column"
        )
    gdf["_join_key"] = gdf[name_col].astype("string").str.strip().str.upper()
    gdf = gdf.drop_duplicates(subset="_join_key", keep="first")
    return gdf


def load_line_sections() -> gpd.GeoDataFrame:
    if not LINE_SECTIONS_FILE.exists():
        ensure_reference_data()
    gdf = gpd.read_file(LINE_SECTIONS_FILE).to_crs(4326)
    gdf["linecalfa"] = gdf["linecalfa"].astype("string").str.strip().str.upper()
    gdf = gdf.dropna(subset=["linecalfa"])
    return gdf.dissolve(by="linecalfa", as_index=False)[["linecalfa", "geometry"]]


# ---------------------------------------------------------------------------
# Geo joins
# ---------------------------------------------------------------------------
def join_stations(
    agg: pd.DataFrame, points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out_cols = [
        "station_label", "hour", "mean_delay_min", "n_observations",
        "lon", "lat", "geometry",
    ]
    if agg.empty:
        return gpd.GeoDataFrame(columns=out_cols, geometry="geometry", crs="EPSG:4326")

    label_col = (
        "commerciallongnamefrench"
        if "commerciallongnamefrench" in points.columns
        else "_join_key"
    )
    merged = agg.merge(
        points[["_join_key", label_col, "geometry"]],
        on="_join_key",
        how="inner",
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=points.crs or "EPSG:4326")
    merged = merged.rename(columns={label_col: "station_label"})
    merged["lon"] = merged.geometry.x
    merged["lat"] = merged.geometry.y
    return merged[out_cols]


def join_lines(
    agg: pd.DataFrame, lines: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out_cols = ["line_no", "hour", "mean_delay_min", "n_observations", "geometry"]
    if agg.empty:
        return gpd.GeoDataFrame(columns=out_cols, geometry="geometry", crs="EPSG:4326")

    merged = agg.merge(lines, left_on="line_no", right_on="linecalfa", how="inner")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=lines.crs or "EPSG:4326")
    return merged[out_cols]

 
# ---------------------------------------------------------------------------
# Trends analysis — NEW functions to add to pipeline.py
# ---------------------------------------------------------------------------
 
def aggregate_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group arrivals by ISO calendar week.
 
    Returns one row per week with:
      iso_year, iso_week, week_label (str "YYYY-Www"),
      mean_delay_min, late_pct (fraction arriving ≥5 min late),
      n_observations, z_score (how many SDs above the overall mean).
 
    Use this to spot structural degradation or outlier weeks (strikes, weather).
    """
    cols = ["iso_year", "iso_week", "week_label",
            "mean_delay_min", "late_pct", "n_observations", "z_score"]
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=cols)
 
    sub = df.dropna(subset=["date", "delay_arr_min"]).copy()
    sub["date"] = pd.to_datetime(sub["date"])
    iso = sub["date"].dt.isocalendar()
    sub["iso_year"] = iso["year"].astype(int)
    sub["iso_week"] = iso["week"].astype(int)
    sub["is_late_5"] = sub["delay_arr_min"] >= 5
 
    agg = (
        sub.groupby(["iso_year", "iso_week"], as_index=False)
        .agg(
            mean_delay_min=("delay_arr_min", "mean"),
            late_pct=("is_late_5", "mean"),
            n_observations=("delay_arr_min", "size"),
        )
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    agg["week_label"] = (
        agg["iso_year"].astype(str) + "-W"
        + agg["iso_week"].astype(str).str.zfill(2)
    )
 
    mu = agg["mean_delay_min"].mean()
    sigma = agg["mean_delay_min"].std()
    agg["z_score"] = ((agg["mean_delay_min"] - mu) / sigma).round(2) if sigma > 0 else 0.0
 
    return agg.round({"mean_delay_min": 2, "late_pct": 4})[cols]
 
 
def recurring_offenders(
    df: pd.DataFrame,
    threshold_min: float = 5.0,
    min_late_days: int = 3,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find (train_no, weekday) pairs that are structurally late.
 
    A 'recurring offender' is a train that arrives ≥ threshold_min late
    on the same weekday at least min_late_days times.  Deduplicating by day
    means a train stopping at 10 stations still counts as ONE run per day.
 
    Returns columns:
      train_no, relation, weekday,
      late_days       — number of distinct days it was late on that weekday,
      total_days      — how many times it ran on that weekday,
      late_rate       — late_days / total_days,
      mean_delay_min  — mean delay on days it ran (including on-time runs),
      max_delay_min.
    """
    cols = ["train_no", "relation", "weekday",
            "late_days", "total_days", "late_rate",
            "mean_delay_min", "max_delay_min"]
    if df.empty or "train_no" not in df.columns:
        return pd.DataFrame(columns=cols)
 
    sub = df.dropna(subset=["train_no", "weekday", "delay_arr_min", "date"]).copy()
    sub["date_only"] = pd.to_datetime(sub["date"]).dt.date
    sub["is_late"] = sub["delay_arr_min"] >= threshold_min
 
    # One row per (train_no, weekday, date) — dedup across stations
    daily = (
        sub.groupby(["train_no", "weekday", "date_only"], as_index=False)
        .agg(
            is_late=("is_late", "any"),
            delay_arr_min=("delay_arr_min", "mean"),
            relation=("relation", "first"),
        )
    )
 
    result = (
        daily.groupby(["train_no", "weekday"], as_index=False)
        .agg(
            relation=("relation", "first"),
            late_days=("is_late", "sum"),
            total_days=("is_late", "size"),
            mean_delay_min=("delay_arr_min", "mean"),
            max_delay_min=("delay_arr_min", "max"),
        )
    )
    result["late_rate"] = (result["late_days"] / result["total_days"]).round(3)
    result = (
        result[result["late_days"] >= min_late_days]
        .sort_values("late_days", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .round({"mean_delay_min": 2, "max_delay_min": 2})
    )
    return result[cols]
 
 
def propagation_delta(df: pd.DataFrame, train_no: str) -> pd.DataFrame:
    """
    Extend train_journey_profile with stop-to-stop delay change and a role label.
 
    Role logic (applied to mean_delay_min delta):
      'initiator'  — first stop where delay jumps by ≥ 2 min
      'amplifier'  — subsequent stop where delay increases by ≥ 1 min
      'absorber'   — stop where delay decreases by ≥ 1 min
      'neutral'    — change within ±1 min
 
    Returns all columns from train_journey_profile plus:
      delta_min  (delay change vs previous stop),
      role       (string label above).
    """
    # Re-use the existing journey profile logic inline
    if df.empty or not train_no:
        return pd.DataFrame()
 
    sub = df[df["train_no"].astype(str) == str(train_no)].dropna(
        subset=["station_name", "delay_arr_min", "planned_datetime_arr"]
    ).copy()
    if sub.empty:
        return pd.DataFrame()
 
    sub["minutes_of_day"] = (
        sub["planned_datetime_arr"].dt.hour * 60
        + sub["planned_datetime_arr"].dt.minute
    )
    profile = (
        sub.groupby("station_name", as_index=False)
        .agg(
            stop_order=("minutes_of_day", "median"),
            n_days=("delay_arr_min", "size"),
            mean_delay_min=("delay_arr_min", "mean"),
            max_delay_min=("delay_arr_min", "max"),
            relation=("relation", "first"),
        )
        .round({"mean_delay_min": 2, "max_delay_min": 2})
        .sort_values("stop_order")
        .reset_index(drop=True)
    )
 
    profile["delta_min"] = profile["mean_delay_min"].diff().round(2)
 
    first_big_jump = profile["delta_min"].ge(2).idxmax() if profile["delta_min"].ge(2).any() else None
 
    def _role(row):
        if pd.isna(row["delta_min"]):
            return "neutral"
        if first_big_jump is not None and row.name == first_big_jump:
            return "initiator"
        if row["delta_min"] >= 1:
            return "amplifier"
        if row["delta_min"] <= -1:
            return "absorber"
        return "neutral"
 
    profile["role"] = profile.apply(_role, axis=1)
    return profile