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
