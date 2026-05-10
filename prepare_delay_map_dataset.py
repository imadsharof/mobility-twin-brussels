"""
Build a map-ready GeoJSON of average train delays per station per hour.

Pipeline:
  1. Load the cleaned punctuality CSV (pandas) and the operational_points
     GeoJSON (geopandas).
  2. Aggregate delays by station and hour of day.
  3. Spatially join the aggregated table with station geometries.
  4. Export the result as a GeoJSON ready for web mapping (Leaflet, Folium,
     deck.gl, MapLibre...).

Run:
    python prepare_delay_map_dataset.py

Requires: pandas, geopandas, shapely (installed transitively by geopandas).
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PUNCTUALITY_CSV = PROCESSED_DIR / "clean_punctuality_30_days.csv"
OPERATIONAL_POINTS_GEOJSON = RAW_DIR / "operational_points.json"
OUTPUT_GEOJSON = PROCESSED_DIR / "delays_by_station_hour.geojson"

# ---------------------------------------------------------------------------
# Join-key configuration
# ---------------------------------------------------------------------------
# The CSV identifies stations by `station_name` (uppercase short label, e.g.
# "ZAVENTEM", "BRUXELLES-MIDI"). The GeoJSON exposes several candidate fields
# in `properties`:
#
#   - ptcarid              numeric internal Infrabel id (e.g. "443")
#   - taftapcode           European TAF/TAP code (e.g. "BE00443")
#   - symbolicname         5-letter telegraphic code (e.g. "FKGLF")
#   - shortnamefrench      short FR label, UPPERCASE  (e.g. "GENK-ZD-FORD")
#   - shortnamedutch       short NL label, UPPERCASE  (e.g. "GENK-ZD-FORD")
#   - longnamefrench       long FR label
#   - commerciallongnamefrench  pretty mixed-case label (e.g. "Genk-Zuid-...")
#
# `station_name` in the CSV matches `shortnamefrench` / `shortnamedutch` in
# almost every case, so we use those as the join key after normalization.
# If you ever switch to a richer ID (e.g. `ptcarid`), change CSV_JOIN_KEY and
# GEO_JOIN_KEYS accordingly — the rest of the pipeline is agnostic.
CSV_JOIN_KEY = "station_name"
GEO_JOIN_KEYS = ("shortnamefrench", "shortnamedutch")  # tried in order

# Delay & time columns coming out of clean_data.py
DELAY_COLUMN = "delay_arr_min"
HOUR_COLUMN = "hour_arr"


def normalize_key(series: pd.Series) -> pd.Series:
    """Uppercase + strip whitespace so labels match across sources."""
    return series.astype("string").str.strip().str.upper()


def load_punctuality(csv_path: Path) -> pd.DataFrame:
    """Read only the columns we need — the file is ~450 MB."""
    cols = [CSV_JOIN_KEY, HOUR_COLUMN, DELAY_COLUMN]
    df = pd.read_csv(csv_path, usecols=cols)

    # Drop rows with no station, no hour, or no delay value
    df = df.dropna(subset=cols)
    # `hour_arr` is float in the CSV (e.g. 17.0) — cast to int for clean output
    df[HOUR_COLUMN] = df[HOUR_COLUMN].astype(int)
    df["_join_key"] = normalize_key(df[CSV_JOIN_KEY])
    return df


def load_operational_points(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load station points and pick the first available join field."""
    gdf = gpd.read_file(geojson_path)

    chosen = next((c for c in GEO_JOIN_KEYS if c in gdf.columns), None)
    if chosen is None:
        raise KeyError(
            f"None of {GEO_JOIN_KEYS} found in operational_points.json. "
            f"Available columns: {list(gdf.columns)}"
        )
    print(f"[info] Using '{chosen}' from GeoJSON as the join field.")

    gdf["_join_key"] = normalize_key(gdf[chosen])
    # Keep one row per station; drop dupes that would explode the merge
    gdf = gdf.drop_duplicates(subset="_join_key", keep="first")
    return gdf


def aggregate_delays(df: pd.DataFrame) -> pd.DataFrame:
    """Mean delay (min) per station × hour, plus a sample-size column."""
    agg = (
        df.groupby(["_join_key", HOUR_COLUMN], as_index=False)
        .agg(
            mean_delay_min=(DELAY_COLUMN, "mean"),
            n_observations=(DELAY_COLUMN, "size"),
        )
        .round({"mean_delay_min": 2})
    )
    return agg


def main() -> None:
    print(f"[info] Loading {PUNCTUALITY_CSV.name} ...")
    punctuality = load_punctuality(PUNCTUALITY_CSV)
    print(f"       {len(punctuality):,} rows kept")

    print(f"[info] Loading {OPERATIONAL_POINTS_GEOJSON.name} ...")
    points = load_operational_points(OPERATIONAL_POINTS_GEOJSON)
    print(f"       {len(points):,} stations")

    print("[info] Aggregating delays by station × hour ...")
    aggregated = aggregate_delays(punctuality)
    print(f"       {len(aggregated):,} (station, hour) combinations")

    print("[info] Joining aggregated delays with station geometries ...")
    # Inner merge: only keep rows with both delay data AND coordinates.
    # Switch to `how="left"` if you want to surface unmatched stations.
    merged = aggregated.merge(
        points[["_join_key", "ptcarid", "symbolicname",
                "commerciallongnamefrench", "geometry"]],
        on="_join_key",
        how="inner",
    )

    matched_stations = merged["_join_key"].nunique()
    csv_stations = punctuality["_join_key"].nunique()
    print(f"       matched {matched_stations}/{csv_stations} unique stations")
    if matched_stations < csv_stations:
        unmatched = sorted(
            set(punctuality["_join_key"]) - set(points["_join_key"])
        )[:10]
        print(f"       sample of unmatched names: {unmatched}")

    # Promote back to GeoDataFrame (the merge with a regular DataFrame
    # demoted it) and set the same CRS as the source GeoJSON (WGS84).
    result = gpd.GeoDataFrame(
        merged.drop(columns="_join_key").rename(
            columns={
                "commerciallongnamefrench": "station_label",
                HOUR_COLUMN: "hour",
            }
        ),
        geometry="geometry",
        crs=points.crs or "EPSG:4326",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing {OUTPUT_GEOJSON} ...")
    # `driver="GeoJSON"` is inferred from the extension but we set it
    # explicitly for clarity. GeoJSON is always written in EPSG:4326.
    result.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"[done] {len(result):,} features exported.")


if __name__ == "__main__":
    main()
