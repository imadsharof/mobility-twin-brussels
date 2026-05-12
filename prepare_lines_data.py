"""
Build a map-ready GeoJSON of average train delays per railway line per hour.

Pipeline:
  1. Load the cleaned punctuality CSV (pandas) and line_sections.json (geopandas).
  2. Aggregate delays by line number and hour-of-day.
  3. Dissolve track segments by line number so each line has ONE geometry.
  4. Inner-join the aggregated table with the dissolved geometries.
  5. Export the result as data/processed/delays_by_segment_hour.geojson.

Run:
    python prepare_lines_data.py
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
LINE_SECTIONS_GEOJSON = RAW_DIR / "line_sections.json"
OUTPUT_GEOJSON = PROCESSED_DIR / "delays_by_segment_hour.geojson"

# ---------------------------------------------------------------------------
# Join-key configuration
# ---------------------------------------------------------------------------
# HOW TO IDENTIFY THE JOIN KEY BETWEEN THE CSV AND line_sections.json
# -------------------------------------------------------------------
# The cleaned CSV exposes two columns referring to the railway line:
#   - line_no_arr : Infrabel line code for the ARRIVAL track  (e.g. "139/1")
#   - line_no_dep : Infrabel line code for the DEPARTURE track (e.g. "27/1")
# DESPITE the "_no_" in the name, these values are STRINGS that follow the
# alphanumeric "linecalfa" notation: a base line number + optional letter
# suffix (A, D, L) + "/<track_index>"  →  e.g. "139/1", "124A/1", "0/5".
# We use `line_no_arr` because the delay we aggregate (`delay_arr_min`) is
# the delay measured AT ARRIVAL.
#
# In line_sections.json each LineString carries several candidate identifiers
# under `properties`:
#   - linecnum   : integer line number, e.g. 139           <-- too coarse,
#                  matches only 10/150 CSV codes because the slash suffix
#                  is dropped.
#   - linecalfa  : alphanumeric line code, e.g. "139/1"    <-- canonical match,
#                  matches 118/150 (~79%) of CSV codes exactly.
#   - trackcode  : segment-level code (e.g. "L 35/1_1")    <-- too granular
#   - trackname  : sub-track letter (A, B, ...)            <-- too granular
#   - id         : segment row id                          <-- not a line id
#
# So the canonical join is:   CSV.line_no_arr  ==  GeoJSON.linecalfa
# Unmatched CSV codes are typically internal yard tracks (e.g. "0/1".."0/6")
# or "L" variants ("139L/1") that have no public geometry — we drop them
# with a warning.
CSV_JOIN_KEY = "line_no_arr"
GEO_JOIN_KEY = "linecalfa"

# Delay & time columns coming out of clean_data.py
DELAY_COLUMN = "delay_arr_min"
HOUR_COLUMN = "hour_arr"


def normalize_line_code(series: pd.Series) -> pd.Series:
    """Strip + uppercase Infrabel line codes so 'csv' and 'geo' match."""
    return series.astype("string").str.strip().str.upper()


def load_punctuality(csv_path: Path) -> pd.DataFrame:
    """Read only the columns we need — the file is ~450 MB."""
    cols = [CSV_JOIN_KEY, HOUR_COLUMN, DELAY_COLUMN]
    df = pd.read_csv(csv_path, usecols=cols, dtype={CSV_JOIN_KEY: "string"})
    df = df.dropna(subset=cols)
    df[CSV_JOIN_KEY] = normalize_line_code(df[CSV_JOIN_KEY])
    df[HOUR_COLUMN] = df[HOUR_COLUMN].astype(int)
    return df


def load_line_sections(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load track segments and dissolve to one geometry per line code."""
    gdf = gpd.read_file(geojson_path)
    if GEO_JOIN_KEY not in gdf.columns:
        raise KeyError(
            f"'{GEO_JOIN_KEY}' not found in {geojson_path.name}. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf[GEO_JOIN_KEY] = normalize_line_code(gdf[GEO_JOIN_KEY])
    gdf = gdf.dropna(subset=[GEO_JOIN_KEY])

    # Dissolve: union all LineStrings sharing the same line code into one
    # MultiLineString. Collapses the 1k+ segments into ~260 line codes.
    dissolved = (
        gdf.dissolve(by=GEO_JOIN_KEY, as_index=False)
        [[GEO_JOIN_KEY, "geometry"]]
    )
    return dissolved


def aggregate_delays(df: pd.DataFrame) -> pd.DataFrame:
    """Mean delay (min) per line × hour, plus a sample-size column."""
    agg = (
        df.groupby([CSV_JOIN_KEY, HOUR_COLUMN], as_index=False)
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
    print(f"       {len(punctuality):,} delay rows kept")

    print(f"[info] Loading {LINE_SECTIONS_GEOJSON.name} ...")
    lines = load_line_sections(LINE_SECTIONS_GEOJSON)
    print(f"       {len(lines):,} unique lines after dissolve")

    print("[info] Aggregating delays by line × hour ...")
    aggregated = aggregate_delays(punctuality)
    print(f"       {len(aggregated):,} (line, hour) combinations")

    print("[info] Joining aggregated delays with line geometries ...")
    merged = aggregated.merge(
        lines,
        left_on=CSV_JOIN_KEY,
        right_on=GEO_JOIN_KEY,
        how="inner",
    )

    matched_lines = merged[CSV_JOIN_KEY].nunique()
    csv_lines = punctuality[CSV_JOIN_KEY].nunique()
    print(f"       matched {matched_lines}/{csv_lines} unique line codes")
    if matched_lines < csv_lines:
        unmatched = sorted(
            set(punctuality[CSV_JOIN_KEY].dropna())
            - set(lines[GEO_JOIN_KEY].dropna())
        )[:10]
        print(f"       sample of unmatched line codes: {unmatched}")

    result = gpd.GeoDataFrame(
        merged.rename(
            columns={
                CSV_JOIN_KEY: "line_no",
                HOUR_COLUMN: "hour",
            }
        ).drop(columns=[GEO_JOIN_KEY], errors="ignore"),
        geometry="geometry",
        crs=lines.crs or "EPSG:4326",
    )
    result["line_no"] = result["line_no"].astype(str)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing {OUTPUT_GEOJSON} ...")
    result.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"[done] {len(result):,} features exported.")


if __name__ == "__main__":
    main()
