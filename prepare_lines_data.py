"""
Build a map-ready GeoJSON of average train delays per railway line per hour.

CLI wrapper around `mobilitytwin.pipeline`. The line-level aggregation and
geo-join logic is the SAME the Streamlit dashboard uses live.

How the join key is identified
------------------------------
The cleaned CSV exposes `line_no_arr` and `line_no_dep` as strings using the
Infrabel alphanumeric line code (e.g. "139/1", "124A/2", "0/5"). These match
the `linecalfa` property of line_sections.json (NOT the integer `linecnum`,
which strips the slash suffix). All this is handled inside
`mobilitytwin.pipeline.{aggregate_line_hour, join_lines}` — see those
docstrings for the full rationale.

Run:
    python prepare_lines_data.py
"""


from __future__ import annotations 

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from mobilitytwin.pipeline import (
    aggregate_line_hour,
    join_lines,
    load_line_sections,
)

PUNCTUALITY_CSV = PROJECT_ROOT / "data" / "processed" / "clean_punctuality_30_days.csv"
OUTPUT_GEOJSON = PROJECT_ROOT / "data" / "processed" / "delays_by_segment_hour.geojson"


def main() -> None:
    print(f"[info] Loading {PUNCTUALITY_CSV.name} ...")
    df = pd.read_csv(PUNCTUALITY_CSV)
    print(f"       {len(df):,} rows")

    print("[info] Aggregating by line × hour ...")
    agg = aggregate_line_hour(df)
    print(f"       {len(agg):,} (line, hour) groups")

    print("[info] Loading line_sections geometries (dissolved by line code) ...")
    lines = load_line_sections()
    print(f"       {len(lines):,} unique line codes with geometry")

    print("[info] Joining ...")
    result = join_lines(agg, lines)
    matched = result["line_no"].nunique() if not result.empty else 0
    print(f"       matched {matched} unique line codes")

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing {OUTPUT_GEOJSON.name} ...")
    result.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"[done] {len(result):,} features exported.")


if __name__ == "__main__":
    main()
