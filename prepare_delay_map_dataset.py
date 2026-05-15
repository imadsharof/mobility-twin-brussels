"""
Build a map-ready GeoJSON of average train delays per station per hour.

CLI wrapper around `mobilitytwin.pipeline`. The aggregation and geo-join
logic is the SAME the Streamlit dashboard uses live — no duplication.

Run:
    python prepare_delay_map_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from mobilitytwin.pipeline import (
    aggregate_station_hour,
    join_stations,
    load_operational_points,
)

PUNCTUALITY_CSV = PROJECT_ROOT / "data" / "processed" / "clean_punctuality_30_days.csv"
OUTPUT_GEOJSON = PROJECT_ROOT / "data" / "processed" / "delays_by_station_hour.geojson"


def main() -> None:
    print(f"[info] Loading {PUNCTUALITY_CSV.name} ...")
    df = pd.read_csv(PUNCTUALITY_CSV)
    print(f"       {len(df):,} rows")

    print("[info] Aggregating by station × hour ...")
    agg = aggregate_station_hour(df)
    print(f"       {len(agg):,} (station, hour) groups")

    print("[info] Loading operational_points geometries ...")
    points = load_operational_points()
    print(f"       {len(points):,} unique stations with geometry")

    print("[info] Joining ...")
    result = join_stations(agg, points)
    matched = result["station_label"].nunique() if not result.empty else 0
    print(f"       matched {matched} unique stations")

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing {OUTPUT_GEOJSON.name} ...")
    result.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"[done] {len(result):,} features exported.")


if __name__ == "__main__":
    main()
