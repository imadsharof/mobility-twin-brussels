"""
Offline batch cleaning: turn the raw 30-day JSON dump into a tidy CSV.

This is a CLI wrapper. All the cleaning logic lives in
`src/mobilitytwin/pipeline.py:records_to_dataframe` — the SAME function the
Streamlit dashboard uses on the records streamed from the API. There is no
duplicated cleaning code anymore.

Run:
    python clean_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from mobilitytwin.pipeline import records_to_dataframe

RAW_JSON = PROJECT_ROOT / "data" / "raw" / "punctuality_30_days.json"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "clean_punctuality_30_days.csv"

# Columns kept in the CSV. The dashboard doesn't need this file at all —
# it's a convenience artifact for notebooks / external tools.
CSV_COLUMNS = [
    "date", "train_no", "relation", "train_service",
    "station_name", "line_no_arr", "line_no_dep",
    "planned_datetime_arr", "planned_datetime_dep",
    "real_datetime_arr", "real_datetime_dep",
    "delay_arr_sec", "delay_dep_sec",
    "delay_arr_min", "delay_dep_min",
    "hour_arr", "hour_dep",
    "weekday", "is_weekend", "is_holiday",
    "is_late_arr_5min", "is_late_dep_5min",
    "relation_direction",
]


def main() -> None:
    print(f"[info] Loading {RAW_JSON.name} ...")
    with RAW_JSON.open("r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"       {len(records):,} raw records")

    print("[info] Cleaning via mobilitytwin.pipeline.records_to_dataframe ...")
    df = records_to_dataframe(records)

    # Fields specific to the CSV artifact (not needed by the dashboard,
    # so kept out of pipeline.records_to_dataframe).
    df["real_datetime_arr"] = pd.to_datetime(
        df["real_date_arr"].astype(str) + " " + df["real_time_arr"].astype(str),
        errors="coerce",
    )
    df["real_datetime_dep"] = pd.to_datetime(
        df["real_date_dep"].astype(str) + " " + df["real_time_dep"].astype(str),
        errors="coerce",
    )
    df["delay_arr_sec"] = pd.to_numeric(df.get("delay_arr"), errors="coerce")
    df["delay_dep_sec"] = pd.to_numeric(df.get("delay_dep"), errors="coerce")
    df["is_late_arr_5min"] = df["delay_arr_min"] >= 5
    df["is_late_dep_5min"] = df["delay_dep_min"] >= 5
    df = df.rename(columns={"train_serv": "train_service"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[CSV_COLUMNS].to_csv(OUT_CSV, index=False)
    print(f"[done] {len(df):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
