import json
import pandas as pd
import os

# Load raw 30-day JSON data
with open("data/raw/punctuality_30_days.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Create proper datetime columns
df["planned_datetime_arr"] = pd.to_datetime(
    df["planned_date_arr"] + " " + df["planned_time_arr"],
    errors="coerce"
)

df["planned_datetime_dep"] = pd.to_datetime(
    df["planned_date_dep"] + " " + df["planned_time_dep"],
    errors="coerce"
)

df["real_datetime_arr"] = pd.to_datetime(
    df["real_date_arr"] + " " + df["real_time_arr"],
    errors="coerce"
)

df["real_datetime_dep"] = pd.to_datetime(
    df["real_date_dep"] + " " + df["real_time_dep"],
    errors="coerce"
)

# Delay values
df["delay_arr_sec"] = df["delay_arr"]
df["delay_dep_sec"] = df["delay_dep"]

df["delay_arr_min"] = df["delay_arr_sec"] / 60
df["delay_dep_min"] = df["delay_dep_sec"] / 60

# Date and time features
df["date"] = pd.to_datetime(df["datdep"], errors="coerce")
df["weekday"] = df["date"].dt.day_name()
df["hour_arr"] = df["planned_datetime_arr"].dt.hour
df["hour_dep"] = df["planned_datetime_dep"].dt.hour
df["is_weekend"] = df["date"].dt.weekday >= 5

# Late train flags
df["is_late_arr_5min"] = df["delay_arr_min"] >= 5
df["is_late_dep_5min"] = df["delay_dep_min"] >= 5

# Rename columns
clean_df = df.rename(columns={
    "train_serv": "train_service",
    "ptcar_lg_nm_nl": "station_name"
})

# Keep only useful columns
clean_df = clean_df[
    [
        "date",
        "train_no",
        "relation",
        "train_service",
        "station_name",
        "line_no_arr",
        "line_no_dep",
        "planned_datetime_arr",
        "planned_datetime_dep",
        "real_datetime_arr",
        "real_datetime_dep",
        "delay_arr_sec",
        "delay_dep_sec",
        "delay_arr_min",
        "delay_dep_min",
        "hour_arr",
        "hour_dep",
        "weekday",
        "is_weekend",
        "is_late_arr_5min",
        "is_late_dep_5min",
        "relation_direction"
    ]
]

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Save clean CSV
clean_df.to_csv("data/processed/clean_punctuality_30_days.csv", index=False)

print("Saved clean data to data/processed/clean_punctuality_30_days.csv")
print("Rows:", len(clean_df))
print("Columns:", clean_df.columns.tolist())