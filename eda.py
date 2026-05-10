import pandas as pd

# Load clean dataset
df = pd.read_csv("data/processed/clean_punctuality.csv")

print("=== Dataset Overview ===")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Delay Statistics ===")
print(df[["delay_arr_min", "delay_dep_min"]].describe())

print("\n=== Top 10 Stations by Records ===")
print(df["station_name"].value_counts().head(10))

print("\n=== Average Arrival Delay by Weekday ===")
print(df.groupby("weekday")["delay_arr_min"].mean().sort_values())

print("\n=== Average Departure Delay by Weekday ===")
print(df.groupby("weekday")["delay_dep_min"].mean().sort_values())

print("\n=== Late Train Percentage (Arrival > 5 min) ===")
late_arr_pct = df["is_late_arr_5min"].mean() * 100
print(f"{late_arr_pct:.2f}%")

print("\n=== Late Train Percentage (Departure > 5 min) ===")
late_dep_pct = df["is_late_dep_5min"].mean() * 100
print(f"{late_dep_pct:.2f}%")

print("\n=== Average Delay by Hour (Arrival) ===")
print(df.groupby("hour_arr")["delay_arr_min"].mean())

print("\n=== Average Delay by Hour (Departure) ===")
print(df.groupby("hour_dep")["delay_dep_min"].mean())

print("\n=== Date Range Check ===")
print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())
print("Unique days:", df["date"].nunique())