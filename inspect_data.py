import json
import pandas as pd
import os

# Load the punctuality data
data_path = "data/raw/punctuality.json"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run test_api.py and save the response first.")
    exit(1)

with open(data_path, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

print("=== Data Inspection ===")
print(f"Number of records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nSample records (first 5):")
print(df.head())

print("\n=== Delay Statistics ===")
if 'delay_arr' in df.columns:
    df['delay_arr_min'] = df['delay_arr'] / 60
    print(f"Average arrival delay: {df['delay_arr_min'].mean():.2f} minutes")
    print(f"Max arrival delay: {df['delay_arr_min'].max():.2f} minutes")
    print(f"Trains with >5 min delay: {len(df[df['delay_arr_min'] > 5])}")

if 'delay_dep' in df.columns:
    df['delay_dep_min'] = df['delay_dep'] / 60
    print(f"Average departure delay: {df['delay_dep_min'].mean():.2f} minutes")

print("\n=== Station Analysis ===")
if 'ptcar_lg_nm_nl' in df.columns:
    station_delays = df.groupby('ptcar_lg_nm_nl')['delay_arr_min'].mean().nlargest(10)
    print("Top 10 stations by average arrival delay:")
    print(station_delays)

print("\n=== Data Types ===")
print(df.dtypes)