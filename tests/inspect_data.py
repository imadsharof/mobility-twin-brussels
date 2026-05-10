import pandas as pd
import json

with open("data/raw/punctuality.json", "r", encoding="utf-8") as f:
    punctuality_data = json.load(f)

df = pd.DataFrame(punctuality_data)

print("========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())