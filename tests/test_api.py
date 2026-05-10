import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

TOKEN = os.getenv("API_TOKEN")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}

os.makedirs("data/raw", exist_ok=True)

# Static endpoints, fetch once
static_endpoints = {
    "operational_points": "https://api.mobilitytwin.brussels/infrabel/operational-points",
    "line_sections": "https://api.mobilitytwin.brussels/infrabel/line-sections"
}

for name, url in static_endpoints.items():
    print(f"\nFetching {name}...")
    response = requests.get(url, headers=headers, timeout=60)

    print("Status:", response.status_code)

    if response.status_code == 200:
        data = response.json()
        output_path = f"data/raw/{name}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path}")

        if isinstance(data, list):
            print("Records:", len(data))
        elif isinstance(data, dict) and "features" in data:
            print("Features:", len(data["features"]))
    else:
        print(response.text[:500])


# Punctuality endpoint, fetch last 30 days using timestamp
punctuality_url = "https://api.mobilitytwin.brussels/infrabel/punctuality"

all_punctuality_data = []

today = datetime.now(timezone.utc)

for i in range(30):
    current_datetime = today - timedelta(days=i)

    # Use noon UTC to avoid midnight/date-boundary problems
    current_datetime = current_datetime.replace(
        hour=12,
        minute=0,
        second=0,
        microsecond=0
    )

    date_str = current_datetime.date().strftime("%Y-%m-%d")

    print(f"\nFetching punctuality for {date_str}...")

    params = {
        "timestamp": current_datetime.isoformat()
    }

    response = requests.get(
        punctuality_url,
        headers=headers,
        params=params,
        timeout=60
    )

    print("Status:", response.status_code)

    if response.status_code == 200:
        data = response.json()

        if isinstance(data, list):
            for record in data:
                record["api_timestamp"] = current_datetime.isoformat()
                record["api_requested_date"] = date_str

            all_punctuality_data.extend(data)
            print("Records:", len(data))
        else:
            print("Unexpected response format")
    else:
        print(response.text[:500])


output_path = "data/raw/punctuality_30_days.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_punctuality_data, f, ensure_ascii=False, indent=2)

print(f"\nSaved: {output_path}")
print("Total records:", len(all_punctuality_data))