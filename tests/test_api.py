import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("API_TOKEN")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}

endpoints = {
    "punctuality": "https://api.mobilitytwin.brussels/infrabel/punctuality",
    "operational_points": "https://api.mobilitytwin.brussels/infrabel/operational-points",
    "line_sections": "https://api.mobilitytwin.brussels/infrabel/line-sections"
}

os.makedirs("data/raw", exist_ok=True)

for name, url in endpoints.items():
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