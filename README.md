# Mobility Twin Brussels — Infrabel rail delay dashboard

Interactive geo-temporal analysis of Belgian train arrival delays.
Live data from **mobilitytwin.brussels** is fetched, cleaned, aggregated and
rendered as a multi-tab Streamlit dashboard combining stations (points),
rail lines (polylines) and IC routes (relations).

> University project — *INFOH509 / Geospatial Web*, MA1.

---

## 1. What it does

Daily punctuality data from Infrabel is rich but hard to read in isolation.
This dashboard turns it into a tool to answer questions like:

- *Which stations are systematically late at rush hour?*
- *Are delays a Monday-morning effect, or constant throughout the week?*
- *Does train 2409 (IC 18) accumulate delay along its route, or does it
  depart late and stay late?*
- *Which IC / S / L relations are the worst performers?*
- *How do delays propagate across the network over 24 hours?*

Five tabs answer those questions for any date range the user picks.

---

## 2. Screenshots

### 🗺️ Map — current state of the network at a chosen hour
Stations (points) and physical rail lines (polylines) are colored by mean
arrival delay during the selected hour, averaged across every day of the
selected period. PyDeck layers, OpenStreetMap basemap.

![Map](docs/screenshots/01_map.png)

### 📊 Patterns — recurring weekly heatmap
Mean arrival delay per (weekday, hour). Grey cells = no train arrivals in
that bucket (e.g. 01:00–04:00 night service gap), **not** zero delay.

![Patterns](docs/screenshots/02_patterns.png)

### 🎬 Animation — 24-hour delay propagation
Plotly animated scatter map. Each frame is the network state at one hour
of the day, averaged across every day of the selected period. Press ▶.

![Animation](docs/screenshots/03_animation.png)

### 🔍 Drill-down — single-entity inspection
Pick a station, a train number, or an IC/S/L relation and see its hourly
delay profile, journey-along-the-route plot, or station-served breakdown.

![Drill-down](docs/screenshots/04_drilldown.png)

### 🚆 Trains — top recurring late trains
The `train_no` values whose arrivals are most often late (≥ threshold)
over the selected period — useful to flag timetable inputs that may need
adjustment.

![Trains](docs/screenshots/05_trains.png)

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  src/mobilitytwin/        ← SINGLE SOURCE OF TRUTH               │
│                                                                  │
│  ├─ api.py        Mobility Twin API client                       │
│  │                + per-day disk cache (data/raw/api_cache/)     │
│  │                + retry/backoff on 5xx, graceful failure       │
│  │                                                               │
│  └─ pipeline.py   records_to_dataframe                           │
│                   aggregate_station_hour / line_hour / relation_hour │
│                   filter_by_weekdays / holidays / stations / relations │
│                   join_stations / join_lines                     │
│                   load_operational_points / load_line_sections   │
│                   station_hourly_profile / train_journey_profile │
└──────────────────────────────────────────────────────────────────┘
       ▲                   ▲                    ▲
       │ imports           │ imports            │ imports
       │                   │                    │
   ┌───┴────────┐  ┌──────┴───────────┐  ┌────┴─────────────────────┐
   │  app.py    │  │ clean_data.py    │  │ prepare_*.py             │
   │ Streamlit  │  │ JSON → CSV       │  │ CSV → GeoJSON            │
   │ (live API) │  │ (one-shot CLI)   │  │ (one-shot CLI)           │
   └────────────┘  └──────────────────┘  └──────────────────────────┘
```

**Key property:** all data-processing logic (cleaning, aggregation,
geo-joins) lives in `src/mobilitytwin/`. Both the Streamlit dashboard and
the one-shot batch scripts are *clients* of this module — none of them
contains business logic. Single source of truth.

---

## 4. Data sources

All from [mobilitytwin.brussels](https://mobilitytwin.brussels/tag/INFRABEL/),
authenticated with an API token stored in `.env`:

| Endpoint | Used as | Refresh strategy |
|---|---|---|
| `/infrabel/punctuality?timestamp=<iso>` | One day of arrival/departure records | Per-day disk cache in `data/raw/api_cache/punctuality_YYYY-MM-DD.json` |
| `/infrabel/operational-points` | Station point geometries | Static — saved once in `data/raw/operational_points.json` |
| `/infrabel/line-sections` | Rail line polylines (1 017 segments) | Static — saved once in `data/raw/line_sections.json` |

Belgian public holidays are detected with the `holidays` Python package
(country code `BE`).

---

## 5. Quick start

### Requirements
- Python 3.12+
- An API token from mobilitytwin.brussels

### Install

```bash
git clone <repo-url>
cd mobility-twin-brussels
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file at the project root with your token:

```
API_TOKEN=<paste-your-mobilitytwin-token-here>
```

### Run

```bash
streamlit run app.py
```

The first time you select a month, the app fetches ~30 days from the API
(~30s with a progress bar). Subsequent loads of the same period are
instant thanks to the per-day disk cache.

---

## 6. Sidebar filters

| Filter | Scope | Effect |
|---|---|---|
| **Date range** | Network-wide | 6 quick presets (Today, Last 7d, Last 30d, This month, Last 3 months, Year-to-date) + custom start/end pickers |
| **Weekdays** | In-memory | Multi-select Mon–Sun |
| **Exclude BE holidays** | In-memory | Filters records whose `date` is a Belgian public holiday |
| **Stations** | In-memory | Multi-select with search (~600 options live-populated from data) |
| **Relations / IC routes** | In-memory | Multi-select (IC 14-1, IC 18, S5-1, L, etc.) |
| **Hour of day** | Map + ranking | 0–23 slider |
| **Min observations** | Ranking | Hides entities with too few data points from the top-10 |
| **Ranking mode** | Top-10 | Switch between **Stations**, **Lines** (physical Infrabel tracks), **Relations** (commercial IC/S/L routes) |
| **🔄 Force refresh** | API | Re-downloads the selected period and clears Streamlit caches |

The sidebar also shows a live indicator of how many days of the selected
range are already cached vs. need fetching, with an ETA.

---

## 7. Pipeline details

### Cleaning ([`src/mobilitytwin/pipeline.py::records_to_dataframe`](src/mobilitytwin/pipeline.py))

Raw API records → tidy DataFrame:

- `planned_datetime_arr/dep` built from `planned_date_arr + planned_time_arr`
- `delay_arr_min`, `delay_dep_min` (delays in minutes, can be negative for
  early arrivals)
- `hour_arr`, `hour_dep` (planned hour 0–23)
- `weekday` (`Monday` … `Sunday`)
- `is_weekend`, `is_holiday` (Belgian public holidays)
- `station_name` (renamed from `ptcar_lg_nm_nl`)

### Aggregation

Three parallel aggregations, all sharing the formula
**`mean = sum(delay) / count(records)`** within each group:

| Function | Group-by keys | Powers |
|---|---|---|
| `aggregate_station_hour` | `(station, hour_arr)` | Map points + station ranking |
| `aggregate_line_hour` | `(line_no_arr, hour_arr)` | Map polylines + line ranking |
| `aggregate_relation_hour` | `(relation, hour_arr)` | IC/S/L relation ranking |

Each row also carries `n_observations` (the count) so the dashboard can
filter unreliable buckets and weight the network-wide hourly average.

### Geo-join — physical lines

`line_sections.json` contains **1 017 `LineString` segments**. Several
segments share the same `linecnum` (integer line number, e.g. 36) but
finer track codes (`linecalfa = "36/1"`, `"36/2"`…). The CSV's
`line_no_arr` matches `linecalfa` exactly (118/150 codes, ~79% coverage).
`pipeline.load_line_sections` dissolves segments by `linecalfa` into one
`MultiLineString` per line code before the join.

---

## 8. Important note on the visualizations

**Every chart shows averages over the entire selected period, not a
single-day snapshot.** This is made explicit in every chart title
(e.g. *"10 worst stations at 08:00–08:59 — averaged over 7 days"*) and
in the caption right above each chart.

Grey / NaN cells in the heatmap mean **zero records in that
(weekday, hour) bucket** — not zero delay. Typical causes:
- Night service gap (01:00–04:00, no scheduled arrivals)
- A combination of filters that excludes all records for that bucket

---

## 9. Project layout

```
mobility-twin-brussels/
├── app.py                          # Streamlit UI (orchestration only)
├── clean_data.py                   # CLI: JSON → CSV  (thin wrapper)
├── prepare_delay_map_dataset.py    # CLI: CSV → stations GeoJSON  (thin wrapper)
├── prepare_lines_data.py           # CLI: CSV → lines GeoJSON  (thin wrapper)
├── eda.py, inspect_data.py         # ad-hoc exploration scripts
├── requirements.txt
├── .env                            # API_TOKEN=...  (gitignored)
│
├── src/mobilitytwin/
│   ├── __init__.py
│   ├── api.py                      # API client + retry + per-day cache
│   ├── pipeline.py                 # ALL data logic lives here
│   └── config.py
│
├── data/
│   ├── raw/
│   │   ├── operational_points.json     # static — fetched once
│   │   ├── line_sections.json          # static — fetched once
│   │   ├── punctuality_30_days.json    # bulk dump (legacy)
│   │   └── api_cache/                  # per-day cache populated by the dashboard
│   │       ├── punctuality_2026-05-01.json
│   │       ├── punctuality_2026-05-02.json
│   │       └── ...
│   └── processed/
│       ├── clean_punctuality_30_days.csv
│       ├── delays_by_station_hour.geojson
│       └── delays_by_segment_hour.geojson
│
├── tests/
│   └── test_api.py                 # one-shot script: fetch 30 days bulk
│
└── docs/
    ├── INFOH509 Project_ MobilityTwin.Brussels Demo.pdf
    └── screenshots/                # README assets
```

---

## 10. Robustness — what happens when things go wrong

| Failure mode | What the app does |
|---|---|
| HTTP 502/503/504/timeout | Up to 4 retries with exponential backoff (1s, 2s, 4s, 8s) |
| A specific day still fails after retries | Added to `failed_days`, skipped — the rest of the range still loads |
| `API_TOKEN` missing | Clear error: *"API_TOKEN missing. Put it in .env at the project root."* |
| Inverted date range | Sidebar error blocks rendering until fixed |
| No records after filters | Friendly warning (*"No records left after the current filters …"*) |

A persistent yellow banner at the top of the app lists any days that
failed; the **🔄 Force refresh** button in the sidebar retries them.

---

## 11. Dependencies

```
pandas
geopandas
folium
requests
python-dotenv
streamlit
pydeck
plotly
holidays
```

Pinned via `requirements.txt`.

---

## 12. Acknowledgements

- Data: [Infrabel](https://infrabel.be/) via
  [mobilitytwin.brussels](https://mobilitytwin.brussels/tag/INFRABEL/)
- Belgian holiday calendar: [`python-holidays`](https://pypi.org/project/holidays/)
- Maps: [Carto](https://carto.com/) basemap, OpenStreetMap contributors
