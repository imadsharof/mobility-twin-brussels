"""Mobility Twin Brussels — Infrabel API client with per-day disk cache.

Endpoint:
    GET https://api.mobilitytwin.brussels/infrabel/punctuality?timestamp=<iso>

Each call returns the records for the day of the supplied timestamp.
We send noon UTC to avoid midnight/date-boundary edge cases.

Cache layout (one file per day, replayed without HTTP next time):
    data/raw/api_cache/punctuality_YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Server-side transient errors that warrant a retry.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "api_cache"

PUNCTUALITY_URL = "https://api.mobilitytwin.brussels/infrabel/punctuality"
STATIC_REFERENCE_ENDPOINTS: dict[str, tuple[str, Path]] = {
    "operational_points": (
        "https://api.mobilitytwin.brussels/infrabel/operational-points",
        PROJECT_ROOT / "data" / "raw" / "operational_points.json",
    ),
    "line_sections": (
        "https://api.mobilitytwin.brussels/infrabel/line-sections",
        PROJECT_ROOT / "data" / "raw" / "line_sections.json",
    ),
}

TOKEN_ENV_NAMES = ("API_TOKEN", "MOBILITYTWIN_API_TOKEN")


def _api_token() -> str:
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if not token or not token.strip():
            continue
        token = token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token:
            return token
    expected = " or ".join(TOKEN_ENV_NAMES)
    raise RuntimeError(
        f"API token missing. Put {expected}=... in {PROJECT_ROOT / '.env'}"
    )


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_token()}", "Accept": "application/json"}


def _request_json(url: str, params: dict[str, str] | None = None) -> object:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers=_auth_headers(),
                params=params,
                timeout=60,
            )
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason}", response=resp
                )
            resp.raise_for_status()
            return resp.json()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as exc:
            last_error = exc
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            retryable = status_code in RETRY_STATUS or isinstance(
                exc,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ),
            )
            if attempt == MAX_RETRIES - 1 or not retryable:
                break
            sleep_for = BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "Fetch failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, MAX_RETRIES, exc, sleep_for,
            )
            time.sleep(sleep_for)

    assert last_error is not None
    raise last_error


def _cache_path(day: date) -> Path:
    return CACHE_DIR / f"punctuality_{day.isoformat()}.json"


def fetch_punctuality_day(day: date, force: bool = False) -> list[dict]:
    """Return the punctuality records for one day.

    Disk-cached. Retries automatically on transient server errors
    (HTTP 429/5xx, timeouts, connection drops) with exponential backoff.
    """
    path = _cache_path(day)
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    ts = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=timezone.utc)
    params = {"timestamp": ts.isoformat()}

    data = _request_json(PUNCTUALITY_URL, params=params)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response for {day}: {type(data).__name__}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def ensure_reference_data(force: bool = False) -> dict[str, Path]:
    """Download the static GeoJSON references if they are absent locally."""
    paths: dict[str, Path] = {}
    for name, (url, path) in STATIC_REFERENCE_ENDPOINTS.items():
        if path.exists() and not force:
            paths[name] = path
            continue

        data = _request_json(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        paths[name] = path
    return paths


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def fetch_punctuality_range(
    start: date,
    end: date,
    on_progress: Callable[[int, int, date], None] | None = None,
    force: bool = False,
    failed_errors: dict[date, str] | None = None,
) -> tuple[list[dict], list[date]]:
    """Fetch every day in [start, end] (inclusive).

    Returns (records, failed_days). A day that exhausts all retries is
    reported in `failed_days` and skipped — the rest of the range still
    loads. Re-running with Force refresh retries the failed days.
    """
    days = list(_daterange(start, end))
    out: list[dict] = []
    failed: list[date] = []
    for i, d in enumerate(days, start=1):
        try:
            out.extend(fetch_punctuality_day(d, force=force))
        except Exception as exc:  # noqa: BLE001 — surface the day, not crash
            logger.error("Giving up on %s: %s", d, exc)
            failed.append(d)
            if failed_errors is not None:
                failed_errors[d] = f"{type(exc).__name__}: {exc}"
        if on_progress is not None:
            on_progress(i, len(days), d)
    return out, failed


def month_range(year: int, month: int) -> tuple[date, date]:
    """First and last day of a given (year, month)."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def cached_days() -> list[date]:
    """Return days already present in the disk cache, sorted."""
    if not CACHE_DIR.exists():
        return []
    out: list[date] = []
    for p in CACHE_DIR.glob("punctuality_*.json"):
        try:
            out.append(date.fromisoformat(p.stem.removeprefix("punctuality_")))
        except ValueError:
            continue
    return sorted(out)
