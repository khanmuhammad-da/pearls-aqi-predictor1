"""
========================================================================
PEARLS AQI PREDICTOR
STEP 25 — LIVE AQI INTELLIGENCE API
========================================================================

Purpose
-------
Backend/API layer for the existing STEP 23 Pearl Intelligence Dashboard.

IMPORTANT
---------
This script DOES NOT redesign or regenerate the dashboard.

It serves the existing:

    reports/pearl_intelligence_dashboard/
        pearl_intelligence_dashboard.html

and exposes live/local intelligence through API endpoints.

Architecture
------------

    STEP 23
        Existing Pearl Intelligence Dashboard
                    |
                    v
    STEP 25
        FastAPI backend
                    |
          +---------+---------+
          |         |         |
        AQI     Pollution   Forecast
          |         |         |
          +---------+---------+
                    |
             Existing files
             in reports/

Local URL
---------
    http://127.0.0.1:8000/

API documentation
-----------------
    http://127.0.0.1:8000/docs

Health
------
    http://127.0.0.1:8000/api/health

No model training.
No model selection.
No model retraining.
No dashboard generation.
No replacement UI.
========================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"

PEARL_DIR = (
    REPORTS_DIR
    / "pearl_intelligence_dashboard"
)

# Existing STEP 23 dashboard
DASHBOARD_HTML = (
    PEARL_DIR
    / "pearl_intelligence_dashboard.html"
)

# STEP 23 data
STEP23_DATA = (
    PEARL_DIR
    / "pearl_intelligence_dashboard_data.json"
)

STEP23_FORECAST = (
    PEARL_DIR
    / "pearl_intelligence_dashboard_forecast.csv"
)

STEP23_POLLUTION = (
    PEARL_DIR
    / "pearl_intelligence_dashboard_pollution.csv"
)

# STEP 25 / live intelligence data
LIVE_INTELLIGENCE = (
    PEARL_DIR
    / "pearl_live_aqi_intelligence.json"
)

LIVE_POLLUTION = (
    PEARL_DIR
    / "pearl_live_aqi_pollution.csv"
)

LIVE_ALERTS = (
    PEARL_DIR
    / "pearl_live_aqi_alerts.csv"
)

LIVE_SUMMARY = (
    PEARL_DIR
    / "pearl_live_aqi_summary.csv"
)

# Location
DEFAULT_CITY = "Lahore"
DEFAULT_COUNTRY = "Pakistan"

# Lahore Cantonment approximate coordinates.
# These are assumed project coordinates, not measured station coordinates.
DEFAULT_LATITUDE = 31.5339
DEFAULT_LONGITUDE = 74.3587

DEFAULT_STATION = "Lahore Cantonment"
DEFAULT_TIMEZONE = "Asia/Karachi"
DEFAULT_UTC_OFFSET = "+05:00"

# Cache
CACHE_ENABLED = True
CACHE_TTL_SECONDS = 30

_CACHE_LOCK = threading.Lock()

_CACHE: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="PEARLS AQI Predictor — Live Intelligence API",
    description=(
        "Live/local AQI intelligence backend for the existing "
        "Pearl Intelligence Dashboard."
    ),
    version="25.0.0",
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# STARTUP INFORMATION
# ============================================================================

START_TIME = time.time()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> str:
    """Return current UTC timestamp."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float safely."""

    if value is None:
        return None

    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to integer safely."""

    if value is None:
        return None

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy/datetime objects into JSON-safe Python objects.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    return value


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def file_mtime(path: Path) -> Optional[float]:
    if not file_exists(path):
        return None

    try:
        return path.stat().st_mtime
    except OSError:
        return None


def file_size(path: Path) -> Optional[int]:
    if not file_exists(path):
        return None

    try:
        return path.stat().st_size
    except OSError:
        return None


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file safely."""

    if not file_exists(path):
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {
            "data": data
        }

    except Exception as exc:
        return {
            "_error": str(exc),
            "_source": str(path),
        }


def read_csv_records(
    path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read CSV into JSON-safe records."""

    if not file_exists(path):
        return []

    try:
        df = pd.read_csv(path)

        if limit is not None:
            df = df.head(limit)

        records = df.to_dict(
            orient="records"
        )

        return json_safe(records)

    except Exception:
        return []


def read_csv_dataframe(path: Path) -> pd.DataFrame:
    """Read CSV into DataFrame."""

    if not file_exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


def recursive_find(
    obj: Any,
    keys: List[str],
) -> Any:
    """
    Find a value recursively in a JSON object.

    Used because Step 23/25 JSON structures may evolve.
    """

    if isinstance(obj, dict):

        normalized = {
            str(k).lower(): v
            for k, v in obj.items()
        }

        for key in keys:
            key_lower = key.lower()

            if key_lower in normalized:
                return normalized[key_lower]

        for value in obj.values():
            found = recursive_find(
                value,
                keys,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:
            found = recursive_find(
                value,
                keys,
            )

            if found is not None:
                return found

    return None


def category_from_aqi(aqi: Optional[float]) -> str:
    """US AQI category."""

    if aqi is None:
        return "Unknown"

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def risk_level_from_aqi(aqi: Optional[float]) -> str:
    """Compact risk classification."""

    if aqi is None:
        return "unknown"

    if aqi <= 50:
        return "low"

    if aqi <= 100:
        return "moderate"

    if aqi <= 150:
        return "elevated"

    if aqi <= 200:
        return "high"

    if aqi <= 300:
        return "very_high"

    return "hazardous"


def find_timestamp_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """Find timestamp-like column."""

    if df.empty:
        return None

    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "ds",
        "forecast_timestamp",
        "prediction_timestamp",
        "utc_timestamp",
    ]

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in df.columns:

        name = str(column).lower()

        if (
            "timestamp" in name
            or "datetime" in name
        ):
            return column

    return None


def find_aqi_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """Find AQI/prediction column."""

    if df.empty:
        return None

    candidates = [
        "predicted_aqi",
        "prediction",
        "predicted_us_aqi",
        "us_aqi",
        "aqi",
        "forecast_aqi",
        "yhat",
    ]

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in df.columns:

        name = str(column).lower()

        if "aqi" in name:
            return column

    return None


# ============================================================================
# LOCATION
# ============================================================================

def build_location() -> Dict[str, Any]:
    """
    Build project location context.

    Coordinates are explicitly assumed for Lahore Cantonment.
    """

    return {
        "city": DEFAULT_CITY,
        "country": DEFAULT_COUNTRY,
        "latitude": DEFAULT_LATITUDE,
        "longitude": DEFAULT_LONGITUDE,
        "station": DEFAULT_STATION,
        "timezone": DEFAULT_TIMEZONE,
        "utc_offset": DEFAULT_UTC_OFFSET,
        "location_source": "project_assumption",
        "coordinate_status": "ASSUMED",
        "station_status": "ASSUMED",
        "updated_at": utc_now(),
    }


# ============================================================================
# FORECAST
# ============================================================================

def build_forecast(
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build forecast API response."""

    df = read_csv_dataframe(
        STEP23_FORECAST
    )

    if df.empty:
        df = read_csv_dataframe(
            PEARL_DIR
            / "pearl_forecast.csv"
        )

    if df.empty:
        return {
            "available": False,
            "rows": [],
            "count": 0,
        }

    timestamp_column = find_timestamp_column(df)
    aqi_column = find_aqi_column(df)

    if timestamp_column:
        timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
            utc=True,
        )

        df = df.copy()

        df["_api_timestamp"] = timestamps

    if aqi_column:
        df = df.copy()

        df["_api_aqi"] = pd.to_numeric(
            df[aqi_column],
            errors="coerce",
        )

        df["_api_category"] = df[
            "_api_aqi"
        ].apply(category_from_aqi)

    if limit is not None:
        output_df = df.head(limit).copy()

    else:
        output_df = df.copy()

    records = []

    for _, row in output_df.iterrows():

        record = {}

        if timestamp_column:
            timestamp = row.get(
                "_api_timestamp"
            )

            if pd.notna(timestamp):
                record["timestamp"] = (
                    timestamp.isoformat()
                )

        if aqi_column:
            aqi = safe_float(
                row.get("_api_aqi")
            )

            record["predicted_aqi"] = aqi
            record["category"] = (
                category_from_aqi(aqi)
            )

        for column in df.columns:

            if str(column).startswith("_api_"):
                continue

            value = row.get(column)

            if pd.isna(value):
                value = None

            record[str(column)] = json_safe(
                value
            )

        records.append(record)

    aqi_values = []

    if "_api_aqi" in df.columns:
        aqi_values = [
            safe_float(v)
            for v in df["_api_aqi"].tolist()
        ]

        aqi_values = [
            v for v in aqi_values
            if v is not None
        ]

    result = {
        "available": True,
        "count": len(records),
        "rows": records,
    }

    if aqi_values:

        result["minimum"] = min(aqi_values)
        result["maximum"] = max(aqi_values)
        result["mean"] = sum(aqi_values) / len(
            aqi_values
        )

        sorted_values = sorted(aqi_values)

        middle = len(sorted_values) // 2

        if len(sorted_values) % 2:
            median = sorted_values[middle]
        else:
            median = (
                sorted_values[middle - 1]
                + sorted_values[middle]
            ) / 2

        result["median"] = median

        result["dominant_category"] = (
            category_from_aqi(
                result["mean"]
            )
        )

        result["risk_level"] = (
            risk_level_from_aqi(
                result["mean"]
            )
        )

    return json_safe(result)


# ============================================================================
# POLLUTION
# ============================================================================

POLLUTANT_ALIASES = {
    "PM2.5": [
        "pm2_5",
        "pm25",
        "pm2.5",
        "pm_2_5",
    ],
    "PM10": [
        "pm10",
        "pm_10",
    ],
    "O3": [
        "o3",
        "ozone",
    ],
    "NO2": [
        "no2",
        "nitrogen_dioxide",
        "nitrogen dioxide",
    ],
    "SO2": [
        "so2",
        "sulphur_dioxide",
        "sulfur_dioxide",
        "sulphur dioxide",
        "sulfur dioxide",
    ],
    "CO": [
        "co",
        "carbon_monoxide",
        "carbon monoxide",
    ],
}


def discover_pollutants(
    df: pd.DataFrame,
) -> Dict[str, Optional[str]]:
    """Discover pollutant columns."""

    if df.empty:
        return {
            pollutant: None
            for pollutant in POLLUTANT_ALIASES
        }

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    discovered = {}

    for pollutant, aliases in POLLUTANT_ALIASES.items():

        found = None

        for alias in aliases:

            if alias.lower() in lower_map:
                found = lower_map[
                    alias.lower()
                ]
                break

        if found is None:

            for column in df.columns:

                normalized = (
                    str(column)
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                if any(
                    alias.lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                    == normalized
                    for alias in aliases
                ):
                    found = column
                    break

        discovered[pollutant] = found

    return discovered


def build_pollution(
    limit: int = 168,
) -> Dict[str, Any]:
    """Build pollution intelligence response."""

    df = read_csv_dataframe(
        LIVE_POLLUTION
    )

    if df.empty:
        df = read_csv_dataframe(
            STEP23_POLLUTION
        )

    if df.empty:
        return {
            "available": False,
            "count": 0,
            "pollutants": {},
            "rows": [],
        }

    timestamp_column = find_timestamp_column(df)

    pollutants = discover_pollutants(df)

    result_rows = []

    work = df.copy()

    if timestamp_column:

        work["_api_timestamp"] = pd.to_datetime(
            work[timestamp_column],
            errors="coerce",
            utc=True,
        )

        work = work.sort_values(
            "_api_timestamp"
        )

    work = work.tail(limit)

    for _, row in work.iterrows():

        record = {}

        if timestamp_column:

            timestamp = row.get(
                "_api_timestamp"
            )

            if pd.notna(timestamp):
                record["timestamp"] = (
                    timestamp.isoformat()
                )

        for pollutant, column in pollutants.items():

            if column is None:
                record[pollutant] = None
                continue

            record[pollutant] = safe_float(
                row.get(column)
            )

        result_rows.append(record)

    statistics = {}

    for pollutant, column in pollutants.items():

        if column is None:
            statistics[pollutant] = {
                "available": False
            }
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:

            statistics[pollutant] = {
                "available": False
            }

            continue

        statistics[pollutant] = {
            "available": True,
            "column": str(column),
            "latest": safe_float(
                values.iloc[-1]
            ),
            "mean": safe_float(
                values.mean()
            ),
            "minimum": safe_float(
                values.min()
            ),
            "maximum": safe_float(
                values.max()
            ),
            "percentile_50": safe_float(
                values.quantile(0.50)
            ),
            "percentile_75": safe_float(
                values.quantile(0.75)
            ),
            "percentile_90": safe_float(
                values.quantile(0.90)
            ),
            "percentile_95": safe_float(
                values.quantile(0.95)
            ),
        }

    return {
        "available": True,
        "count": len(df),
        "returned_rows": len(result_rows),
        "pollutants": statistics,
        "rows": result_rows,
        "updated_at": utc_now(),
    }


# ============================================================================
# LIVE AQI
# ============================================================================

def build_live_aqi() -> Dict[str, Any]:
    """
    Build latest AQI intelligence.

    Priority:
        1. live intelligence JSON
        2. live summary CSV
        3. STEP 23 forecast
    """

    live_json = read_json(
        LIVE_INTELLIGENCE
    )

    location = build_location()

    # ------------------------------------------------------------------------
    # Try explicit current AQI fields from live JSON
    # ------------------------------------------------------------------------

    current_aqi = recursive_find(
        live_json,
        [
            "current_aqi",
            "latest_aqi",
            "current_us_aqi",
            "latest_us_aqi",
        ],
    )

    current_aqi = safe_float(
        current_aqi
    )

    # ------------------------------------------------------------------------
    # Try summary CSV
    # ------------------------------------------------------------------------

    if current_aqi is None:

        summary = read_csv_dataframe(
            LIVE_SUMMARY
        )

        if not summary.empty:

            aqi_column = find_aqi_column(
                summary
            )

            if aqi_column:

                values = pd.to_numeric(
                    summary[aqi_column],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    current_aqi = safe_float(
                        values.iloc[-1]
                    )

    # ------------------------------------------------------------------------
    # Fallback to latest forecast
    # ------------------------------------------------------------------------

    forecast = build_forecast(
        limit=72
    )

    if current_aqi is None:

        rows = forecast.get(
            "rows",
            []
        )

        if rows:

            first = rows[0]

            current_aqi = safe_float(
                first.get(
                    "predicted_aqi"
                )
            )

    # ------------------------------------------------------------------------
    # Determine current category
    # ------------------------------------------------------------------------

    category = category_from_aqi(
        current_aqi
    )

    result = {
        "available": current_aqi is not None,
        "current_aqi": current_aqi,
        "category": category,
        "risk_level": risk_level_from_aqi(
            current_aqi
        ),
        "location": location,
        "source": (
            "live_intelligence"
            if current_aqi is not None
            else None
        ),
        "updated_at": utc_now(),
    }

    # Add useful intelligence fields from JSON
    if live_json:

        for key in [
            "dominant_pollutant",
            "forecast_trend",
            "forecast_mean",
            "forecast_maximum",
            "pollution_events",
            "high_severity_events",
            "elevated_events",
        ]:

            value = recursive_find(
                live_json,
                [key],
            )

            if value is not None:
                result[key] = json_safe(
                    value
                )

    return json_safe(result)


# ============================================================================
# ALERTS
# ============================================================================

def build_alerts() -> Dict[str, Any]:
    """Return live AQI/pollution alerts."""

    records = read_csv_records(
        LIVE_ALERTS
    )

    return {
        "available": file_exists(
            LIVE_ALERTS
        ),
        "count": len(records),
        "alerts": records,
        "updated_at": utc_now(),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary() -> Dict[str, Any]:
    """Build compact dashboard summary."""

    live = build_live_aqi()
    pollution = build_pollution(
        limit=1
    )
    forecast = build_forecast(
        limit=72
    )

    pollutants = pollution.get(
        "pollutants",
        {}
    )

    dominant_pollutant = None
    dominant_percentile = None

    ranked = []

    for pollutant, data in pollutants.items():

        if not isinstance(data, dict):
            continue

        percentile = safe_float(
            data.get(
                "percentile_90"
            )
        )

        if percentile is not None:
            ranked.append(
                (
                    pollutant,
                    percentile,
                )
            )

    if ranked:

        ranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        dominant_pollutant = ranked[0][0]
        dominant_percentile = ranked[0][1]

    forecast_mean = forecast.get(
        "mean"
    )

    forecast_maximum = forecast.get(
        "maximum"
    )

    return {
        "location": build_location(),
        "current": live,
        "forecast": {
            "count": forecast.get(
                "count",
                0,
            ),
            "minimum": forecast.get(
                "minimum"
            ),
            "maximum": forecast_maximum,
            "mean": forecast_mean,
            "median": forecast.get(
                "median"
            ),
            "category": forecast.get(
                "dominant_category"
            ),
            "risk_level": forecast.get(
                "risk_level"
            ),
        },
        "pollution": {
            "available": len(
                pollutants
            ),
            "dominant_pollutant": (
                dominant_pollutant
            ),
            "dominant_percentile": (
                dominant_percentile
            ),
            "ingredients": pollutants,
        },
        "alerts": {
            "count": len(
                build_alerts().get(
                    "alerts",
                    []
                )
            )
        },
        "generated_at": utc_now(),
    }


# ============================================================================
# CACHE
# ============================================================================

def source_signature() -> str:
    """
    Generate a signature from source files.

    If any source changes, cached API data becomes stale.
    """

    files = [
        STEP23_DATA,
        STEP23_FORECAST,
        STEP23_POLLUTION,
        LIVE_INTELLIGENCE,
        LIVE_POLLUTION,
        LIVE_ALERTS,
        LIVE_SUMMARY,
    ]

    parts = []

    for path in files:

        parts.append(
            "|".join(
                [
                    str(path),
                    str(file_mtime(path)),
                    str(file_size(path)),
                ]
            )
        )

    raw = "\n".join(parts)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def cache_get(
    key: str,
) -> Optional[Any]:

    if not CACHE_ENABLED:
        return None

    with _CACHE_LOCK:

        entry = _CACHE.get(key)

        if not entry:
            return None

        age = time.time() - entry["created"]

        if age > CACHE_TTL_SECONDS:
            return None

        if (
            entry.get("signature")
            != source_signature()
        ):
            return None

        return entry["data"]


def cache_set(
    key: str,
    data: Any,
) -> None:

    if not CACHE_ENABLED:
        return

    with _CACHE_LOCK:

        _CACHE[key] = {
            "created": time.time(),
            "signature": source_signature(),
            "data": data,
        }


def cached(
    key: str,
    builder,
):
    """Generic cache wrapper."""

    value = cache_get(key)

    if value is not None:
        return value

    value = builder()

    cache_set(
        key,
        value,
    )

    return value


# ============================================================================
# ROOT — EXISTING STEP 23 DASHBOARD
# ============================================================================

@app.get(
    "/",
    include_in_schema=False,
)
def root_dashboard():
    """
    SERVE THE EXISTING STEP 23 DASHBOARD.

    No HTML is generated here.
    No CSS is generated here.
    No JavaScript is generated here.
    """

    if not file_exists(
        DASHBOARD_HTML
    ):

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": (
                    "Step 23 Pearl Intelligence Dashboard "
                    "was not found."
                ),
                "expected_file": str(
                    DASHBOARD_HTML
                ),
            },
        )

    return FileResponse(
        path=DASHBOARD_HTML,
        media_type="text/html",
        headers={
            "Cache-Control": (
                "no-cache, no-store, "
                "must-revalidate"
            ),
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get(
    "/dashboard",
    include_in_schema=False,
)
def dashboard_alias():
    """Alternative URL for the same Step 23 dashboard."""

    return root_dashboard()


@app.get(
    "/pearl",
    include_in_schema=False,
)
def pearl_alias():
    """Alternative URL for the same Step 23 dashboard."""

    return root_dashboard()


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/api/health")
def api_health():
    """Server and data health."""

    source_files = {
        "dashboard": DASHBOARD_HTML,
        "step23_data": STEP23_DATA,
        "step23_forecast": STEP23_FORECAST,
        "step23_pollution": STEP23_POLLUTION,
        "live_intelligence": LIVE_INTELLIGENCE,
        "live_pollution": LIVE_POLLUTION,
        "live_alerts": LIVE_ALERTS,
        "live_summary": LIVE_SUMMARY,
    }

    files = {}

    for name, path in source_files.items():

        files[name] = {
            "exists": file_exists(path),
            "path": str(path),
            "size": file_size(path),
            "modified": (
                datetime.fromtimestamp(
                    file_mtime(path),
                    tz=timezone.utc,
                ).isoformat()
                if file_mtime(path)
                else None
            ),
        }

    return {
        "status": "ok",
        "service": (
            "PEARLS AQI Predictor "
            "Live Intelligence API"
        ),
        "version": "25.0.0",
        "dashboard": "STEP 23",
        "api_layer": "STEP 25",
        "cache_enabled": CACHE_ENABLED,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "uptime_seconds": (
            time.time() - START_TIME
        ),
        "location": build_location(),
        "files": files,
        "server_time_utc": utc_now(),
    }


# ============================================================================
# LOCATION API
# ============================================================================

@app.get("/api/location")
def api_location():
    """Location intelligence."""

    return cached(
        "location",
        build_location,
    )


# ============================================================================
# LIVE AQI API
# ============================================================================

@app.get("/api/live-aqi")
def api_live_aqi():
    """Current/latest AQI intelligence."""

    return cached(
        "live_aqi",
        build_live_aqi,
    )


@app.get("/api/current-aqi")
def api_current_aqi():
    """Alias for live AQI."""

    return api_live_aqi()


# ============================================================================
# FORECAST API
# ============================================================================

@app.get("/api/forecast")
def api_forecast(
    limit: int = Query(
        default=72,
        ge=1,
        le=720,
    ),
):
    """AQI forecast."""

    # Forecast is cached separately by requested size.
    key = f"forecast_{limit}"

    return cached(
        key,
        lambda: build_forecast(
            limit=limit
        ),
    )


@app.get("/api/forecast/72h")
def api_forecast_72h():
    """72-hour forecast."""

    return api_forecast(
        limit=72
    )


# ============================================================================
# POLLUTION API
# ============================================================================

@app.get("/api/pollution")
def api_pollution(
    limit: int = Query(
        default=168,
        ge=1,
        le=8760,
    ),
):
    """Pollution ingredients and recent history."""

    key = f"pollution_{limit}"

    return cached(
        key,
        lambda: build_pollution(
            limit=limit
        ),
    )


@app.get("/api/pollution/latest")
def api_pollution_latest():
    """Latest available pollutant readings."""

    return api_pollution(
        limit=1
    )


# ============================================================================
# ALERT API
# ============================================================================

@app.get("/api/alerts")
def api_alerts():
    """AQI/pollution alerts."""

    return cached(
        "alerts",
        build_alerts,
    )


# ============================================================================
# SUMMARY API
# ============================================================================

@app.get("/api/summary")
def api_summary():
    """Compact Pearl dashboard intelligence."""

    return cached(
        "summary",
        build_summary,
    )


@app.get("/api/intelligence")
def api_intelligence():
    """
    Main intelligence endpoint.

    Useful for an external frontend, mobile application,
    deployment layer, or future dashboard integration.
    """

    return {
        "location": cached(
            "location",
            build_location,
        ),
        "current": cached(
            "live_aqi",
            build_live_aqi,
        ),
        "forecast": cached(
            "forecast_72",
            lambda: build_forecast(
                limit=72
            ),
        ),
        "pollution": cached(
            "pollution_168",
            lambda: build_pollution(
                limit=168
            ),
        ),
        "alerts": cached(
            "alerts",
            build_alerts,
        ),
        "generated_at": utc_now(),
    }


# ============================================================================
# DATA FILE API
# ============================================================================

@app.get("/api/raw/step23")
def api_raw_step23():
    """Return the original Step 23 dashboard JSON."""

    if not file_exists(
        STEP23_DATA
    ):
        raise HTTPException(
            status_code=404,
            detail="Step 23 dashboard data not found.",
        )

    return JSONResponse(
        content=read_json(
            STEP23_DATA
        )
    )


@app.get("/api/raw/live")
def api_raw_live():
    """Return the raw Step 25 live intelligence JSON."""

    if not file_exists(
        LIVE_INTELLIGENCE
    ):
        raise HTTPException(
            status_code=404,
            detail="Live intelligence JSON not found.",
        )

    return JSONResponse(
        content=read_json(
            LIVE_INTELLIGENCE
        )
    )


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

@app.post("/api/cache/clear")
def api_cache_clear():
    """Clear in-memory API cache."""

    with _CACHE_LOCK:
        _CACHE.clear()

    return {
        "status": "cleared",
        "cache_enabled": CACHE_ENABLED,
        "timestamp": utc_now(),
    }


@app.get("/api/cache")
def api_cache_status():
    """Inspect cache state."""

    with _CACHE_LOCK:

        entries = {}

        for key, value in _CACHE.items():

            age = (
                time.time()
                - value["created"]
            )

            entries[key] = {
                "age_seconds": round(
                    age,
                    3,
                ),
                "valid": (
                    age
                    <= CACHE_TTL_SECONDS
                    and value.get(
                        "signature"
                    )
                    == source_signature()
                ),
            }

    return {
        "enabled": CACHE_ENABLED,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "entries": entries,
        "source_signature": (
            source_signature()
        ),
    }


# ============================================================================
# VERSION
# ============================================================================

@app.get("/api/version")
def api_version():
    return {
        "project": "PEARLS AQI Predictor",
        "step": 25,
        "name": "Live AQI Intelligence API",
        "dashboard": "Pearl Intelligence Dashboard",
        "dashboard_step": 23,
        "api_version": "25.0.0",
        "model_training": False,
        "model_selection": False,
        "model_retraining": False,
        "dashboard_redesign": False,
        "timestamp": utc_now(),
    }


# ============================================================================
# ERROR HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc: Exception,
):
    """
    Return useful JSON instead of an unhelpful server traceback.
    """

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
            "path": str(
                request.url.path
            ),
            "timestamp": utc_now(),
        },
    )


# ============================================================================
# STARTUP
# ============================================================================

def print_header() -> None:

    print()
    print("=" * 72)
    print("PEARLS AQI PREDICTOR")
    print("=" * 72)
    print("STEP 25 — LIVE AQI INTELLIGENCE API")
    print("=" * 72)

    print(
        f"Base directory              : {BASE_DIR}"
    )

    print(
        f"Dashboard directory         : {PEARL_DIR}"
    )

    print(
        f"Step 23 dashboard           : "
        f"{'FOUND' if file_exists(DASHBOARD_HTML) else 'MISSING'}"
    )

    print(
        f"Live intelligence           : "
        f"{'FOUND' if file_exists(LIVE_INTELLIGENCE) else 'MISSING'}"
    )

    print(
        f"Live pollution              : "
        f"{'FOUND' if file_exists(LIVE_POLLUTION) else 'MISSING'}"
    )

    print(
        f"Live alerts                 : "
        f"{'FOUND' if file_exists(LIVE_ALERTS) else 'MISSING'}"
    )

    print(
        f"Live summary                : "
        f"{'FOUND' if file_exists(LIVE_SUMMARY) else 'MISSING'}"
    )

    print()
    print("Dashboard                   : STEP 23 EXISTING HTML")
    print("Dashboard redesign          : NO")
    print("Model training              : NO")
    print("Model selection             : NO")
    print("Model retraining            : NO")
    print(
        f"Caching                     : "
        f"{'ENABLED' if CACHE_ENABLED else 'DISABLED'}"
    )
    print(
        f"Cache TTL                   : "
        f"{CACHE_TTL_SECONDS} seconds"
    )

    print("=" * 72)


def validate_files() -> None:

    print()
    print("=" * 72)
    print("VALIDATING STEP 25")
    print("=" * 72)

    required = {
        "Step 23 dashboard": DASHBOARD_HTML,
        "Step 23 data": STEP23_DATA,
        "Step 23 forecast": STEP23_FORECAST,
    }

    optional = {
        "Live intelligence": LIVE_INTELLIGENCE,
        "Live pollution": LIVE_POLLUTION,
        "Live alerts": LIVE_ALERTS,
        "Live summary": LIVE_SUMMARY,
    }

    print()
    print("Required files:")

    for name, path in required.items():

        status = (
            "PASS"
            if file_exists(path)
            else "FAIL"
        )

        print(
            f"{name:<28}: {status}"
        )

    print()
    print("Optional live files:")

    for name, path in optional.items():

        status = (
            "FOUND"
            if file_exists(path)
            else "NOT FOUND"
        )

        print(
            f"{name:<28}: {status}"
        )

    print()
    print("Dashboard URL:")
    print(
        "http://127.0.0.1:8000/"
    )

    print()
    print("API documentation:")
    print(
        "http://127.0.0.1:8000/docs"
    )

    print()
    print("Health:")
    print(
        "http://127.0.0.1:8000/api/health"
    )

    print("=" * 72)


def main() -> None:

    print_header()

    validate_files()

    print()
    print("=" * 72)
    print("STARTING PEARL INTELLIGENCE SERVER")
    print("=" * 72)

    print()
    print(
        "Dashboard:"
    )
    print(
        "http://127.0.0.1:8000/"
    )

    print()
    print(
        "Alternative:"
    )
    print(
        "http://127.0.0.1:8000/dashboard"
    )

    print()
    print(
        "API:"
    )
    print(
        "http://127.0.0.1:8000/api/intelligence"
    )

    print()
    print(
        "Swagger:"
    )
    print(
        "http://127.0.0.1:8000/docs"
    )

    print()
    print(
        "Press CTRL+C to stop."
    )

    print("=" * 72)
    print()

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()