"""
PEARLS AQI PREDICTOR
STEP 21 — LOCATION + POLLUTION INTELLIGENCE

Purpose
-------
Build a reusable intelligence layer between the production forecast/dashboard
and future dashboard presentation layers.

Inputs
------
1. Step 20 production dashboard forecast:
   reports/production_dashboard_v2/production_dashboard_forecast.csv

2. Pollution/features dataset:
   data/processed/lahore_features_hourly.csv

3. Optional Step 20 metadata:
   reports/production_dashboard_v2/production_dashboard_data.json
   reports/production_dashboard_v2/production_dashboard_v2_results.json

Outputs
-------
reports/location_pollution_intelligence/
    location_intelligence.json
    pollution_intelligence.csv
    pollution_hourly.csv
    aqi_intelligence.csv
    pollution_events.csv
    location_pollution_package.json
    intelligence_report.json
    cache/
        location_hash.json
        pollution_hash.json
        intelligence_hash.json
        package_hash.json

Design principles
-----------------
- No model training
- No model selection
- No hyperparameter tuning
- No validation/test claims
- No fabricated coordinates
- No fabricated pollutant values
- Dataset-derived location identity only
- Component-level caching
- Deterministic output
- Explicit validation
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TARGET = "us_aqi"
FORECAST_HORIZON = 72

POLLUTION_SOURCE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lahore_features_hourly.csv"
)

STEP20_DIR = (
    BASE_DIR
    / "reports"
    / "production_dashboard_v2"
)

FORECAST_SOURCE = (
    STEP20_DIR
    / "production_dashboard_forecast.csv"
)

STEP20_DATA_SOURCE = (
    STEP20_DIR
    / "production_dashboard_data.json"
)

STEP20_REPORT_SOURCE = (
    STEP20_DIR
    / "production_dashboard_v2_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "location_pollution_intelligence"
)

CACHE_DIR = OUTPUT_DIR / "cache"


# ============================================================================
# POLLUTANT DETECTION
# ============================================================================

POLLUTANT_ALIASES = {
    "PM2.5": [
        "pm2.5",
        "pm25",
        "pm_25",
        "pm2_5",
        "pm25_value",
        "pm25_concentration",
        "pm2.5_concentration",
    ],
    "PM10": [
        "pm10",
        "pm_10",
        "pm10_value",
        "pm10_concentration",
    ],
    "O3": [
        "o3",
        "o_3",
        "ozone",
        "o3_value",
        "ozone_concentration",
    ],
    "NO2": [
        "no2",
        "no_2",
        "nitrogen_dioxide",
        "no2_value",
        "no2_concentration",
    ],
    "SO2": [
        "so2",
        "so_2",
        "sulfur_dioxide",
        "sulphur_dioxide",
        "so2_value",
        "so2_concentration",
    ],
    "CO": [
        "co",
        "carbon_monoxide",
        "co_value",
        "co_concentration",
    ],
}


LOCATION_ALIASES = {
    "city": [
        "city",
        "location_city",
        "municipality",
    ],
    "country": [
        "country",
        "location_country",
    ],
    "latitude": [
        "latitude",
        "lat",
        "location_latitude",
    ],
    "longitude": [
        "longitude",
        "lon",
        "lng",
        "location_longitude",
    ],
    "timezone": [
        "timezone",
        "time_zone",
        "tz",
    ],
    "station": [
        "station",
        "station_id",
        "monitoring_station",
        "site",
        "site_id",
    ],
}


# ============================================================================
# CONSOLE
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def line(label: str, value: Any) -> None:
    print(f"{label:<32}: {value}")


# ============================================================================
# JSON HELPERS
# ============================================================================

def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        return None if not math.isfinite(value) else value

    return str(value)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ============================================================================
# HASHING / CACHE
# ============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_json(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=json_default,
    )
    return sha256_text(canonical)


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_hash.json"


def save_hash(name: str, digest: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "component": name,
        "sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    save_json(cache_path(name), payload)


def read_cached_hash(name: str) -> Optional[str]:
    path = cache_path(name)

    if not path.exists():
        return None

    payload = load_json(path)

    if not isinstance(payload, dict):
        return None

    return payload.get("sha256")


def cache_hit(name: str, digest: str) -> bool:
    return read_cached_hash(name) == digest


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def normalize_column_name(name: Any) -> str:
    value = str(name).strip().lower()

    value = value.replace("µ", "u")
    value = value.replace("μ", "u")

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def build_normalized_column_map(columns: List[str]) -> Dict[str, str]:
    result = {}

    for column in columns:
        result[normalize_column_name(column)] = column

    return result


def find_column(
    columns: List[str],
    aliases: List[str],
) -> Optional[str]:

    normalized = build_normalized_column_map(columns)

    normalized_aliases = [
        normalize_column_name(alias)
        for alias in aliases
    ]

    # Exact match first.
    for alias in normalized_aliases:
        if alias in normalized:
            return normalized[alias]

    # Conservative contains match.
    for alias in normalized_aliases:
        if not alias:
            continue

        for normalized_column, original_column in normalized.items():
            if (
                normalized_column.startswith(alias + "_")
                or normalized_column.endswith("_" + alias)
            ):
                return original_column

    return None


# ============================================================================
# TIMESTAMP DETECTION
# ============================================================================

def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:

    preferred = [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "utc_timestamp",
        "datetime_utc",
        "ds",
    ]

    return find_column(list(df.columns), preferred)


def parse_timestamp_series(
    df: pd.DataFrame,
) -> Tuple[pd.Series, Optional[str]]:

    timestamp_column = find_timestamp_column(df)

    if timestamp_column is None:
        raise ValueError(
            "Could not identify a timestamp column in pollution dataset."
        )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    if timestamps.notna().sum() == 0:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains no "
            "parseable timestamps."
        )

    return timestamps, timestamp_column


# ============================================================================
# LOCATION INTELLIGENCE
# ============================================================================

def infer_location_from_path(path: Path) -> Dict[str, Any]:

    stem = path.stem.lower()

    city = None
    country = None

    # Project-specific dataset identity.
    if "lahore" in stem:
        city = "Lahore"
        country = "Pakistan"

    return {
        "city": city,
        "country": country,
        "latitude": None,
        "longitude": None,
        "timezone": None,
        "station": None,
        "source": "dataset_filename",
        "confidence": "dataset-derived",
    }


def clean_scalar(value: Any) -> Optional[Any]:

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        return value

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    return value


def build_location_intelligence(
    df: pd.DataFrame,
    source_path: Path,
) -> Dict[str, Any]:

    fallback = infer_location_from_path(source_path)

    discovered = {
        "city": None,
        "country": None,
        "latitude": None,
        "longitude": None,
        "timezone": None,
        "station": None,
    }

    fields_found = []

    for field, aliases in LOCATION_ALIASES.items():

        column = find_column(list(df.columns), aliases)

        if column is None:
            continue

        fields_found.append(field)

        series = df[column].dropna()

        if series.empty:
            continue

        value = clean_scalar(series.iloc[0])

        if value is not None:
            discovered[field] = value

    # Dataset-derived fallback for Lahore.
    if discovered["city"] is None and fallback["city"] is not None:
        discovered["city"] = fallback["city"]

    if discovered["country"] is None and fallback["country"] is not None:
        discovered["country"] = fallback["country"]

    source = (
        "dataset_fields"
        if fields_found
        else fallback["source"]
    )

    return {
        "city": discovered["city"],
        "country": discovered["country"],
        "latitude": discovered["latitude"],
        "longitude": discovered["longitude"],
        "timezone": discovered["timezone"],
        "station": discovered["station"],
        "source": source,
        "confidence": "dataset-derived",
        "fields_discovered": fields_found,
        "coordinates_available": (
            discovered["latitude"] is not None
            and discovered["longitude"] is not None
        ),
    }


# ============================================================================
# POLLUTANT DETECTION
# ============================================================================

def discover_pollutant_columns(
    df: pd.DataFrame,
) -> Dict[str, str]:

    result = {}

    columns = list(df.columns)

    for pollutant, aliases in POLLUTANT_ALIASES.items():

        column = find_column(columns, aliases)

        if column is not None:
            result[pollutant] = column

    return result


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================================
# TREND ANALYSIS
# ============================================================================

def trend_label(
    series: pd.Series,
) -> str:

    clean = series.dropna()

    if len(clean) < 4:
        return "insufficient_data"

    recent = clean.tail(min(6, len(clean)))

    if len(clean) >= 12:
        previous = clean.iloc[-12:-6]
    else:
        midpoint = max(1, len(clean) // 2)
        previous = clean.iloc[:midpoint]

    if previous.empty:
        return "insufficient_data"

    recent_mean = float(recent.mean())
    previous_mean = float(previous.mean())

    if abs(previous_mean) < 1e-12:
        return "stable"

    change_pct = (
        (recent_mean - previous_mean)
        / abs(previous_mean)
        * 100.0
    )

    if change_pct >= 10:
        return "rising"

    if change_pct <= -10:
        return "falling"

    return "stable"


def trend_percent(
    series: pd.Series,
) -> Optional[float]:

    clean = series.dropna()

    if len(clean) < 4:
        return None

    recent = clean.tail(min(6, len(clean)))

    if len(clean) >= 12:
        previous = clean.iloc[-12:-6]
    else:
        midpoint = max(1, len(clean) // 2)
        previous = clean.iloc[:midpoint]

    if previous.empty:
        return None

    previous_mean = float(previous.mean())
    recent_mean = float(recent.mean())

    if abs(previous_mean) < 1e-12:
        return None

    return (
        (recent_mean - previous_mean)
        / abs(previous_mean)
        * 100.0
    )


# ============================================================================
# POLLUTANT STATISTICS
# ============================================================================

def safe_float(value: Any) -> Optional[float]:

    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    return value if math.isfinite(value) else None


def pollutant_statistics(
    df: pd.DataFrame,
    timestamp_column: str,
    pollutant: str,
    source_column: str,
) -> Dict[str, Any]:

    work = pd.DataFrame(
        {
            "timestamp": df[timestamp_column],
            "value": numeric_series(df, source_column),
        }
    ).dropna(subset=["timestamp"])

    work = work.sort_values("timestamp")

    values = work["value"].dropna()

    if values.empty:
        return {
            "pollutant": pollutant,
            "source_column": source_column,
            "observations": 0,
            "missing": int(len(work)),
            "missing_percentage": 100.0,
            "latest": None,
            "latest_timestamp": None,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "trend": "insufficient_data",
            "trend_percent": None,
            "peak_timestamp": None,
        }

    latest_idx = values.index[-1]

    # 24h and 72h windows use the latest available observation.
    latest_timestamp = work.loc[latest_idx, "timestamp"]

    last_24h = work[
        work["timestamp"]
        >= latest_timestamp - pd.Timedelta(hours=24)
    ]["value"].dropna()

    last_72h = work[
        work["timestamp"]
        >= latest_timestamp - pd.Timedelta(hours=72)
    ]["value"].dropna()

    peak_idx = values.idxmax()

    return {
        "pollutant": pollutant,
        "source_column": source_column,
        "observations": int(values.notna().sum()),
        "missing": int(work["value"].isna().sum()),
        "missing_percentage": round(
            float(work["value"].isna().mean() * 100),
            4,
        ),
        "latest": safe_float(values.iloc[-1]),
        "latest_timestamp": latest_timestamp,
        "min": safe_float(values.min()),
        "max": safe_float(values.max()),
        "mean": safe_float(values.mean()),
        "median": safe_float(values.median()),
        "std": safe_float(values.std()),
        "trend": trend_label(values),
        "trend_percent": safe_float(trend_percent(values)),
        "peak_timestamp": work.loc[peak_idx, "timestamp"],
        "last_24h": {
            "observations": int(last_24h.count()),
            "mean": safe_float(last_24h.mean()),
            "min": safe_float(last_24h.min()),
            "max": safe_float(last_24h.max()),
        },
        "last_72h": {
            "observations": int(last_72h.count()),
            "mean": safe_float(last_72h.mean()),
            "min": safe_float(last_72h.min()),
            "max": safe_float(last_72h.max()),
        },
    }


# ============================================================================
# POLLUTION HOURLY DATA
# ============================================================================

def build_pollution_hourly(
    df: pd.DataFrame,
    timestamp_column: str,
    pollutant_columns: Dict[str, str],
) -> pd.DataFrame:

    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                df[timestamp_column],
                errors="coerce",
                utc=True,
            )
        }
    )

    for pollutant, column in pollutant_columns.items():
        output[pollutant] = numeric_series(df, column)

    output = output.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp")

    output = (
        output
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return output


# ============================================================================
# DOMINANT POLLUTANT
# ============================================================================

def determine_dominant_pollutant(
    statistics: Dict[str, Dict[str, Any]],
) -> Optional[str]:

    candidates = []

    for pollutant, stats in statistics.items():

        value = stats.get("last_24h", {}).get("mean")

        if value is None:
            value = stats.get("latest")

        if value is None:
            continue

        candidates.append(
            (pollutant, float(value))
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates[0][0]


# ============================================================================
# POLLUTION EVENT DETECTION
# ============================================================================

def build_pollution_events(
    pollution_hourly: pd.DataFrame,
    pollutant_columns: Dict[str, str],
) -> pd.DataFrame:

    events = []

    for pollutant in pollutant_columns:

        if pollutant not in pollution_hourly.columns:
            continue

        values = pd.to_numeric(
            pollution_hourly[pollutant],
            errors="coerce",
        )

        clean = values.dropna()

        if len(clean) < 8:
            continue

        mean = float(clean.mean())
        std = float(clean.std())

        if not math.isfinite(std) or std <= 0:
            continue

        threshold = mean + std

        mask = values >= threshold

        for index in pollution_hourly.index[mask.fillna(False)]:

            value = safe_float(
                pollution_hourly.loc[index, pollutant]
            )

            timestamp = pollution_hourly.loc[
                index,
                "timestamp",
            ]

            if value is None:
                continue

            severity = (
                "high"
                if value >= mean + 2 * std
                else "elevated"
            )

            events.append(
                {
                    "timestamp": timestamp,
                    "pollutant": pollutant,
                    "value": value,
                    "baseline_mean": mean,
                    "baseline_std": std,
                    "threshold": threshold,
                    "severity": severity,
                    "event_type": "elevated_pollution",
                }
            )

    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "pollutant",
                "value",
                "baseline_mean",
                "baseline_std",
                "threshold",
                "severity",
                "event_type",
            ]
        )

    result = pd.DataFrame(events)

    return result.sort_values(
        ["timestamp", "pollutant"]
    ).reset_index(drop=True)


# ============================================================================
# AQI FORECAST INTELLIGENCE
# ============================================================================

def find_forecast_timestamp_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "timestamp",
        "forecast_timestamp",
        "datetime",
        "date_time",
        "valid_time",
        "time",
    ]

    return find_column(
        list(df.columns),
        candidates,
    )


def find_forecast_value_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "predicted_aqi",
        "prediction",
        "predicted_us_aqi",
        "forecast_aqi",
        "us_aqi_prediction",
        "aqi",
        "us_aqi",
    ]

    return find_column(
        list(df.columns),
        candidates,
    )


def load_forecast() -> Tuple[pd.DataFrame, str, str]:

    if not FORECAST_SOURCE.exists():
        raise FileNotFoundError(
            f"Production forecast not found:\n{FORECAST_SOURCE}"
        )

    df = pd.read_csv(FORECAST_SOURCE)

    if df.empty:
        raise ValueError(
            "Production forecast CSV is empty."
        )

    timestamp_column = find_forecast_timestamp_column(df)

    if timestamp_column is None:
        raise ValueError(
            "Could not identify forecast timestamp column."
        )

    value_column = find_forecast_value_column(df)

    if value_column is None:
        raise ValueError(
            "Could not identify forecast AQI value column."
        )

    df = df.copy()

    df["_timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    df["_aqi"] = pd.to_numeric(
        df[value_column],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["_timestamp", "_aqi"])
        .sort_values("_timestamp")
        .drop_duplicates("_timestamp")
        .reset_index(drop=True)
    )

    return df, "_timestamp", "_aqi"


def aqi_category_us(aqi: float) -> str:

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


def aqi_risk_level(aqi: float) -> str:

    if aqi <= 50:
        return "Low"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Elevated"

    if aqi <= 200:
        return "High"

    if aqi <= 300:
        return "Very High"

    return "Extreme"


def build_aqi_intelligence(
    forecast: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    values = forecast[value_column].astype(float)

    output = pd.DataFrame(
        {
            "timestamp": forecast[timestamp_column],
            "forecast_aqi": values,
        }
    )

    output["category"] = output["forecast_aqi"].apply(
        aqi_category_us
    )

    output["risk_level"] = output["forecast_aqi"].apply(
        aqi_risk_level
    )

    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    median = float(values.median())

    category_counts = (
        output["category"]
        .value_counts()
        .to_dict()
    )

    dominant_category = max(
        category_counts,
        key=category_counts.get,
    )

    peak_idx = values.idxmax()
    minimum_idx = values.idxmin()

    peak_row = forecast.loc[peak_idx]
    minimum_row = forecast.loc[minimum_idx]

    # Forecast trend: first six vs last six.
    if len(values) >= 12:
        first_mean = float(values.head(6).mean())
        last_mean = float(values.tail(6).mean())

        if first_mean != 0:
            change_pct = (
                (last_mean - first_mean)
                / abs(first_mean)
                * 100
            )
        else:
            change_pct = None
    else:
        change_pct = None

    if change_pct is None:
        forecast_trend = "insufficient_data"
    elif change_pct >= 10:
        forecast_trend = "rising"
    elif change_pct <= -10:
        forecast_trend = "falling"
    else:
        forecast_trend = "stable"

    summary = {
        "forecast_horizon_hours": int(len(output)),
        "forecast_start": output["timestamp"].min(),
        "forecast_end": output["timestamp"].max(),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "dominant_category": dominant_category,
        "category_distribution": category_counts,
        "peak": {
            "timestamp": peak_row[timestamp_column],
            "aqi": maximum,
            "category": aqi_category_us(maximum),
            "risk_level": aqi_risk_level(maximum),
        },
        "minimum_point": {
            "timestamp": minimum_row[timestamp_column],
            "aqi": minimum,
            "category": aqi_category_us(minimum),
        },
        "forecast_trend": forecast_trend,
        "forecast_change_percent": change_pct,
        "negative_values": int((values < 0).sum()),
        "non_finite_values": int(
            (~np.isfinite(values)).sum()
        ),
    }

    return output, summary


# ============================================================================
# COMBINED INTELLIGENCE
# ============================================================================

def build_pollution_summary_table(
    statistics: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    rows = []

    for pollutant, stats in statistics.items():

        last24 = stats.get("last_24h", {})
        last72 = stats.get("last_72h", {})

        rows.append(
            {
                "pollutant": pollutant,
                "source_column": stats.get("source_column"),
                "latest": stats.get("latest"),
                "latest_timestamp": stats.get("latest_timestamp"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "std": stats.get("std"),
                "trend": stats.get("trend"),
                "trend_percent": stats.get("trend_percent"),
                "last_24h_mean": last24.get("mean"),
                "last_24h_min": last24.get("min"),
                "last_24h_max": last24.get("max"),
                "last_72h_mean": last72.get("mean"),
                "last_72h_min": last72.get("min"),
                "last_72h_max": last72.get("max"),
                "observations": stats.get("observations"),
                "missing": stats.get("missing"),
                "missing_percentage": stats.get(
                    "missing_percentage"
                ),
                "peak_timestamp": stats.get(
                    "peak_timestamp"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_intelligence_payload(
    location: Dict[str, Any],
    pollutant_statistics_data: Dict[str, Dict[str, Any]],
    pollutant_columns: Dict[str, str],
    aqi_summary: Dict[str, Any],
    event_df: pd.DataFrame,
    pollution_hourly: pd.DataFrame,
) -> Dict[str, Any]:

    dominant_pollutant = determine_dominant_pollutant(
        pollutant_statistics_data
    )

    trend_counts = {}

    for pollutant, stats in pollutant_statistics_data.items():

        trend = stats.get("trend", "unknown")

        trend_counts[trend] = (
            trend_counts.get(trend, 0) + 1
        )

    if dominant_pollutant is not None:

        dominant_stats = pollutant_statistics_data[
            dominant_pollutant
        ]

    else:
        dominant_stats = {}

    if not pollution_hourly.empty:
        observed_start = pollution_hourly["timestamp"].min()
        observed_end = pollution_hourly["timestamp"].max()
    else:
        observed_start = None
        observed_end = None

    high_event_count = 0
    elevated_event_count = 0

    if not event_df.empty:

        high_event_count = int(
            (event_df["severity"] == "high").sum()
        )

        elevated_event_count = int(
            (event_df["severity"] == "elevated").sum()
        )

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "location": location,

        "data_coverage": {
            "pollution_source": str(POLLUTION_SOURCE),
            "observed_start": observed_start,
            "observed_end": observed_end,
            "pollution_rows": int(
                len(pollution_hourly)
            ),
        },

        "pollutants": {
            "available": list(
                pollutant_columns.keys()
            ),
            "count": int(
                len(pollutant_columns)
            ),
            "statistics": pollutant_statistics_data,
        },

        "pollution_intelligence": {
            "dominant_pollutant": dominant_pollutant,
            "dominant_pollutant_latest": (
                dominant_stats.get("latest")
                if dominant_stats
                else None
            ),
            "dominant_pollutant_24h_mean": (
                dominant_stats
                .get("last_24h", {})
                .get("mean")
                if dominant_stats
                else None
            ),
            "trend_counts": trend_counts,
            "pollution_events": {
                "total": int(len(event_df)),
                "elevated": elevated_event_count,
                "high": high_event_count,
            },
        },

        "aqi_intelligence": aqi_summary,

        "model_information": {
            "model_training_performed": False,
            "model_selection_performed": False,
            "hyperparameter_tuning_performed": False,
            "model_retraining_performed": False,
            "validation_used": False,
            "test_used": False,
            "future_target_leakage": "NONE",
            "source": "production_forecast",
        },
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_location(
    location: Dict[str, Any],
) -> Dict[str, Any]:

    checks = {}

    checks["location_object"] = isinstance(
        location,
        dict,
    )

    checks["city_identified"] = bool(
        location.get("city")
    )

    # Coordinates are optional. If supplied, validate them.
    if location.get("latitude") is not None:

        try:
            latitude = float(
                location["latitude"]
            )

            checks["latitude_valid"] = (
                -90 <= latitude <= 90
            )

        except Exception:
            checks["latitude_valid"] = False

    else:
        checks["latitude_valid"] = True

    if location.get("longitude") is not None:

        try:
            longitude = float(
                location["longitude"]
            )

            checks["longitude_valid"] = (
                -180 <= longitude <= 180
            )

        except Exception:
            checks["longitude_valid"] = False

    else:
        checks["longitude_valid"] = True

    checks["passed"] = all(
        checks.values()
    )

    return checks


def validate_pollution(
    pollution_hourly: pd.DataFrame,
    pollutant_columns: Dict[str, str],
) -> Dict[str, Any]:

    checks = {}

    checks["rows_available"] = (
        len(pollution_hourly) > 0
    )

    checks["pollutants_available"] = (
        len(pollutant_columns) > 0
    )

    checks["timestamp_valid"] = (
        pollution_hourly["timestamp"]
        .notna()
        .all()
        if not pollution_hourly.empty
        else False
    )

    checks["no_duplicate_timestamps"] = (
        pollution_hourly["timestamp"]
        .duplicated()
        .sum()
        == 0
        if not pollution_hourly.empty
        else False
    )

    negative_counts = {}

    for pollutant in pollutant_columns:

        if pollutant not in pollution_hourly.columns:
            continue

        values = pd.to_numeric(
            pollution_hourly[pollutant],
            errors="coerce",
        )

        negative_counts[pollutant] = int(
            (values < 0).sum()
        )

    checks["negative_pollutant_values"] = (
        sum(negative_counts.values()) == 0
    )

    checks["finite_pollutant_values"] = True

    for pollutant in pollutant_columns:

        values = pd.to_numeric(
            pollution_hourly[pollutant],
            errors="coerce",
        ).dropna()

        if not np.isfinite(values).all():
            checks["finite_pollutant_values"] = False
            break

    checks["passed"] = all(
        checks.values()
    )

    checks["negative_counts"] = negative_counts

    return checks


def validate_aqi(
    aqi_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> Dict[str, Any]:

    checks = {}

    checks["rows_available"] = (
        len(aqi_df) > 0
    )

    checks["expected_horizon"] = (
        len(aqi_df) == FORECAST_HORIZON
    )

    checks["timestamp_unique"] = (
        aqi_df["timestamp"]
        .duplicated()
        .sum()
        == 0
        if not aqi_df.empty
        else False
    )

    checks["aqi_finite"] = (
        np.isfinite(
            aqi_df["forecast_aqi"].astype(float)
        ).all()
        if not aqi_df.empty
        else False
    )

    checks["negative_aqi"] = (
        int(
            (
                aqi_df["forecast_aqi"]
                < 0
            ).sum()
        )
        == 0
        if not aqi_df.empty
        else False
    )

    checks["summary_valid"] = (
        summary.get("minimum") is not None
        and summary.get("maximum") is not None
        and summary.get("mean") is not None
    )

    checks["passed"] = all(
        checks.values()
    )

    return checks


def validate_package(
    paths: Dict[str, Path],
    package: Dict[str, Any],
) -> Dict[str, Any]:

    checks = {}

    for key, path in paths.items():

        checks[key] = path.exists() and path.stat().st_size > 0

    checks["package_structure"] = all(
        key in package
        for key in [
            "location",
            "pollutants",
            "pollution_intelligence",
            "aqi_intelligence",
            "model_information",
        ]
    )

    checks["passed"] = all(
        checks.values()
    )

    return checks


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    started = time.perf_counter()

    banner("PEARLS AQI PREDICTOR")
    banner("STEP 21 — LOCATION + POLLUTION INTELLIGENCE")

    line("Base directory", BASE_DIR)
    line("Target", TARGET)
    line("Forecast horizon", FORECAST_HORIZON)
    line("Dashboard input", "STEP 20")
    line("Caching", "Enabled")
    line("Model training", "NOT performed")
    line("Model selection", "NOT performed")
    line("Validation/test", "NOT performed")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ------------------------------------------------------------------
        # LOAD FORECAST
        # ------------------------------------------------------------------

        banner("LOADING PRODUCTION FORECAST")

        forecast, forecast_ts, forecast_value = (
            load_forecast()
        )

        line(
            "Forecast source",
            FORECAST_SOURCE,
        )

        line(
            "Forecast rows",
            len(forecast),
        )

        line(
            "Forecast start",
            forecast[forecast_ts].min(),
        )

        line(
            "Forecast end",
            forecast[forecast_ts].max(),
        )

        # ------------------------------------------------------------------
        # LOAD POLLUTION DATA
        # ------------------------------------------------------------------

        banner("LOADING POLLUTION INGREDIENT DATA")

        if not POLLUTION_SOURCE.exists():
            raise FileNotFoundError(
                f"Pollution source not found:\n"
                f"{POLLUTION_SOURCE}"
            )

        pollution_df = pd.read_csv(
            POLLUTION_SOURCE
        )

        if pollution_df.empty:
            raise ValueError(
                "Pollution dataset is empty."
            )

        line(
            "Pollution source",
            POLLUTION_SOURCE,
        )

        line(
            "Pollution rows",
            len(pollution_df),
        )

        line(
            "Pollution columns",
            len(pollution_df.columns),
        )

        timestamps, timestamp_column = (
            parse_timestamp_series(
                pollution_df
            )
        )

        pollution_df = pollution_df.copy()

        # Assign in one operation to avoid DataFrame fragmentation.
        pollution_df = pollution_df.assign(
            _dashboard_timestamp=timestamps
        )

        # ------------------------------------------------------------------
        # LOCATION
        # ------------------------------------------------------------------

        banner("BUILDING LOCATION INTELLIGENCE")

        location = build_location_intelligence(
            pollution_df,
            POLLUTION_SOURCE,
        )

        line(
            "City",
            location.get("city") or "NOT AVAILABLE",
        )

        line(
            "Country",
            location.get("country") or "NOT AVAILABLE",
        )

        line(
            "Latitude",
            location.get("latitude")
            if location.get("latitude") is not None
            else "NOT AVAILABLE",
        )

        line(
            "Longitude",
            location.get("longitude")
            if location.get("longitude") is not None
            else "NOT AVAILABLE",
        )

        line(
            "Timezone",
            location.get("timezone")
            or "NOT AVAILABLE",
        )

        line(
            "Station",
            location.get("station")
            or "NOT AVAILABLE",
        )

        line(
            "Location source",
            location.get("source"),
        )

        # ------------------------------------------------------------------
        # POLLUTANTS
        # ------------------------------------------------------------------

        banner("DISCOVERING POLLUTION INGREDIENTS")

        pollutant_columns = (
            discover_pollutant_columns(
                pollution_df
            )
        )

        line(
            "Pollutants discovered",
            len(pollutant_columns),
        )

        for pollutant in POLLUTANT_ALIASES:

            if pollutant in pollutant_columns:
                print(
                    f"{pollutant:<32}: FOUND -> "
                    f"{pollutant_columns[pollutant]}"
                )
            else:
                print(
                    f"{pollutant:<32}: NOT AVAILABLE"
                )

        if not pollutant_columns:
            raise ValueError(
                "No supported pollution ingredient fields found."
            )

        # ------------------------------------------------------------------
        # BUILD HOURLY POLLUTION TABLE
        # ------------------------------------------------------------------

        banner("BUILDING POLLUTION HOURLY DATA")

        pollution_hourly = build_pollution_hourly(
            pollution_df,
            timestamp_column,
            pollutant_columns,
        )

        line(
            "Hourly rows",
            len(pollution_hourly),
        )

        line(
            "Hourly start",
            pollution_hourly["timestamp"].min(),
        )

        line(
            "Hourly end",
            pollution_hourly["timestamp"].max(),
        )

        # ------------------------------------------------------------------
        # CACHE POLLUTION INPUT
        # ------------------------------------------------------------------

        pollution_input_hash = sha256_file(
            POLLUTION_SOURCE
        )

        forecast_input_hash = sha256_file(
            FORECAST_SOURCE
        )

        location_hash = hash_json(
            location
        )

        pollution_component_hash = hash_json(
            {
                "source_hash": pollution_input_hash,
                "pollutants": pollutant_columns,
                "timestamp_column": timestamp_column,
            }
        )

        intelligence_component_hash = hash_json(
            {
                "pollution_hash": pollution_component_hash,
                "forecast_hash": forecast_input_hash,
                "location_hash": location_hash,
            }
        )

        package_component_hash = hash_json(
            {
                "intelligence_hash": intelligence_component_hash,
                "schema_version": "1.0",
            }
        )

        banner("CHECKING INTELLIGENCE CACHE")

        location_hit = cache_hit(
            "location",
            location_hash,
        )

        pollution_hit = cache_hit(
            "pollution",
            pollution_component_hash,
        )

        intelligence_hit = cache_hit(
            "intelligence",
            intelligence_component_hash,
        )

        package_hit = cache_hit(
            "package",
            package_component_hash,
        )

        line(
            "Location cache",
            "HIT" if location_hit else "MISS",
        )

        line(
            "Pollution cache",
            "HIT" if pollution_hit else "MISS",
        )

        line(
            "Intelligence cache",
            "HIT" if intelligence_hit else "MISS",
        )

        line(
            "Package cache",
            "HIT" if package_hit else "MISS",
        )

        overall_cache = all(
            [
                location_hit,
                pollution_hit,
                intelligence_hit,
                package_hit,
            ]
        )

        line(
            "Overall cache",
            "HIT" if overall_cache else "MISS",
        )

        # ------------------------------------------------------------------
        # POLLUTION STATISTICS
        # ------------------------------------------------------------------

        banner("CALCULATING POLLUTION INTELLIGENCE")

        pollutant_statistics_data = {}

        for pollutant, column in pollutant_columns.items():

            stats = pollutant_statistics(
                pollution_df,
                "_dashboard_timestamp",
                pollutant,
                column,
            )

            pollutant_statistics_data[
                pollutant
            ] = stats

        dominant_pollutant = (
            determine_dominant_pollutant(
                pollutant_statistics_data
            )
        )

        line(
            "Dominant pollutant",
            dominant_pollutant
            or "NOT AVAILABLE",
        )

        # ------------------------------------------------------------------
        # EVENTS
        # ------------------------------------------------------------------

        banner("DETECTING POLLUTION EVENTS")

        event_df = build_pollution_events(
            pollution_hourly,
            pollutant_columns,
        )

        line(
            "Pollution events",
            len(event_df),
        )

        if not event_df.empty:
            line(
                "High severity events",
                int(
                    (
                        event_df["severity"]
                        == "high"
                    ).sum()
                ),
            )

            line(
                "Elevated events",
                int(
                    (
                        event_df["severity"]
                        == "elevated"
                    ).sum()
                ),
            )
        else:
            line(
                "High severity events",
                0,
            )

            line(
                "Elevated events",
                0,
            )

        # ------------------------------------------------------------------
        # AQI
        # ------------------------------------------------------------------

        banner("BUILDING AQI FORECAST INTELLIGENCE")

        aqi_df, aqi_summary = (
            build_aqi_intelligence(
                forecast,
                forecast_ts,
                forecast_value,
            )
        )

        line(
            "Forecast minimum",
            round(
                aqi_summary["minimum"],
                3,
            ),
        )

        line(
            "Forecast maximum",
            round(
                aqi_summary["maximum"],
                3,
            ),
        )

        line(
            "Forecast mean",
            round(
                aqi_summary["mean"],
                3,
            ),
        )

        line(
            "Dominant AQI category",
            aqi_summary[
                "dominant_category"
            ],
        )

        line(
            "Forecast trend",
            aqi_summary[
                "forecast_trend"
            ],
        )

        # ------------------------------------------------------------------
        # PACKAGE
        # ------------------------------------------------------------------

        banner("BUILDING INTELLIGENCE PACKAGE")

        package = build_intelligence_payload(
            location=location,
            pollutant_statistics_data=(
                pollutant_statistics_data
            ),
            pollutant_columns=pollutant_columns,
            aqi_summary=aqi_summary,
            event_df=event_df,
            pollution_hourly=pollution_hourly,
        )

        package["cache"] = {
            "enabled": True,
            "location": {
                "hash": location_hash,
                "hit": location_hit,
            },
            "pollution": {
                "hash": pollution_component_hash,
                "hit": pollution_hit,
            },
            "intelligence": {
                "hash": intelligence_component_hash,
                "hit": intelligence_hit,
            },
            "package": {
                "hash": package_component_hash,
                "hit": package_hit,
            },
        }

        # ------------------------------------------------------------------
        # SAVE OUTPUTS
        # ------------------------------------------------------------------

        banner("SAVING LOCATION + POLLUTION INTELLIGENCE")

        location_output = (
            OUTPUT_DIR
            / "location_intelligence.json"
        )

        pollution_summary_output = (
            OUTPUT_DIR
            / "pollution_intelligence.csv"
        )

        pollution_hourly_output = (
            OUTPUT_DIR
            / "pollution_hourly.csv"
        )

        aqi_output = (
            OUTPUT_DIR
            / "aqi_intelligence.csv"
        )

        events_output = (
            OUTPUT_DIR
            / "pollution_events.csv"
        )

        package_output = (
            OUTPUT_DIR
            / "location_pollution_package.json"
        )

        report_output = (
            OUTPUT_DIR
            / "intelligence_report.json"
        )

        save_json(
            location_output,
            location,
        )

        pollution_summary_df = (
            build_pollution_summary_table(
                pollutant_statistics_data
            )
        )

        pollution_summary_df.to_csv(
            pollution_summary_output,
            index=False,
        )

        pollution_hourly.to_csv(
            pollution_hourly_output,
            index=False,
        )

        aqi_df.to_csv(
            aqi_output,
            index=False,
        )

        event_df.to_csv(
            events_output,
            index=False,
        )

        save_json(
            package_output,
            package,
        )

        # ------------------------------------------------------------------
        # UPDATE CACHE
        # ------------------------------------------------------------------

        save_hash(
            "location",
            location_hash,
            {
                "city": location.get("city"),
                "country": location.get("country"),
            },
        )

        save_hash(
            "pollution",
            pollution_component_hash,
            {
                "source": str(
                    POLLUTION_SOURCE
                ),
                "pollutants": pollutant_columns,
            },
        )

        save_hash(
            "intelligence",
            intelligence_component_hash,
            {
                "forecast_source": str(
                    FORECAST_SOURCE
                ),
            },
        )

        save_hash(
            "package",
            package_component_hash,
            {
                "schema_version": "1.0",
            },
        )

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------

        banner("VALIDATING LOCATION + POLLUTION INTELLIGENCE")

        location_validation = (
            validate_location(location)
        )

        pollution_validation = (
            validate_pollution(
                pollution_hourly,
                pollutant_columns,
            )
        )

        aqi_validation = (
            validate_aqi(
                aqi_df,
                aqi_summary,
            )
        )

        output_paths = {
            "location_intelligence": location_output,
            "pollution_intelligence": (
                pollution_summary_output
            ),
            "pollution_hourly": (
                pollution_hourly_output
            ),
            "aqi_intelligence": aqi_output,
            "pollution_events": events_output,
            "location_pollution_package": (
                package_output
            ),
        }

        package_validation = validate_package(
            output_paths,
            package,
        )

        line(
            "Location validation",
            "PASS"
            if location_validation["passed"]
            else "FAIL",
        )

        line(
            "Pollution validation",
            "PASS"
            if pollution_validation["passed"]
            else "FAIL",
        )

        line(
            "AQI validation",
            "PASS"
            if aqi_validation["passed"]
            else "FAIL",
        )

        line(
            "Package validation",
            "PASS"
            if package_validation["passed"]
            else "FAIL",
        )

        all_validation_passed = all(
            [
                location_validation["passed"],
                pollution_validation["passed"],
                aqi_validation["passed"],
                package_validation["passed"],
            ]
        )

        line(
            "Overall validation",
            "PASS"
            if all_validation_passed
            else "FAIL",
        )

        # ------------------------------------------------------------------
        # REPORT
        # ------------------------------------------------------------------

        elapsed = (
            time.perf_counter()
            - started
        )

        report = {
            "step": 21,
            "name": (
                "Location + Pollution Intelligence"
            ),
            "status": (
                "PASS"
                if all_validation_passed
                else "FAIL"
            ),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "base_directory": str(BASE_DIR),
            "target": TARGET,
            "forecast_horizon": FORECAST_HORIZON,
            "sources": {
                "forecast": str(
                    FORECAST_SOURCE
                ),
                "pollution": str(
                    POLLUTION_SOURCE
                ),
            },
            "location": location,
            "pollutants": {
                "count": len(
                    pollutant_columns
                ),
                "fields": pollutant_columns,
            },
            "aqi": aqi_summary,
            "intelligence": {
                "dominant_pollutant": (
                    dominant_pollutant
                ),
                "pollution_event_count": int(
                    len(event_df)
                ),
            },
            "cache": {
                "overall": (
                    "HIT"
                    if overall_cache
                    else "MISS"
                ),
                "location": location_hit,
                "pollution": pollution_hit,
                "intelligence": intelligence_hit,
                "package": package_hit,
            },
            "validation": {
                "location": location_validation,
                "pollution": pollution_validation,
                "aqi": aqi_validation,
                "package": package_validation,
                "overall": all_validation_passed,
            },
            "outputs": {
                key: str(path)
                for key, path in output_paths.items()
            },
            "execution_time_seconds": elapsed,
        }

        save_json(
            report_output,
            report,
        )

        # ------------------------------------------------------------------
        # COMPLETE
        # ------------------------------------------------------------------

        banner("STEP 21 COMPLETE")

        line(
            "Status",
            "PASS"
            if all_validation_passed
            else "FAIL",
        )

        line(
            "Location",
            (
                f"{location.get('city')}, "
                f"{location.get('country')}"
                if location.get("city")
                else "NOT AVAILABLE"
            ),
        )

        line(
            "Pollution ingredients",
            f"{len(pollutant_columns)} available",
        )

        line(
            "Dominant pollutant",
            dominant_pollutant
            or "NOT AVAILABLE",
        )

        line(
            "Forecast horizon",
            len(aqi_df),
        )

        line(
            "Forecast AQI mean",
            round(
                aqi_summary["mean"],
                3,
            ),
        )

        line(
            "Forecast AQI maximum",
            round(
                aqi_summary["maximum"],
                3,
            ),
        )

        line(
            "Dominant AQI category",
            aqi_summary[
                "dominant_category"
            ],
        )

        line(
            "Pollution events",
            len(event_df),
        )

        line(
            "Location metadata",
            (
                "AVAILABLE"
                if location.get(
                    "coordinates_available"
                )
                else "PARTIAL / NO COORDINATES"
            ),
        )

        line(
            "Cache status",
            (
                "HIT"
                if overall_cache
                else "MISS"
            ),
        )

        line(
            "Validation",
            "PASS"
            if all_validation_passed
            else "FAIL",
        )

        print()
        print("Location intelligence:")
        print(location_output)

        print()
        print("Pollution intelligence:")
        print(pollution_summary_output)

        print()
        print("Pollution hourly:")
        print(pollution_hourly_output)

        print()
        print("AQI intelligence:")
        print(aqi_output)

        print()
        print("Pollution events:")
        print(events_output)

        print()
        print("Intelligence package:")
        print(package_output)

        print()
        print("Step 21 report:")
        print(report_output)

        line(
            "Execution time",
            f"{elapsed:.3f}s",
        )

        return 0 if all_validation_passed else 1

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - started
        )

        banner("STEP 21 FAILED")

        print(str(exc))

        failure_report = {
            "step": 21,
            "name": (
                "Location + Pollution Intelligence"
            ),
            "status": "FAIL",
            "error": str(exc),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "execution_time_seconds": elapsed,
        }

        failure_path = (
            OUTPUT_DIR
            / "intelligence_report.json"
        )

        save_json(
            failure_path,
            failure_report,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())