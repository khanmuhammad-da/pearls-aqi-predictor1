"""
PEARLS AQI PREDICTOR
STEP 24 — AQI RISK + ALERT INTELLIGENCE

Purpose
-------
Convert the production AQI forecast and pollution intelligence into
operational risk / alert intelligence.

Inputs
------
STEP 20:
    reports/production_dashboard_v2/production_dashboard_forecast.csv

STEP 21:
    reports/location_pollution_intelligence/location_intelligence.json
    reports/location_pollution_intelligence/pollution_intelligence.csv
    reports/location_pollution_intelligence/pollution_hourly.csv
    reports/location_pollution_intelligence/aqi_intelligence.csv
    reports/location_pollution_intelligence/pollution_events.csv

Outputs
-------
reports/aqi_risk_alert_intelligence/
    risk_intelligence.json
    risk_hourly.csv
    risk_windows.csv
    pollutant_risk.csv
    category_exposure.csv
    alert_intelligence.json
    aqi_risk_package.json
    aqi_risk_alert_results.json
    cache/
        risk_hash.json

Design
------
- No model training
- No model selection
- No validation/test leakage
- Uses forecast output only
- Deterministic calculations
- Cache aware
- JSON serialization safe
- Robust to missing optional columns
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

STEP_NAME = "STEP 24 — AQI RISK + ALERT INTELLIGENCE"

TARGET = "us_aqi"
FORECAST_HORIZON = 72

BASE_DIR = Path(__file__).resolve().parents[1]

STEP20_DIR = BASE_DIR / "reports" / "production_dashboard_v2"
STEP21_DIR = BASE_DIR / "reports" / "location_pollution_intelligence"

OUTPUT_DIR = BASE_DIR / "reports" / "aqi_risk_alert_intelligence"
CACHE_DIR = OUTPUT_DIR / "cache"

FORECAST_FILE = STEP20_DIR / "production_dashboard_forecast.csv"

LOCATION_FILE = STEP21_DIR / "location_intelligence.json"
POLLUTION_FILE = STEP21_DIR / "pollution_intelligence.csv"
POLLUTION_HOURLY_FILE = STEP21_DIR / "pollution_hourly.csv"
AQI_INTELLIGENCE_FILE = STEP21_DIR / "aqi_intelligence.csv"
EVENTS_FILE = STEP21_DIR / "pollution_events.csv"

RISK_JSON = OUTPUT_DIR / "risk_intelligence.json"
RISK_HOURLY_CSV = OUTPUT_DIR / "risk_hourly.csv"
RISK_WINDOWS_CSV = OUTPUT_DIR / "risk_windows.csv"
POLLUTANT_RISK_CSV = OUTPUT_DIR / "pollutant_risk.csv"
CATEGORY_EXPOSURE_CSV = OUTPUT_DIR / "category_exposure.csv"
ALERT_JSON = OUTPUT_DIR / "alert_intelligence.json"
PACKAGE_JSON = OUTPUT_DIR / "aqi_risk_package.json"
RESULTS_JSON = OUTPUT_DIR / "aqi_risk_alert_results.json"

CACHE_HASH_FILE = CACHE_DIR / "risk_hash.json"


# ============================================================================
# CONSOLE
# ============================================================================

WIDTH = 72


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def info(label: str, value: Any) -> None:
    print(f"{label:<34}: {value}")


def fail(message: str) -> None:
    banner("STEP 24 FAILED")
    print(message)
    sys.exit(1)


# ============================================================================
# JSON UTILITIES
# ============================================================================

def json_safe(value: Any) -> Any:
    """
    Recursively convert numpy/pandas/path/timestamp objects into
    JSON-compatible Python objects.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.Series):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [
            {
                str(k): json_safe(v)
                for k, v in row.items()
            }
            for row in value.to_dict(orient="records")
        ]

    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(data),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# FILE / CACHE
# ============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def build_input_hash() -> str:
    """
    Hash the relevant input files.

    The cache therefore invalidates automatically whenever an upstream
    forecast or intelligence file changes.
    """

    files = [
        FORECAST_FILE,
        LOCATION_FILE,
        POLLUTION_FILE,
        POLLUTION_HOURLY_FILE,
        AQI_INTELLIGENCE_FILE,
        EVENTS_FILE,
    ]

    h = hashlib.sha256()

    for path in files:
        h.update(str(path).encode("utf-8"))

        if path.exists():
            h.update(sha256_file(path).encode("utf-8"))
        else:
            h.update(b"MISSING")

    return h.hexdigest()


def check_cache(input_hash: str) -> bool:
    if not CACHE_HASH_FILE.exists():
        return False

    data = read_json(CACHE_HASH_FILE, {})

    return (
        data.get("input_hash") == input_hash
        and PACKAGE_JSON.exists()
        and RISK_JSON.exists()
        and RISK_HOURLY_CSV.exists()
        and RISK_WINDOWS_CSV.exists()
        and POLLUTANT_RISK_CSV.exists()
        and CATEGORY_EXPOSURE_CSV.exists()
        and ALERT_JSON.exists()
    )


def save_cache_hash(input_hash: str) -> None:
    write_json(
        CACHE_HASH_FILE,
        {
            "input_hash": input_hash,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "step": 24,
        },
    )


# ============================================================================
# AQI CLASSIFICATION
# ============================================================================

AQI_BANDS = [
    {
        "name": "Good",
        "min": 0,
        "max": 50,
        "risk": "LOW",
        "severity": 0,
    },
    {
        "name": "Moderate",
        "min": 51,
        "max": 100,
        "risk": "LOW",
        "severity": 1,
    },
    {
        "name": "Unhealthy for Sensitive Groups",
        "min": 101,
        "max": 150,
        "risk": "MODERATE",
        "severity": 2,
    },
    {
        "name": "Unhealthy",
        "min": 151,
        "max": 200,
        "risk": "HIGH",
        "severity": 3,
    },
    {
        "name": "Very Unhealthy",
        "min": 201,
        "max": 300,
        "risk": "VERY HIGH",
        "severity": 4,
    },
    {
        "name": "Hazardous",
        "min": 301,
        "max": 500,
        "risk": "EXTREME",
        "severity": 5,
    },
]


def aqi_category(aqi: float) -> str:
    if not math.isfinite(float(aqi)):
        return "Unknown"

    value = float(aqi)

    for band in AQI_BANDS:
        if band["min"] <= value <= band["max"]:
            return band["name"]

    if value < 0:
        return "Invalid"

    return "Hazardous"


def aqi_risk_level(aqi: float) -> str:
    category = aqi_category(aqi)

    for band in AQI_BANDS:
        if band["name"] == category:
            return band["risk"]

    return "UNKNOWN"


def aqi_severity(aqi: float) -> int:
    category = aqi_category(aqi)

    for band in AQI_BANDS:
        if band["name"] == category:
            return band["severity"]

    return -1


# ============================================================================
# COLUMN DISCOVERY
# ============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(c).lower()): c
        for c in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())

        if key in normalized:
            return normalized[key]

    for col in df.columns:
        norm = re.sub(r"[^a-z0-9]+", "", str(col).lower())

        for candidate in candidates:
            key = re.sub(r"[^a-z0-9]+", "", candidate.lower())

            if key in norm or norm in key:
                return col

    return None


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "timestamp",
            "datetime",
            "date_time",
            "time",
            "date",
            "forecast_timestamp",
            "prediction_timestamp",
        ],
    )


def find_prediction_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "predicted_aqi",
            "prediction",
            "prediction_us_aqi",
            "forecast_aqi",
            "aqi",
            "us_aqi",
            "y_pred",
            "predicted",
        ],
    )


# ============================================================================
# LOCATION
# ============================================================================

def build_location(location_data: Dict[str, Any]) -> Dict[str, Any]:

    city = location_data.get("city") or "Lahore"
    country = location_data.get("country") or "Pakistan"

    latitude = location_data.get("latitude")
    longitude = location_data.get("longitude")

    if latitude is None:
        latitude = 31.467
    if longitude is None:
        longitude = 74.409

    timezone = (
        location_data.get("timezone")
        or location_data.get("time_zone")
        or "Asia/Karachi"
    )

    station = (
        location_data.get("station")
        or location_data.get("station_name")
        or "Lahore Cantonment"
    )

    return {
        "city": city,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "station": station,
        "source": (
            "Step 21 location intelligence with Lahore Cantonment "
            "fallback metadata"
        ),
    }


# ============================================================================
# FORECAST
# ============================================================================

def load_forecast() -> pd.DataFrame:

    if not FORECAST_FILE.exists():
        fail(f"Production forecast not found:\n{FORECAST_FILE}")

    df = pd.read_csv(FORECAST_FILE)

    if df.empty:
        fail("Production forecast is empty.")

    timestamp_col = find_timestamp_column(df)
    prediction_col = find_prediction_column(df)

    if timestamp_col is None:
        fail(
            "Forecast timestamp column could not be discovered.\n"
            f"Available columns: {list(df.columns)}"
        )

    if prediction_col is None:
        fail(
            "Forecast AQI prediction column could not be discovered.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["_timestamp"] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True,
    )

    df["_predicted_aqi"] = pd.to_numeric(
        df[prediction_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=["_timestamp", "_predicted_aqi"]
    ).sort_values("_timestamp")

    df = df.reset_index(drop=True)

    return df


# ============================================================================
# TREND
# ============================================================================

def calculate_trend(values: pd.Series) -> Tuple[str, float]:

    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 4:
        return "stable", 0.0

    n = len(values)

    first_n = max(3, n // 4)
    last_n = max(3, n // 4)

    first_mean = float(values.iloc[:first_n].mean())
    last_mean = float(values.iloc[-last_n:].mean())

    delta = last_mean - first_mean

    if delta > 5:
        return "rising", delta

    if delta < -5:
        return "falling", delta

    return "stable", delta


def calculate_hourly_change(df: pd.DataFrame) -> pd.Series:
    return df["_predicted_aqi"].diff()


# ============================================================================
# RISK SCORE
# ============================================================================

def calculate_risk_score(
    aqi: float,
    change: float,
    pollutant_pressure: float,
) -> float:

    # AQI component.
    aqi_component = min(max(float(aqi) / 200.0 * 70.0, 0.0), 70.0)

    # Rising AQI contributes additional risk.
    trend_component = 0.0

    if math.isfinite(change):
        trend_component = min(max(change * 4.0, -10.0), 15.0)

    # Pollution pressure is expected to be percentile-like 0–100.
    pollutant_component = min(
        max(float(pollutant_pressure) * 0.15, 0.0),
        15.0,
    )

    score = aqi_component + trend_component + pollutant_component

    return round(min(max(score, 0.0), 100.0), 2)


def risk_score_label(score: float) -> str:

    if score >= 80:
        return "EXTREME"

    if score >= 65:
        return "HIGH"

    if score >= 45:
        return "MODERATE"

    if score >= 25:
        return "LOW"

    return "MINIMAL"


# ============================================================================
# POLLUTION INTELLIGENCE
# ============================================================================

POLLUTANT_ALIASES = {
    "pm2.5": [
        "pm2_5",
        "pm25",
        "pm2.5",
        "pm_2_5",
    ],
    "pm10": [
        "pm10",
        "pm_10",
    ],
    "o3": [
        "o3",
        "ozone",
    ],
    "no2": [
        "no2",
        "nitrogen_dioxide",
        "nitrogendioxide",
    ],
    "so2": [
        "so2",
        "sulphur_dioxide",
        "sulfur_dioxide",
    ],
    "co": [
        "co",
        "carbon_monoxide",
        "carbonmonoxide",
    ],
}


def discover_pollutants(df: pd.DataFrame) -> Dict[str, str]:

    mapping = {}

    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(c).lower()): c
        for c in df.columns
    }

    for display_name, aliases in POLLUTANT_ALIASES.items():

        found = None

        for alias in aliases:
            key = re.sub(r"[^a-z0-9]+", "", alias.lower())

            if key in normalized:
                found = normalized[key]
                break

        if found is not None:
            mapping[display_name] = found

    return mapping


def load_pollution_data() -> Tuple[pd.DataFrame, Dict[str, str]]:

    if not POLLUTION_HOURLY_FILE.exists():
        return pd.DataFrame(), {}

    df = pd.read_csv(POLLUTION_HOURLY_FILE)

    if df.empty:
        return df, {}

    timestamp_col = find_timestamp_column(df)

    if timestamp_col:
        df["_timestamp"] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            utc=True,
        )
    else:
        df["_timestamp"] = pd.NaT

    mapping = discover_pollutants(df)

    return df, mapping


def load_pollution_summary() -> pd.DataFrame:

    if not POLLUTION_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(POLLUTION_FILE)


def calculate_pollutant_risk(
    pollution_df: pd.DataFrame,
    pollutant_mapping: Dict[str, str],
) -> pd.DataFrame:

    records = []

    for pollutant, column in pollutant_mapping.items():

        values = pd.to_numeric(
            pollution_df[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        current = float(values.iloc[-1])
        mean_value = float(values.mean())
        max_value = float(values.max())

        percentile = float(
            (values <= current).mean() * 100.0
        )

        q75 = float(values.quantile(0.75))
        q90 = float(values.quantile(0.90))

        if percentile >= 90:
            level = "EXTREME"
        elif percentile >= 75:
            level = "HIGH"
        elif percentile >= 50:
            level = "MODERATE"
        else:
            level = "LOW"

        if current > q90:
            direction = "ELEVATED"
        elif current > q75:
            direction = "ABOVE NORMAL"
        else:
            direction = "NORMAL"

        records.append(
            {
                "pollutant": pollutant.upper(),
                "source_column": column,
                "current_value": current,
                "mean_value": mean_value,
                "maximum_value": max_value,
                "percentile": percentile,
                "p75": q75,
                "p90": q90,
                "risk_level": level,
                "status": direction,
            }
        )

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            ["percentile", "current_value"],
            ascending=False,
        ).reset_index(drop=True)

        result["rank"] = np.arange(1, len(result) + 1)

    return result


def dominant_pollutant(
    pollutant_risk: pd.DataFrame,
) -> Tuple[str, float]:

    if pollutant_risk.empty:
        return "N/A", 0.0

    row = pollutant_risk.iloc[0]

    return (
        str(row["pollutant"]),
        float(row["percentile"]),
    )


# ============================================================================
# CATEGORY EXPOSURE
# ============================================================================

def build_category_exposure(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:

    total = len(forecast_df)

    records = []

    for band in AQI_BANDS:

        mask = (
            (forecast_df["_predicted_aqi"] >= band["min"])
            & (forecast_df["_predicted_aqi"] <= band["max"])
        )

        hours = int(mask.sum())

        percentage = (
            hours / total * 100.0
            if total
            else 0.0
        )

        records.append(
            {
                "category": band["name"],
                "min_aqi": band["min"],
                "max_aqi": band["max"],
                "hours": hours,
                "percentage": round(percentage, 2),
                "risk_level": band["risk"],
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# RISK HOURLY
# ============================================================================

def build_risk_hourly(
    forecast_df: pd.DataFrame,
    pollutant_pressure: float,
) -> pd.DataFrame:

    df = forecast_df[
        ["_timestamp", "_predicted_aqi"]
    ].copy()

    df["hour_index"] = np.arange(1, len(df) + 1)

    df["aqi_change"] = df["_predicted_aqi"].diff()

    df["category"] = df["_predicted_aqi"].apply(aqi_category)

    df["aqi_risk"] = df["_predicted_aqi"].apply(
        aqi_risk_level
    )

    df["severity"] = df["_predicted_aqi"].apply(
        aqi_severity
    )

    df["risk_score"] = [
        calculate_risk_score(
            aqi,
            change if math.isfinite(change) else 0.0,
            pollutant_pressure,
        )
        for aqi, change in zip(
            df["_predicted_aqi"],
            df["aqi_change"].fillna(0.0),
        )
    ]

    df["risk_label"] = df["risk_score"].apply(
        risk_score_label
    )

    df["direction"] = np.select(
        [
            df["aqi_change"] > 3,
            df["aqi_change"] < -3,
        ],
        [
            "rising",
            "falling",
        ],
        default="stable",
    )

    df["alert"] = np.select(
        [
            df["_predicted_aqi"] >= 301,
            df["_predicted_aqi"] >= 201,
            df["_predicted_aqi"] >= 151,
            df["_predicted_aqi"] >= 101,
        ],
        [
            "EXTREME",
            "VERY HIGH",
            "HIGH",
            "MODERATE",
        ],
        default="LOW",
    )

    df = df.rename(
        columns={
            "_timestamp": "timestamp",
            "_predicted_aqi": "predicted_aqi",
        }
    )

    return df


# ============================================================================
# RISK WINDOWS
# ============================================================================

def build_risk_windows(
    risk_hourly: pd.DataFrame,
) -> pd.DataFrame:

    if risk_hourly.empty:
        return pd.DataFrame()

    rows = []

    current_start = 0
    current_category = risk_hourly.iloc[0]["category"]

    for i in range(1, len(risk_hourly)):

        category = risk_hourly.iloc[i]["category"]

        if category != current_category:

            chunk = risk_hourly.iloc[
                current_start:i
            ]

            rows.append(
                make_window_record(
                    chunk,
                    current_category,
                )
            )

            current_start = i
            current_category = category

    chunk = risk_hourly.iloc[current_start:]

    rows.append(
        make_window_record(
            chunk,
            current_category,
        )
    )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["window_rank"] = (
            result["max_aqi"]
            .rank(
                ascending=False,
                method="dense",
            )
            .astype(int)
        )

    return result


def make_window_record(
    chunk: pd.DataFrame,
    category: str,
) -> Dict[str, Any]:

    max_row = chunk.loc[
        chunk["predicted_aqi"].idxmax()
    ]

    return {
        "start": chunk["timestamp"].iloc[0],
        "end": chunk["timestamp"].iloc[-1],
        "hours": int(len(chunk)),
        "category": category,
        "risk_level": aqi_risk_level(
            float(max_row["predicted_aqi"])
        ),
        "mean_aqi": float(
            chunk["predicted_aqi"].mean()
        ),
        "max_aqi": float(
            chunk["predicted_aqi"].max()
        ),
        "min_aqi": float(
            chunk["predicted_aqi"].min()
        ),
        "peak_timestamp": max_row["timestamp"],
    }


# ============================================================================
# ALERT INTELLIGENCE
# ============================================================================

def build_alert_intelligence(
    forecast_df: pd.DataFrame,
    risk_hourly: pd.DataFrame,
    risk_windows: pd.DataFrame,
    pollutant_risk: pd.DataFrame,
    location: Dict[str, Any],
) -> Dict[str, Any]:

    first_aqi = float(
        forecast_df["_predicted_aqi"].iloc[0]
    )

    max_aqi = float(
        forecast_df["_predicted_aqi"].max()
    )

    mean_aqi = float(
        forecast_df["_predicted_aqi"].mean()
    )

    median_aqi = float(
        forecast_df["_predicted_aqi"].median()
    )

    peak_idx = forecast_df["_predicted_aqi"].idxmax()

    peak_timestamp = forecast_df.loc[
        peak_idx,
        "_timestamp",
    ]

    trend, trend_delta = calculate_trend(
        forecast_df["_predicted_aqi"]
    )

    dominant, dominant_percentile = dominant_pollutant(
        pollutant_risk
    )

    unhealthy_hours = int(
        (
            forecast_df["_predicted_aqi"] >= 151
        ).sum()
    )

    very_unhealthy_hours = int(
        (
            forecast_df["_predicted_aqi"] >= 201
        ).sum()
    )

    hazardous_hours = int(
        (
            forecast_df["_predicted_aqi"] >= 301
        ).sum()
    )

    max_consecutive_unhealthy = (
        calculate_consecutive_hours(
            forecast_df["_predicted_aqi"] >= 151
        )
    )

    max_consecutive_very_unhealthy = (
        calculate_consecutive_hours(
            forecast_df["_predicted_aqi"] >= 201
        )
    )

    alert_level = determine_alert_level(
        max_aqi=max_aqi,
        unhealthy_hours=unhealthy_hours,
        very_unhealthy_hours=very_unhealthy_hours,
        hazardous_hours=hazardous_hours,
        trend=trend,
        dominant_percentile=dominant_percentile,
    )

    action = recommended_action(alert_level)

    watch_window = None

    if not risk_windows.empty:

        candidate = risk_windows.loc[
            risk_windows["max_aqi"].idxmax()
        ]

        watch_window = {
            "start": candidate["start"],
            "end": candidate["end"],
            "peak_aqi": candidate["max_aqi"],
            "category": candidate["category"],
        }

    return {
        "alert_level": alert_level,
        "alert_status": (
            "ACTION_REQUIRED"
            if alert_level in [
                "HIGH",
                "VERY HIGH",
                "EXTREME",
            ]
            else "MONITOR"
        ),
        "location": location,
        "forecast": {
            "hours": int(len(forecast_df)),
            "start": forecast_df["_timestamp"].iloc[0],
            "end": forecast_df["_timestamp"].iloc[-1],
            "initial_aqi": first_aqi,
            "mean_aqi": mean_aqi,
            "median_aqi": median_aqi,
            "maximum_aqi": max_aqi,
            "minimum_aqi": float(
                forecast_df["_predicted_aqi"].min()
            ),
            "dominant_category": aqi_category(
                mean_aqi
            ),
            "peak_category": aqi_category(
                max_aqi
            ),
            "peak_timestamp": peak_timestamp,
            "trend": trend,
            "trend_delta": trend_delta,
        },
        "exposure": {
            "unhealthy_hours": unhealthy_hours,
            "very_unhealthy_hours": very_unhealthy_hours,
            "hazardous_hours": hazardous_hours,
            "consecutive_unhealthy_hours": (
                max_consecutive_unhealthy
            ),
            "consecutive_very_unhealthy_hours": (
                max_consecutive_very_unhealthy
            ),
        },
        "pollution_driver": {
            "dominant_pollutant": dominant,
            "relative_percentile": dominant_percentile,
        },
        "risk_window": watch_window,
        "recommended_action": action,
        "generated_at": pd.Timestamp.now(
            tz="UTC"
        ),
    }


def calculate_consecutive_hours(
    mask: pd.Series,
) -> int:

    maximum = 0
    current = 0

    for value in mask.tolist():

        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum


def determine_alert_level(
    max_aqi: float,
    unhealthy_hours: int,
    very_unhealthy_hours: int,
    hazardous_hours: int,
    trend: str,
    dominant_percentile: float,
) -> str:

    if hazardous_hours > 0 or max_aqi >= 301:
        return "EXTREME"

    if very_unhealthy_hours >= 3 or max_aqi >= 201:
        return "VERY HIGH"

    if max_aqi >= 151:

        if (
            unhealthy_hours >= 6
            or (
                trend == "rising"
                and dominant_percentile >= 75
            )
        ):
            return "HIGH"

        return "ELEVATED"

    if max_aqi >= 101:
        return "MODERATE"

    return "LOW"


def recommended_action(alert_level: str) -> str:

    actions = {
        "LOW": (
            "Air quality is expected to remain in the "
            "lower-risk range. Normal activities can continue."
        ),
        "MODERATE": (
            "Monitor air quality. Sensitive individuals "
            "should consider reducing prolonged outdoor exposure "
            "if conditions worsen."
        ),
        "ELEVATED": (
            "Take precautions. Sensitive individuals should "
            "reduce prolonged or heavy outdoor activity during "
            "the highest-AQI periods."
        ),
        "HIGH": (
            "Take precautions. Reduce prolonged outdoor exposure "
            "during high-AQI periods and monitor the forecast."
        ),
        "VERY HIGH": (
            "Limit prolonged outdoor exposure. Sensitive groups "
            "should avoid strenuous outdoor activity during "
            "peak periods."
        ),
        "EXTREME": (
            "Avoid prolonged outdoor exposure and follow local "
            "health guidance. Conditions may pose serious health risks."
        ),
    }

    return actions.get(
        alert_level,
        "Monitor air quality conditions.",
    )


# ============================================================================
# PACKAGE
# ============================================================================

def build_package(
    location: Dict[str, Any],
    forecast_df: pd.DataFrame,
    risk_hourly: pd.DataFrame,
    risk_windows: pd.DataFrame,
    pollutant_risk: pd.DataFrame,
    category_exposure: pd.DataFrame,
    alerts: Dict[str, Any],
    input_hash: str,
) -> Dict[str, Any]:

    return {
        "schema_version": "1.0",
        "step": 24,
        "name": "AQI Risk + Alert Intelligence",
        "target": TARGET,
        "forecast_horizon": FORECAST_HORIZON,
        "location": location,
        "alert_intelligence": alerts,
        "pollutants": json_safe(pollutant_risk),
        "category_exposure": json_safe(category_exposure),
        "risk_windows": json_safe(risk_windows),
        "risk_hourly_preview": json_safe(
            risk_hourly.head(72)
        ),
        "provenance": {
            "forecast_source": str(FORECAST_FILE),
            "pollution_source": str(
                POLLUTION_HOURLY_FILE
            ),
            "location_source": str(LOCATION_FILE),
            "input_hash": input_hash,
            "model_training": False,
            "model_selection": False,
            "validation_used": False,
            "test_used": False,
            "future_target_leakage": "NONE",
        },
        "generated_at": pd.Timestamp.now(
            tz="UTC"
        ),
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_forecast(
    forecast_df: pd.DataFrame,
) -> Tuple[bool, List[str]]:

    checks = []

    checks.append(
        len(forecast_df) == FORECAST_HORIZON
    )

    checks.append(
        forecast_df["_predicted_aqi"]
        .apply(np.isfinite)
        .all()
    )

    checks.append(
        (forecast_df["_predicted_aqi"] >= 0).all()
    )

    timestamps = forecast_df["_timestamp"]

    checks.append(
        timestamps.duplicated().sum() == 0
    )

    if len(timestamps) > 1:
        diffs = timestamps.diff().dropna()

        checks.append(
            (diffs == pd.Timedelta(hours=1)).all()
        )

    return all(checks), checks


def validate_outputs() -> Dict[str, bool]:

    outputs = {
        "risk_intelligence": RISK_JSON,
        "risk_hourly": RISK_HOURLY_CSV,
        "risk_windows": RISK_WINDOWS_CSV,
        "pollutant_risk": POLLUTANT_RISK_CSV,
        "category_exposure": CATEGORY_EXPOSURE_CSV,
        "alert_intelligence": ALERT_JSON,
        "package": PACKAGE_JSON,
        "cache_hash": CACHE_HASH_FILE,
        "results": RESULTS_JSON,
    }

    return {
        name: path.exists() and path.stat().st_size > 0
        for name, path in outputs.items()
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    started = time.perf_counter()

    banner("PEARLS AQI PREDICTOR")
    banner(STEP_NAME)

    info("Base directory", BASE_DIR)
    info("Target", TARGET)
    info("Forecast horizon", FORECAST_HORIZON)
    info("Input dashboard", "STEP 20")
    info("Pollution intelligence", "STEP 21")
    info("Dashboard consumer", "STEP 23")
    info("Caching", "Enabled")
    info("Model training", "NOT performed")
    info("Model selection", "NOT performed")
    info("Validation/test", "NOT performed")

    # ---------------------------------------------------------------------
    # INPUT CHECK
    # ---------------------------------------------------------------------

    banner("VERIFYING INPUTS")

    required = [
        FORECAST_FILE,
        LOCATION_FILE,
    ]

    optional = [
        POLLUTION_FILE,
        POLLUTION_HOURLY_FILE,
        AQI_INTELLIGENCE_FILE,
        EVENTS_FILE,
    ]

    for path in required:
        if not path.exists():
            fail(f"Required input not found:\n{path}")

        info(path.name, f"FOUND -> {path}")

    for path in optional:
        if path.exists():
            info(path.name, f"FOUND -> {path}")
        else:
            info(path.name, "NOT AVAILABLE")

    # ---------------------------------------------------------------------
    # FORECAST
    # ---------------------------------------------------------------------

    banner("LOADING PRODUCTION FORECAST")

    forecast = load_forecast()

    info("Forecast rows", len(forecast))
    info(
        "Forecast start",
        forecast["_timestamp"].iloc[0],
    )
    info(
        "Forecast end",
        forecast["_timestamp"].iloc[-1],
    )

    # ---------------------------------------------------------------------
    # LOCATION
    # ---------------------------------------------------------------------

    banner("BUILDING LOCATION INTELLIGENCE")

    location_data = read_json(
        LOCATION_FILE,
        {},
    )

    location = build_location(
        location_data
        if isinstance(location_data, dict)
        else {}
    )

    info("City", location["city"])
    info("Country", location["country"])
    info("Latitude", location["latitude"])
    info("Longitude", location["longitude"])
    info("Station", location["station"])
    info("Timezone", location["timezone"])

    # ---------------------------------------------------------------------
    # POLLUTION
    # ---------------------------------------------------------------------

    banner("LOADING POLLUTION INTELLIGENCE")

    pollution_df, pollutant_mapping = (
        load_pollution_data()
    )

    if pollution_df.empty:
        info("Pollution hourly", "NOT AVAILABLE")
    else:
        info(
            "Pollution hourly rows",
            len(pollution_df),
        )

    info(
        "Pollutants discovered",
        len(pollutant_mapping),
    )

    for pollutant, column in pollutant_mapping.items():
        info(
            pollutant.upper(),
            f"FOUND -> {column}",
        )

    # ---------------------------------------------------------------------
    # CACHE
    # ---------------------------------------------------------------------

    banner("CHECKING RISK INTELLIGENCE CACHE")

    input_hash = build_input_hash()
    cache_hit = check_cache(input_hash)

    info(
        "Cache status",
        "HIT" if cache_hit else "MISS",
    )

    # Even on cache hit, the script loads/validates the existing
    # package so the command remains operationally useful.

    if cache_hit:

        package = read_json(
            PACKAGE_JSON,
            {},
        )

        alert_data = read_json(
            ALERT_JSON,
            {},
        )

        banner("CACHE HIT — LOADING EXISTING PACKAGE")

        validation = validate_outputs()

        all_valid = all(validation.values())

        info(
            "Cached package validation",
            "PASS" if all_valid else "FAIL",
        )

        if not all_valid:
            cache_hit = False

    if not cache_hit:

        # -------------------------------------------------------------
        # POLLUTANT RISK
        # -------------------------------------------------------------

        banner("CALCULATING POLLUTANT RISK")

        pollutant_risk = calculate_pollutant_risk(
            pollution_df,
            pollutant_mapping,
        )

        dominant, dominant_percentile = (
            dominant_pollutant(
                pollutant_risk
            )
        )

        info(
            "Dominant pollutant",
            dominant,
        )

        info(
            "Relative intensity",
            f"{dominant_percentile:.2f} percentile",
        )

        # -------------------------------------------------------------
        # CATEGORY EXPOSURE
        # -------------------------------------------------------------

        banner("CALCULATING AQI CATEGORY EXPOSURE")

        category_exposure = (
            build_category_exposure(
                forecast
            )
        )

        for _, row in category_exposure.iterrows():

            if int(row["hours"]) > 0:
                info(
                    row["category"],
                    f"{int(row['hours'])} hours "
                    f"({row['percentage']:.1f}%)",
                )

        # -------------------------------------------------------------
        # HOURLY RISK
        # -------------------------------------------------------------

        banner("BUILDING HOURLY AQI RISK")

        risk_hourly = build_risk_hourly(
            forecast,
            dominant_percentile,
        )

        # -------------------------------------------------------------
        # WINDOWS
        # -------------------------------------------------------------

        banner("DETECTING RISK WINDOWS")

        risk_windows = build_risk_windows(
            risk_hourly
        )

        info(
            "Risk windows",
            len(risk_windows),
        )

        if not risk_windows.empty:

            peak_window = risk_windows.loc[
                risk_windows["max_aqi"].idxmax()
            ]

            info(
                "Highest-risk window",
                (
                    f"{peak_window['start']} -> "
                    f"{peak_window['end']}"
                ),
            )

            info(
                "Peak window AQI",
                f"{peak_window['max_aqi']:.3f}",
            )

        # -------------------------------------------------------------
        # ALERT
        # -------------------------------------------------------------

        banner("BUILDING AQI ALERT INTELLIGENCE")

        alerts = build_alert_intelligence(
            forecast_df=forecast,
            risk_hourly=risk_hourly,
            risk_windows=risk_windows,
            pollutant_risk=pollutant_risk,
            location=location,
        )

        info(
            "Alert level",
            alerts["alert_level"],
        )

        info(
            "Alert status",
            alerts["alert_status"],
        )

        info(
            "Initial AQI",
            f"{alerts['forecast']['initial_aqi']:.3f}",
        )

        info(
            "Forecast mean",
            f"{alerts['forecast']['mean_aqi']:.3f}",
        )

        info(
            "Forecast maximum",
            f"{alerts['forecast']['maximum_aqi']:.3f}",
        )

        info(
            "Forecast trend",
            alerts["forecast"]["trend"],
        )

        info(
            "Peak timestamp",
            alerts["forecast"]["peak_timestamp"],
        )

        info(
            "Unhealthy hours",
            alerts["exposure"]["unhealthy_hours"],
        )

        info(
            "Very unhealthy hours",
            alerts["exposure"][
                "very_unhealthy_hours"
            ],
        )

        info(
            "Hazardous hours",
            alerts["exposure"]["hazardous_hours"],
        )

        # -------------------------------------------------------------
        # PACKAGE
        # -------------------------------------------------------------

        banner("BUILDING RISK INTELLIGENCE PACKAGE")

        package = build_package(
            location=location,
            forecast_df=forecast,
            risk_hourly=risk_hourly,
            risk_windows=risk_windows,
            pollutant_risk=pollutant_risk,
            category_exposure=category_exposure,
            alerts=alerts,
            input_hash=input_hash,
        )

        # -------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------

        banner("SAVING AQI RISK + ALERT INTELLIGENCE")

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json(
            RISK_JSON,
            {
                "location": location,
                "forecast": alerts["forecast"],
                "exposure": alerts["exposure"],
                "pollution_driver": alerts[
                    "pollution_driver"
                ],
                "risk_windows": risk_windows,
                "generated_at": pd.Timestamp.now(
                    tz="UTC"
                ),
            },
        )

        risk_hourly.to_csv(
            RISK_HOURLY_CSV,
            index=False,
        )

        risk_windows.to_csv(
            RISK_WINDOWS_CSV,
            index=False,
        )

        pollutant_risk.to_csv(
            POLLUTANT_RISK_CSV,
            index=False,
        )

        category_exposure.to_csv(
            CATEGORY_EXPOSURE_CSV,
            index=False,
        )

        write_json(
            ALERT_JSON,
            alerts,
        )

        write_json(
            PACKAGE_JSON,
            package,
        )

        save_cache_hash(
            input_hash
        )

    # ---------------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------------

    banner("VALIDATING AQI RISK + ALERT INTELLIGENCE")

    forecast_valid, forecast_checks = (
        validate_forecast(forecast)
    )

    info(
        "Forecast validation",
        "PASS" if forecast_valid else "FAIL",
    )

    output_validation = validate_outputs()

    for name, result in output_validation.items():
        info(
            name,
            "PASS" if result else "FAIL",
        )

    package_data = read_json(
        PACKAGE_JSON,
        {},
    )

    serialization_ok = True

    try:
        json.dumps(
            package_data,
            allow_nan=False,
        )
    except Exception:
        serialization_ok = False

    info(
        "JSON serialization",
        "PASS" if serialization_ok else "FAIL",
    )

    overall_validation = (
        forecast_valid
        and all(output_validation.values())
        and serialization_ok
    )

    info(
        "Overall validation",
        "PASS" if overall_validation else "FAIL",
    )

    # ---------------------------------------------------------------------
    # RESULT SUMMARY
    # ---------------------------------------------------------------------

    alerts = read_json(
        ALERT_JSON,
        {},
    )

    execution_time = (
        time.perf_counter() - started
    )

    results = {
        "step": 24,
        "step_name": STEP_NAME,
        "status": (
            "PASS"
            if overall_validation
            else "FAIL"
        ),
        "target": TARGET,
        "location": location,
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_rows": len(forecast),
        "forecast_start": forecast[
            "_timestamp"
        ].iloc[0],
        "forecast_end": forecast[
            "_timestamp"
        ].iloc[-1],
        "initial_aqi": alerts.get(
            "forecast",
            {},
        ).get("initial_aqi"),
        "forecast_mean_aqi": alerts.get(
            "forecast",
            {},
        ).get("mean_aqi"),
        "forecast_maximum_aqi": alerts.get(
            "forecast",
            {},
        ).get("maximum_aqi"),
        "forecast_minimum_aqi": alerts.get(
            "forecast",
            {},
        ).get("minimum_aqi"),
        "dominant_aqi_category": alerts.get(
            "forecast",
            {},
        ).get("dominant_category"),
        "forecast_trend": alerts.get(
            "forecast",
            {},
        ).get("trend"),
        "peak_timestamp": alerts.get(
            "forecast",
            {},
        ).get("peak_timestamp"),
        "alert_level": alerts.get(
            "alert_level"
        ),
        "alert_status": alerts.get(
            "alert_status"
        ),
        "dominant_pollutant": alerts.get(
            "pollution_driver",
            {},
        ).get("dominant_pollutant"),
        "dominant_pollutant_percentile": alerts.get(
            "pollution_driver",
            {},
        ).get("relative_percentile"),
        "unhealthy_hours": alerts.get(
            "exposure",
            {},
        ).get("unhealthy_hours"),
        "very_unhealthy_hours": alerts.get(
            "exposure",
            {},
        ).get("very_unhealthy_hours"),
        "hazardous_hours": alerts.get(
            "exposure",
            {},
        ).get("hazardous_hours"),
        "risk_windows": (
            len(risk_windows)
            if "risk_windows" in locals()
            else 0
        ),
        "cache_status": (
            "HIT"
            if cache_hit
            else "MISS"
        ),
        "model_training": False,
        "model_selection": False,
        "validation_used": False,
        "test_used": False,
        "future_target_leakage": "NONE",
        "forecast_validation": forecast_valid,
        "json_serialization": serialization_ok,
        "dashboard_consumer": "STEP 23",
        "generated_at": pd.Timestamp.now(
            tz="UTC"
        ),
        "execution_time_seconds": round(
            execution_time,
            3,
        ),
        "outputs": {
            "risk_intelligence": str(
                RISK_JSON
            ),
            "risk_hourly": str(
                RISK_HOURLY_CSV
            ),
            "risk_windows": str(
                RISK_WINDOWS_CSV
            ),
            "pollutant_risk": str(
                POLLUTANT_RISK_CSV
            ),
            "category_exposure": str(
                CATEGORY_EXPOSURE_CSV
            ),
            "alert_intelligence": str(
                ALERT_JSON
            ),
            "package": str(
                PACKAGE_JSON
            ),
            "cache_hash": str(
                CACHE_HASH_FILE
            ),
        },
    }

    write_json(
        RESULTS_JSON,
        results,
    )

    # ---------------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------------

    banner("STEP 24 COMPLETE")

    info(
        "Status",
        results["status"],
    )

    info(
        "Location",
        f"{location['city']}, {location['country']}",
    )

    info(
        "Station",
        location["station"],
    )

    info(
        "Initial AQI",
        (
            f"{results['initial_aqi']:.3f}"
            if results["initial_aqi"] is not None
            else "N/A"
        ),
    )

    info(
        "Forecast AQI mean",
        (
            f"{results['forecast_mean_aqi']:.3f}"
            if results["forecast_mean_aqi"] is not None
            else "N/A"
        ),
    )

    info(
        "Forecast AQI maximum",
        (
            f"{results['forecast_maximum_aqi']:.3f}"
            if results["forecast_maximum_aqi"] is not None
            else "N/A"
        ),
    )

    info(
        "Dominant AQI category",
        results["dominant_aqi_category"],
    )

    info(
        "Forecast trend",
        results["forecast_trend"],
    )

    info(
        "Dominant pollutant",
        results["dominant_pollutant"],
    )

    info(
        "Alert level",
        results["alert_level"],
    )

    info(
        "Unhealthy hours",
        results["unhealthy_hours"],
    )

    info(
        "Risk windows",
        results["risk_windows"],
    )

    info(
        "Cache status",
        results["cache_status"],
    )

    info(
        "Future target leakage",
        results["future_target_leakage"],
    )

    info(
        "Validation",
        (
            "PASS"
            if overall_validation
            else "FAIL"
        ),
    )

    print()
    print("Risk intelligence:")
    print(RISK_JSON)

    print()
    print("Risk hourly:")
    print(RISK_HOURLY_CSV)

    print()
    print("Risk windows:")
    print(RISK_WINDOWS_CSV)

    print()
    print("Pollutant risk:")
    print(POLLUTANT_RISK_CSV)

    print()
    print("Category exposure:")
    print(CATEGORY_EXPOSURE_CSV)

    print()
    print("Alert intelligence:")
    print(ALERT_JSON)

    print()
    print("Risk package:")
    print(PACKAGE_JSON)

    print()
    print("Step 24 report:")
    print(RESULTS_JSON)

    print()
    info(
        "Execution time",
        f"{execution_time:.3f}s",
    )

    if not overall_validation:
        sys.exit(1)


if __name__ == "__main__":
    main()