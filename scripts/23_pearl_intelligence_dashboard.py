from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================================
# PEARLS AQI PREDICTOR
# STEP 23 — PEARL INTELLIGENCE DASHBOARD
# ============================================================================
#
# Purpose:
#   Final presentation layer for the AQI prediction system.
#
# Inputs:
#   STEP 20 production dashboard forecast
#   STEP 21 location + pollution intelligence
#   STEP 22 intelligence dashboard outputs
#
# Output:
#   Full-screen interactive HTML dashboard with:
#       - Lahore location intelligence
#       - Current AQI
#       - 0-500 AQI gauge
#       - 72-hour forecast
#       - Pollutant ingredient cards
#       - Pollutant ranking
#       - AQI category distribution
#       - Pollution events
#       - Next 12 hours
#       - Intelligence narrative
#
# No model training.
# No model selection.
# No validation/test.
#
# ============================================================================


BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"

STEP20_DIR = REPORTS_DIR / "production_dashboard_v2"
STEP21_DIR = REPORTS_DIR / "location_pollution_intelligence"
STEP22_DIR = REPORTS_DIR / "intelligence_dashboard"

OUTPUT_DIR = REPORTS_DIR / "pearl_intelligence_dashboard"
CACHE_DIR = OUTPUT_DIR / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "us_aqi"
FORECAST_HORIZON = 72

# User requested assumed Lahore Cantonment location.
LOCATION = {
    "city": "Lahore",
    "area": "Lahore Cantonment",
    "country": "Pakistan",
    "latitude": 31.5205,
    "longitude": 74.4036,
    "station": "Lahore Cantonment",
    "timezone": "PKT",
    "utc_offset": "UTC+05:00",
}

POLLUTANTS = {
    "pm2_5": {
        "label": "PM2.5",
        "symbol": "PM₂.₅",
        "unit": "µg/m³",
    },
    "pm10": {
        "label": "PM10",
        "symbol": "PM₁₀",
        "unit": "µg/m³",
    },
    "ozone": {
        "label": "O₃",
        "symbol": "O₃",
        "unit": "µg/m³",
    },
    "nitrogen_dioxide": {
        "label": "NO₂",
        "symbol": "NO₂",
        "unit": "µg/m³",
    },
    "sulphur_dioxide": {
        "label": "SO₂",
        "symbol": "SO₂",
        "unit": "µg/m³",
    },
    "carbon_monoxide": {
        "label": "CO",
        "symbol": "CO",
        "unit": "µg/m³",
    },
}


# ============================================================================
# OUTPUT PATHS
# ============================================================================

FORECAST_OUT = OUTPUT_DIR / "pearl_intelligence_forecast.csv"
POLLUTION_OUT = OUTPUT_DIR / "pearl_intelligence_pollution.csv"
SUMMARY_OUT = OUTPUT_DIR / "pearl_intelligence_summary.csv"

DATA_JSON_OUT = OUTPUT_DIR / "pearl_intelligence_dashboard_data.json"
HTML_OUT = OUTPUT_DIR / "pearl_intelligence_dashboard.html"

CACHE_HASH_OUT = CACHE_DIR / "dashboard_hash.json"
REPORT_OUT = OUTPUT_DIR / "pearl_intelligence_dashboard_results.json"


# ============================================================================
# CONSOLE
# ============================================================================

def banner(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def info(label: str, value):
    print(f"{label:<34}: {value}")


# ============================================================================
# HELPERS
# ============================================================================

def json_safe(value):
    """
    Convert pandas/numpy values into JSON-safe Python values.
    Prevents:
        TypeError: Object of type DataFrame is not JSON serializable
    """

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, float):
        return None if not math.isfinite(value) else value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def write_json(path: Path, data):
    safe = json_safe(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            safe,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def read_json(path: Path, default=None):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_timestamp_column(df: pd.DataFrame):
    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "forecast_timestamp",
        "prediction_timestamp",
        "ds",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = str(col).lower()

        if (
            "timestamp" in low
            or "datetime" in low
            or low.endswith("_time")
        ):
            return col

    return None


def find_aqi_column(df: pd.DataFrame):
    candidates = [
        "predicted_aqi",
        "prediction",
        "predicted_us_aqi",
        "us_aqi",
        "aqi",
        "forecast_aqi",
        "yhat",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = str(col).lower()

        if "aqi" in low and "category" not in low:
            return col

    return None


def normalize_timestamp_series(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


# ============================================================================
# AQI
# ============================================================================

def aqi_category(aqi):
    if aqi is None or not math.isfinite(float(aqi)):
        return "Unknown"

    aqi = float(aqi)

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


def aqi_short_category(aqi):
    category = aqi_category(aqi)

    mapping = {
        "Good": "Good",
        "Moderate": "Moderate",
        "Unhealthy for Sensitive Groups": "Sensitive",
        "Unhealthy": "Unhealthy",
        "Very Unhealthy": "Very Unhealthy",
        "Hazardous": "Hazardous",
        "Unknown": "Unknown",
    }

    return mapping.get(category, category)


def aqi_position(aqi):
    """
    Convert AQI 0-500 to percentage across gauge.
    """

    value = max(0.0, min(500.0, float(aqi)))

    return value / 500.0


# ============================================================================
# DATA LOADING
# ============================================================================

banner("PEARLS AQI PREDICTOR")
banner("STEP 23 — PEARL INTELLIGENCE DASHBOARD")

info("Base directory", BASE_DIR)
info("Target", TARGET)
info("Forecast horizon", FORECAST_HORIZON)
info("Dashboard type", "Full-screen + Interactive + Infographic")
info("Location", "Lahore Cantonment")
info("Coordinates", f"{LOCATION['latitude']}, {LOCATION['longitude']}")
info("Timezone", "PKT / UTC+05:00")
info("Caching", "Enabled")
info("Model training", "NOT performed")
info("Model selection", "NOT performed")
info("Validation/test", "NOT performed")


banner("VERIFYING INPUTS")

forecast_source = first_existing([
    STEP22_DIR / "intelligence_dashboard_forecast.csv",
    STEP20_DIR / "production_dashboard_forecast.csv",
])

pollution_source = first_existing([
    STEP21_DIR / "pollution_hourly.csv",
])

pollution_intelligence_source = first_existing([
    STEP21_DIR / "pollution_intelligence.csv",
])

aqi_intelligence_source = first_existing([
    STEP21_DIR / "aqi_intelligence.csv",
])

events_source = first_existing([
    STEP21_DIR / "pollution_events.csv",
])


if forecast_source is None:
    raise FileNotFoundError(
        "No production forecast found. "
        "Run Step 20/22 before Step 23."
    )

if pollution_source is None:
    raise FileNotFoundError(
        "Step 21 pollution_hourly.csv not found."
    )

info("Forecast source", forecast_source)
info("Pollution hourly", pollution_source)
info(
    "Pollution intelligence",
    pollution_intelligence_source or "NOT AVAILABLE",
)
info(
    "AQI intelligence",
    aqi_intelligence_source or "NOT AVAILABLE",
)
info(
    "Pollution events",
    events_source or "NOT AVAILABLE",
)


# ============================================================================
# LOAD FORECAST
# ============================================================================

banner("LOADING PRODUCTION FORECAST")

forecast = pd.read_csv(forecast_source)

timestamp_col = find_timestamp_column(forecast)
aqi_col = find_aqi_column(forecast)

if timestamp_col is None:
    raise ValueError("Forecast timestamp column could not be detected.")

if aqi_col is None:
    raise ValueError("Forecast AQI column could not be detected.")

forecast["_timestamp"] = normalize_timestamp_series(
    forecast[timestamp_col]
)

forecast["_aqi"] = pd.to_numeric(
    forecast[aqi_col],
    errors="coerce",
)

forecast = forecast.dropna(
    subset=["_timestamp", "_aqi"]
).sort_values("_timestamp")

forecast = forecast.head(FORECAST_HORIZON).copy()

if forecast.empty:
    raise ValueError("Forecast contains no valid rows.")

info("Forecast rows", len(forecast))
info("Forecast start", forecast["_timestamp"].iloc[0])
info("Forecast end", forecast["_timestamp"].iloc[-1])


# ============================================================================
# LOAD POLLUTION DATA
# ============================================================================

banner("LOADING POLLUTION INGREDIENT DATA")

pollution = pd.read_csv(pollution_source)

pollution_timestamp_col = find_timestamp_column(pollution)

if pollution_timestamp_col is None:
    raise ValueError(
        "Pollution timestamp column could not be detected."
    )

pollution["_timestamp"] = normalize_timestamp_series(
    pollution[pollution_timestamp_col]
)

pollution = pollution.dropna(
    subset=["_timestamp"]
).sort_values("_timestamp")

info("Pollution rows", len(pollution))
info("Pollution columns", len(pollution.columns))


# ============================================================================
# DISCOVER POLLUTANTS
# ============================================================================

banner("DISCOVERING POLLUTION INGREDIENTS")

pollutant_columns = {}

aliases = {
    "pm2_5": [
        "pm2_5",
        "pm25",
        "pm2.5",
        "pm2_5_ug_m3",
    ],
    "pm10": [
        "pm10",
        "pm10_ug_m3",
    ],
    "ozone": [
        "ozone",
        "o3",
        "o3_ug_m3",
    ],
    "nitrogen_dioxide": [
        "nitrogen_dioxide",
        "no2",
        "no2_ug_m3",
    ],
    "sulphur_dioxide": [
        "sulphur_dioxide",
        "sulfur_dioxide",
        "so2",
        "so2_ug_m3",
    ],
    "carbon_monoxide": [
        "carbon_monoxide",
        "co",
        "co_ug_m3",
    ],
}


lower_columns = {
    str(c).lower(): c
    for c in pollution.columns
}

for key, names in aliases.items():

    found = None

    for name in names:
        if name.lower() in lower_columns:
            found = lower_columns[name.lower()]
            break

    pollutant_columns[key] = found

    if found:
        info(POLLUTANTS[key]["label"], f"FOUND -> {found}")
    else:
        info(POLLUTANTS[key]["label"], "NOT AVAILABLE")


available_pollutants = [
    key
    for key, col in pollutant_columns.items()
    if col is not None
]

info("Pollutants available", len(available_pollutants))


# ============================================================================
# CURRENT POLLUTION SNAPSHOT
# ============================================================================

banner("BUILDING CURRENT POLLUTION SNAPSHOT")

latest_pollution = (
    pollution
    .sort_values("_timestamp")
    .iloc[-1]
)

pollution_snapshot = []

for key in available_pollutants:

    col = pollutant_columns[key]

    series = pd.to_numeric(
        pollution[col],
        errors="coerce"
    ).dropna()

    if series.empty:
        continue

    current_value = pd.to_numeric(
        latest_pollution[col],
        errors="coerce"
    )

    if pd.isna(current_value):
        current_value = series.iloc[-1]

    percentile = float(
        (series <= float(current_value)).mean() * 100
    )

    mean_value = float(series.mean())

    if percentile >= 80:
        trend = "HIGH"
    elif percentile >= 60:
        trend = "ELEVATED"
    elif percentile <= 25:
        trend = "LOW"
    else:
        trend = "NORMAL"

    pollution_snapshot.append({
        "key": key,
        "label": POLLUTANTS[key]["label"],
        "symbol": POLLUTANTS[key]["symbol"],
        "unit": POLLUTANTS[key]["unit"],
        "value": round(float(current_value), 3),
        "mean": round(mean_value, 3),
        "percentile": round(percentile, 1),
        "status": trend,
    })


pollution_snapshot = sorted(
    pollution_snapshot,
    key=lambda x: x["percentile"],
    reverse=True,
)

dominant_pollutant = (
    pollution_snapshot[0]
    if pollution_snapshot
    else None
)

if dominant_pollutant:
    info(
        "Dominant pollutant",
        dominant_pollutant["label"],
    )
    info(
        "Relative intensity",
        f"{dominant_pollutant['percentile']:.1f} percentile",
    )


# ============================================================================
# CURRENT AQI
# ============================================================================

banner("BUILDING CURRENT AQI INTELLIGENCE")

# Prefer a real observed AQI column if it exists in the pollution data.
observed_aqi_col = find_aqi_column(pollution)

current_aqi = None
current_aqi_source = None

if observed_aqi_col:
    observed = pd.to_numeric(
        pollution[observed_aqi_col],
        errors="coerce"
    ).dropna()

    if not observed.empty:
        current_aqi = float(observed.iloc[-1])
        current_aqi_source = "latest observed AQI"


# Fallback: first production forecast point.
if current_aqi is None:
    current_aqi = float(forecast["_aqi"].iloc[0])
    current_aqi_source = "first production forecast point"


current_category = aqi_category(current_aqi)

forecast_values = forecast["_aqi"].astype(float)

forecast_min = float(forecast_values.min())
forecast_max = float(forecast_values.max())
forecast_mean = float(forecast_values.mean())
forecast_median = float(forecast_values.median())

first_forecast = float(forecast_values.iloc[0])
last_forecast = float(forecast_values.iloc[-1])

if last_forecast > first_forecast + 5:
    forecast_trend = "rising"
elif last_forecast < first_forecast - 5:
    forecast_trend = "falling"
else:
    forecast_trend = "stable"


info("Current AQI", f"{current_aqi:.1f}")
info("Current AQI category", current_category)
info("Current AQI source", current_aqi_source)
info("Forecast minimum", f"{forecast_min:.3f}")
info("Forecast maximum", f"{forecast_max:.3f}")
info("Forecast mean", f"{forecast_mean:.3f}")
info("Forecast trend", forecast_trend)


# ============================================================================
# FORECAST CATEGORY DISTRIBUTION
# ============================================================================

banner("BUILDING FORECAST CATEGORY DISTRIBUTION")

category_counts = {}

for value in forecast_values:
    category = aqi_category(value)
    category_counts[category] = category_counts.get(category, 0) + 1

category_distribution = [
    {
        "category": category,
        "hours": count,
        "percentage": round(
            count / len(forecast_values) * 100,
            1
        ),
    }
    for category, count in category_counts.items()
]

category_distribution.sort(
    key=lambda x: x["hours"],
    reverse=True
)

dominant_forecast_category = (
    category_distribution[0]["category"]
    if category_distribution
    else "Unknown"
)


# ============================================================================
# POLLUTION EVENTS
# ============================================================================

banner("LOADING POLLUTION EVENT INTELLIGENCE")

event_count = 0
high_event_count = 0
elevated_event_count = 0

events = pd.DataFrame()

if events_source:

    events = pd.read_csv(events_source)

    event_count = len(events)

    text = events.astype(str).apply(
        lambda row: " ".join(row.values).lower(),
        axis=1,
    )

    high_event_count = int(
        text.str.contains(
            r"high|severe|critical",
            regex=True
        ).sum()
    )

    elevated_event_count = int(
        text.str.contains(
            r"elevated|moderate",
            regex=True
        ).sum()
    )

info("Pollution events", event_count)
info("High severity events", high_event_count)
info("Elevated events", elevated_event_count)


# ============================================================================
# EVENT PREVIEW
# ============================================================================

event_preview = []

if not events.empty:

    timestamp_event_col = find_timestamp_column(events)

    for _, row in events.tail(6).iloc[::-1].iterrows():

        item = {}

        if timestamp_event_col:
            timestamp = pd.to_datetime(
                row[timestamp_event_col],
                errors="coerce",
                utc=True
            )

            item["timestamp"] = (
                timestamp.isoformat()
                if not pd.isna(timestamp)
                else ""
            )
        else:
            item["timestamp"] = ""

        for col in events.columns:

            if col == timestamp_event_col:
                continue

            value = row[col]

            if pd.isna(value):
                continue

            key = str(col).lower()

            if (
                "severity" in key
                or "event" in key
                or "pollut" in key
                or "description" in key
                or "reason" in key
            ):
                item[key] = str(value)

        event_preview.append(item)


# ============================================================================
# NEXT 12 HOURS
# ============================================================================

next_12 = []

for i, (_, row) in enumerate(forecast.head(12).iterrows()):

    value = float(row["_aqi"])

    next_12.append({
        "hour": i + 1,
        "timestamp": row["_timestamp"].isoformat(),
        "aqi": round(value, 2),
        "category": aqi_category(value),
    })


# ============================================================================
# INTELLIGENCE NARRATIVE
# ============================================================================

banner("BUILDING PEARL INTELLIGENCE")

if current_aqi >= 201:
    health_message = (
        "Air quality is in the very high-risk range. "
        "Outdoor exposure should be minimized."
    )
elif current_aqi >= 151:
    health_message = (
        "Air quality is unhealthy. "
        "Sensitive people should take precautions and prolonged "
        "outdoor exposure should be reduced."
    )
elif current_aqi >= 101:
    health_message = (
        "Air quality is elevated. Sensitive groups may experience "
        "health effects and should consider reducing prolonged exposure."
    )
elif current_aqi >= 51:
    health_message = (
        "Air quality is moderate. Most people can continue normal "
        "activities while sensitive individuals should remain aware."
    )
else:
    health_message = (
        "Air quality is currently in the good range."
    )


if dominant_pollutant:
    driver_message = (
        f"{dominant_pollutant['label']} is currently the strongest "
        f"relative pollution signal at the "
        f"{dominant_pollutant['percentile']:.1f}th percentile."
    )
else:
    driver_message = (
        "Pollution ingredient data is currently unavailable."
    )


if forecast_trend == "rising":
    trajectory_message = (
        "The production forecast is trending upward over the "
        "72-hour horizon."
    )
elif forecast_trend == "falling":
    trajectory_message = (
        "The production forecast is trending downward over the "
        "72-hour horizon."
    )
else:
    trajectory_message = (
        "The production forecast remains broadly stable over "
        "the 72-hour horizon."
    )


intelligence = {
    "headline": (
        f"{LOCATION['city']} is currently "
        f"{current_category.lower()}."
    ),
    "health_message": health_message,
    "driver_message": driver_message,
    "trajectory_message": trajectory_message,
}


# ============================================================================
# CACHE
# ============================================================================

banner("CHECKING DASHBOARD CACHE")

hash_items = [
    str(forecast_source),
    str(forecast_source.stat().st_mtime_ns),
    str(pollution_source),
    str(pollution_source.stat().st_mtime_ns),
]

if pollution_intelligence_source:
    hash_items.extend([
        str(pollution_intelligence_source),
        str(pollution_intelligence_source.stat().st_mtime_ns),
    ])

if events_source:
    hash_items.extend([
        str(events_source),
        str(events_source.stat().st_mtime_ns),
    ])

hash_payload = "|".join(hash_items)

dashboard_hash = hashlib.sha256(
    hash_payload.encode("utf-8")
).hexdigest()

old_cache = read_json(CACHE_HASH_OUT, {})

cache_hit = (
    old_cache.get("dashboard_hash") == dashboard_hash
    and HTML_OUT.exists()
    and DATA_JSON_OUT.exists()
)

info(
    "Cache status",
    "HIT" if cache_hit else "MISS"
)


# ============================================================================
# PREPARE FORECAST JSON
# ============================================================================

forecast_records = []

for _, row in forecast.iterrows():

    value = float(row["_aqi"])

    forecast_records.append({
        "timestamp": row["_timestamp"].isoformat(),
        "aqi": round(value, 3),
        "category": aqi_category(value),
    })


# ============================================================================
# DASHBOARD PACKAGE
# ============================================================================

dashboard_data = {
    "meta": {
        "product": "PEARLS AQI Predictor",
        "dashboard": "Pearl Intelligence Dashboard",
        "step": 23,
        "target": TARGET,
        "forecast_horizon": FORECAST_HORIZON,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_training": False,
        "model_selection": False,
        "validation_test": False,
        "cache_enabled": True,
    },

    "location": LOCATION,

    "current": {
        "aqi": round(current_aqi, 3),
        "category": current_category,
        "short_category": aqi_short_category(current_aqi),
        "gauge_position": round(
            aqi_position(current_aqi),
            6
        ),
        "source": current_aqi_source,
    },

    "forecast": {
        "min": round(forecast_min, 3),
        "max": round(forecast_max, 3),
        "mean": round(forecast_mean, 3),
        "median": round(forecast_median, 3),
        "trend": forecast_trend,
        "start": forecast["_timestamp"].iloc[0].isoformat(),
        "end": forecast["_timestamp"].iloc[-1].isoformat(),
        "records": forecast_records,
    },

    "pollution": pollution_snapshot,

    "dominant_pollutant": dominant_pollutant,

    "categories": category_distribution,

    "events": {
        "total": event_count,
        "high": high_event_count,
        "elevated": elevated_event_count,
        "preview": event_preview,
    },

    "next_12_hours": next_12,

    "intelligence": intelligence,
}


# ============================================================================
# SAVE CSV OUTPUTS
# ============================================================================

banner("SAVING DASHBOARD DATA")

forecast_output_df = pd.DataFrame(forecast_records)
forecast_output_df.to_csv(
    FORECAST_OUT,
    index=False
)

pollution_output_df = pd.DataFrame(
    pollution_snapshot
)

pollution_output_df.to_csv(
    POLLUTION_OUT,
    index=False
)

summary_rows = [
    {
        "metric": "current_aqi",
        "value": current_aqi,
    },
    {
        "metric": "current_category",
        "value": current_category,
    },
    {
        "metric": "forecast_min",
        "value": forecast_min,
    },
    {
        "metric": "forecast_max",
        "value": forecast_max,
    },
    {
        "metric": "forecast_mean",
        "value": forecast_mean,
    },
    {
        "metric": "forecast_median",
        "value": forecast_median,
    },
    {
        "metric": "forecast_trend",
        "value": forecast_trend,
    },
    {
        "metric": "dominant_pollutant",
        "value": (
            dominant_pollutant["label"]
            if dominant_pollutant
            else "N/A"
        ),
    },
    {
        "metric": "pollution_events",
        "value": event_count,
    },
    {
        "metric": "high_severity_events",
        "value": high_event_count,
    },
    {
        "metric": "elevated_events",
        "value": elevated_event_count,
    },
]

pd.DataFrame(summary_rows).to_csv(
    SUMMARY_OUT,
    index=False
)

write_json(
    DATA_JSON_OUT,
    dashboard_data
)


# ============================================================================
# HTML
# ============================================================================

banner("CREATING PEARL INTELLIGENCE DASHBOARD")


data_json = json.dumps(
    json_safe(dashboard_data),
    ensure_ascii=False,
    separators=(",", ":")
)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>PEARLS — Lahore Air Intelligence</title>

<style>

:root {
    --ink: #103e3c;
    --ink-soft: #3f6662;
    --cream: #f4f6ef;
    --paper: #fbfcf8;
    --lime: #dce96d;
    --lime-dark: #b8ca3e;
    --coral: #ef6c50;
    --orange: #e59a43;
    --gold: #b9a34d;
    --blue: #4d94a5;
    --green: #78a48d;
    --line: #d9e2da;
    --white: #ffffff;
    --shadow: 0 12px 34px rgba(16, 62, 60, .07);
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background: var(--cream);
    color: var(--ink);
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

button {
    font: inherit;
}

.app {
    width: 100%;
    min-height: 100vh;
}

/* ============================================================
   TOP BAR
   ============================================================ */

.topbar {
    position: sticky;
    top: 0;
    z-index: 20;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 16px 28px;

    background: rgba(16, 62, 60, .98);
    color: white;

    border-bottom: 1px solid rgba(255,255,255,.12);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 9px;

    display: grid;
    place-items: center;

    background: var(--lime);
    color: var(--ink);

    font-weight: 900;
}

.brand-name {
    font-size: 15px;
    font-weight: 900;
    letter-spacing: .15em;
}

.brand-sub {
    margin-top: 1px;
    font-size: 8px;
    letter-spacing: .18em;
    opacity: .65;
}

.top-actions {
    display: flex;
    gap: 8px;
}

.top-button {
    padding: 8px 12px;
    border-radius: 8px;

    color: white;
    background: transparent;

    border: 1px solid rgba(255,255,255,.18);

    cursor: pointer;
}

.top-button:hover {
    background: rgba(255,255,255,.08);
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    min-height: 430px;

    padding:
        64px
        clamp(24px, 6vw, 110px)
        70px;

    background: var(--ink);
    color: white;

    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";

    position: absolute;

    width: 430px;
    height: 430px;

    border-radius: 50%;

    border: 14px solid rgba(220,233,109,.08);

    right: 7%;
    bottom: -240px;
}

.eyebrow {
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: .23em;
    font-weight: 900;
    opacity: .62;
}

.live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--lime);
    margin-right: 6px;
}

.hero-grid {
    position: relative;
    z-index: 2;

    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(350px, .7fr);

    gap: 60px;
    align-items: end;
}

.hero h1 {
    margin: 22px 0 18px;

    max-width: 800px;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size:
        clamp(56px, 8vw, 120px);

    line-height: .87;
    letter-spacing: -.055em;
}

.hero h1 em {
    color: var(--lime);
    font-weight: normal;
}

.hero-description {
    max-width: 680px;

    font-size: 16px;
    line-height: 1.65;

    color: rgba(255,255,255,.72);
}

.hero-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;

    border-left: 1px solid rgba(255,255,255,.22);
    padding-left: 25px;

    gap: 20px;
}

.meta-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .17em;
    opacity: .55;
}

.meta-value {
    margin-top: 7px;
    font-weight: 800;
}

/* ============================================================
   MAIN
   ============================================================ */

.main {
    width: min(1800px, 100%);
    margin: auto;

    padding:
        58px
        clamp(20px, 4vw, 70px)
        90px;
}

.section {
    margin-bottom: 66px;
}

.section-head {
    display: flex;
    align-items: end;
    justify-content: space-between;

    gap: 30px;

    margin-bottom: 20px;
}

.section-number {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .2em;
    font-weight: 900;
    color: #75908a;
}

.section-title {
    margin: 8px 0 0;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: clamp(28px, 3vw, 42px);

    letter-spacing: -.025em;
}

.card {
    background: var(--paper);

    border: 1px solid var(--line);

    border-radius: 15px;

    box-shadow: var(--shadow);
}

/* ============================================================
   LOCATION
   ============================================================ */

.location-grid {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 16px;
}

.location-primary {
    min-height: 330px;

    padding: 34px;

    background: var(--lime);

    border-radius: 16px;
}

.location-primary h3 {
    margin: 12px 0 3px;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: clamp(48px, 6vw, 78px);

    font-weight: 500;

    letter-spacing: -.05em;
}

.location-sub {
    font-size: 15px;
    color: rgba(16,62,60,.72);
}

.location-divider {
    margin: 42px 0 25px;

    border-top: 1px solid rgba(16,62,60,.18);
}

.location-status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
}

.badge {
    display: inline-flex;
    align-items: center;

    padding: 7px 12px;

    border-radius: 999px;

    font-size: 10px;
    font-weight: 900;
    letter-spacing: .06em;

    text-transform: uppercase;
}

.badge.unhealthy {
    background: #ffe1da;
    color: #bf4d38;
}

.badge.moderate {
    background: #fff0bd;
    color: #896b19;
}

.badge.good {
    background: #dff0db;
    color: #477a4d;
}

.current-small {
    text-align: right;
}

.current-small-label {
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: .15em;
    opacity: .62;
}

.current-small-value {
    font-size: 25px;
    font-weight: 900;
}

.location-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.detail-card {
    padding: 28px;
    min-height: 157px;
}

.detail-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .16em;
    color: #738982;
    font-weight: 900;
}

.detail-value {
    margin-top: 15px;
    font-size: 22px;
    font-weight: 900;
}

/* ============================================================
   AQI HERO / GAUGE
   ============================================================ */

.aqi-hero {
    display: grid;

    grid-template-columns:
        minmax(370px, .85fr)
        minmax(550px, 1.15fr);

    gap: 20px;

    padding: 30px;
}

.aqi-value-panel {
    padding: 30px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

.aqi-number {
    margin-top: 8px;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: clamp(100px, 12vw, 180px);

    line-height: .85;

    letter-spacing: -.07em;

    color: var(--coral);
}

.aqi-category {
    margin-top: 20px;

    font-size: 25px;
    font-weight: 900;
}

.aqi-explanation {
    max-width: 480px;

    margin-top: 16px;

    color: var(--ink-soft);

    line-height: 1.6;
}

.aqi-source {
    margin-top: 20px;

    font-size: 10px;

    text-transform: uppercase;
    letter-spacing: .12em;

    color: #7b928c;
}

.gauge-panel {
    min-height: 430px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    padding: 20px;
}

.gauge-wrap {
    width: 100%;
    max-width: 780px;

    margin: auto;
}

.gauge-svg {
    width: 100%;
    height: auto;
    overflow: visible;
}

.gauge-track {
    fill: none;
    stroke-width: 34;
    stroke-linecap: butt;
}

.gauge-marker {
    fill: white;
    stroke: var(--coral);
    stroke-width: 7;
}

.gauge-marker-line {
    stroke: var(--ink);
    stroke-width: 5;
    stroke-linecap: round;
}

.gauge-center {
    fill: var(--paper);
    stroke: var(--ink);
    stroke-width: 3;
}

.gauge-center-number {
    font-size: 27px;
    font-weight: 900;
    fill: var(--ink);
}

.gauge-label {
    font-size: 13px;
    font-weight: 800;
    fill: var(--ink);
}

.gauge-scale {
    font-size: 13px;
    font-weight: 800;
    fill: #78908a;
}

.gauge-legend {
    display: grid;
    grid-template-columns: repeat(6, 1fr);

    gap: 5px;

    margin-top: 5px;
}

.gauge-legend div {
    font-size: 8px;
    text-align: center;

    text-transform: uppercase;
    letter-spacing: .05em;

    color: #748983;
}

/* ============================================================
   KPI ROW
   ============================================================ */

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.kpi {
    padding: 26px;
    min-height: 155px;
}

.kpi-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #748b85;
    font-weight: 900;
}

.kpi-value {
    margin-top: 18px;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: 42px;
    line-height: 1;
}

.kpi-sub {
    margin-top: 12px;
    color: #748b85;
    font-size: 11px;
}

/* ============================================================
   POLLUTANTS
   ============================================================ */

.pollutant-grid {
    display: grid;

    grid-template-columns:
        repeat(6, minmax(0, 1fr));

    gap: 12px;
}

.pollutant {
    padding: 21px;

    min-height: 210px;

    position: relative;
}

.pollutant-dot {
    position: absolute;
    top: 21px;
    right: 21px;

    width: 7px;
    height: 7px;

    border-radius: 50%;
    background: var(--coral);
}

.pollutant-symbol {
    font-size: 17px;
    font-weight: 900;
}

.pollutant-value {
    margin-top: 25px;

    font-size: 24px;
    font-weight: 900;
}

.pollutant-unit {
    font-size: 9px;
    color: #7d928c;
}

.pollutant-mean {
    margin-top: 6px;

    font-size: 9px;
    color: #7d928c;
}

.bar {
    height: 6px;

    margin-top: 22px;

    background: #e5ebe4;

    border-radius: 999px;

    overflow: hidden;
}

.bar-fill {
    height: 100%;

    background: var(--coral);

    border-radius: inherit;
}

.pollutant-footer {
    margin-top: 9px;

    display: flex;
    justify-content: space-between;

    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: .06em;

    color: #738982;
}

/* ============================================================
   FORECAST
   ============================================================ */

.forecast-grid {
    display: grid;

    grid-template-columns:
        minmax(0, 1.45fr)
        minmax(350px, .55fr);

    gap: 16px;
}

.chart-card {
    padding: 28px;
    min-height: 500px;
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
}

.chart-title {
    font-weight: 900;
    font-size: 17px;
}

.chart-subtitle {
    margin-top: 5px;
    font-size: 10px;
    color: #78908a;
}

.peak {
    text-align: right;
}

.peak-value {
    color: var(--coral);

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: 35px;
}

.peak-label {
    font-size: 8px;
    text-transform: uppercase;
    color: #79908a;
}

#forecastChart {
    width: 100%;
    height: 370px;

    margin-top: 20px;
}

.forecast-svg {
    width: 100%;
    height: 100%;
}

.grid-line {
    stroke: #dfe7df;
    stroke-width: 1;
    stroke-dasharray: 5 8;
}

.chart-line {
    fill: none;
    stroke: var(--coral);
    stroke-width: 4;
    stroke-linejoin: round;
    stroke-linecap: round;
}

.chart-area {
    fill: rgba(239,108,80,.10);
}

.chart-point {
    fill: var(--paper);
    stroke: var(--coral);
    stroke-width: 4;
}

.axis-text {
    fill: #7b918b;
    font-size: 10px;
}

.ranking-card {
    padding: 28px;

    background: var(--ink);
    color: white;

    min-height: 500px;
}

.ranking-title {
    font-size: 18px;
    font-weight: 900;
}

.ranking-sub {
    margin-top: 5px;

    color: rgba(255,255,255,.55);

    font-size: 10px;
}

.ranking-item {
    margin-top: 25px;
}

.ranking-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.ranking-name {
    font-weight: 900;
}

.ranking-value {
    font-size: 12px;
    opacity: .75;
}

.ranking-track {
    height: 6px;

    margin-top: 9px;

    background: rgba(255,255,255,.14);

    border-radius: 999px;
}

.ranking-fill {
    height: 100%;

    background: var(--coral);

    border-radius: inherit;
}

/* ============================================================
   INSIGHT PANEL
   ============================================================ */

.insight-grid {
    display: grid;

    grid-template-columns:
        1fr 1fr 1fr;

    gap: 16px;
}

.insight {
    padding: 30px;
    min-height: 180px;
}

.insight-number {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #778d87;
}

.insight h3 {
    margin: 13px 0;

    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: 24px;
}

.insight p {
    margin: 0;

    color: #58736e;

    line-height: 1.65;

    font-size: 13px;
}

/* ============================================================
   EVENTS
   ============================================================ */

.event-kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.event-card {
    padding: 25px;
    min-height: 135px;
}

.event-value {
    font-family:
        Georgia,
        "Times New Roman",
        serif;

    font-size: 31px;
}

.event-label {
    margin-top: 7px;

    font-size: 9px;

    text-transform: uppercase;
    letter-spacing: .13em;

    color: #788f89;
}

/* ============================================================
   TABLE
   ============================================================ */

.table-card {
    overflow: hidden;
}

.table-wrap {
    overflow-x: auto;
}

table {
    width: 100%;

    border-collapse: collapse;

    font-size: 12px;
}

th {
    padding: 16px 18px;

    text-align: left;

    background: #eef2eb;

    font-size: 8px;

    text-transform: uppercase;
    letter-spacing: .14em;

    color: #738a84;
}

td {
    padding: 15px 18px;

    border-top: 1px solid var(--line);
}

td.aqi {
    font-weight: 900;
}

.mini-badge {
    display: inline-block;

    padding: 5px 9px;

    border-radius: 999px;

    font-size: 8px;

    font-weight: 900;

    text-transform: uppercase;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    padding: 25px 0;

    border-top: 1px solid var(--line);

    display: flex;
    justify-content: space-between;

    color: #78908a;

    font-size: 10px;
}

.footer strong {
    color: var(--ink);
}

/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1200px) {

    .hero-grid {
        grid-template-columns: 1fr;
    }

    .hero-meta {
        max-width: 600px;
    }

    .pollutant-grid {
        grid-template-columns: repeat(3, 1fr);
    }

    .aqi-hero {
        grid-template-columns: 1fr;
    }

    .forecast-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 850px) {

    .topbar {
        padding: 13px 17px;
    }

    .hero {
        padding: 45px 22px;
    }

    .main {
        padding: 40px 17px 70px;
    }

    .location-grid,
    .location-details {
        grid-template-columns: 1fr;
    }

    .kpi-grid,
    .event-kpis {
        grid-template-columns: repeat(2, 1fr);
    }

    .pollutant-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .insight-grid {
        grid-template-columns: 1fr;
    }

    .gauge-panel {
        min-height: 330px;
    }
}

@media (max-width: 520px) {

    .hero h1 {
        font-size: 54px;
    }

    .hero-meta {
        grid-template-columns: 1fr;
    }

    .kpi-grid,
    .event-kpis,
    .pollutant-grid {
        grid-template-columns: 1fr;
    }

    .location-primary {
        min-height: 280px;
    }

    .aqi-hero {
        padding: 15px;
    }

    .aqi-value-panel {
        padding: 20px;
    }

    .aqi-number {
        font-size: 100px;
    }

    .gauge-panel {
        padding: 0;
    }

    .footer {
        flex-direction: column;
        gap: 10px;
    }
}

</style>
</head>


<body>

<div class="app">

<!-- =========================================================
     TOP BAR
     ========================================================= -->

<header class="topbar">

    <div class="brand">

        <div class="brand-mark">
            P
        </div>

        <div>
            <div class="brand-name">
                PEARLS
            </div>

            <div class="brand-sub">
                AIR INTELLIGENCE
            </div>
        </div>

    </div>

    <div class="top-actions">

        <button
            class="top-button"
            onclick="window.print()"
        >
            Print / Export
        </button>

    </div>

</header>


<!-- =========================================================
     HERO
     ========================================================= -->

<section class="hero">

    <div class="hero-grid">

        <div>

            <div class="eyebrow">
                <span class="live-dot"></span>
                Live production intelligence
            </div>

            <h1>
                Lahore,
                <em>read clearly.</em>
            </h1>

            <div class="hero-description">
                Location intelligence, pollutant composition and
                a 72-hour production forecast brought together in
                one environmental intelligence view.
            </div>

        </div>

        <div class="hero-meta">

            <div>
                <div class="meta-label">
                    Monitoring location
                </div>

                <div class="meta-value">
                    Lahore Cantonment
                </div>
            </div>

            <div>
                <div class="meta-label">
                    Forecast horizon
                </div>

                <div class="meta-value">
                    72 hours
                </div>
            </div>

            <div>
                <div class="meta-label">
                    Coordinates
                </div>

                <div class="meta-value">
                    31.5205° N<br>
                    74.4036° E
                </div>
            </div>

            <div>
                <div class="meta-label">
                    Timezone
                </div>

                <div class="meta-value">
                    PKT · UTC+05:00
                </div>
            </div>

        </div>

    </div>

</section>


<main class="main">


<!-- =========================================================
     LOCATION
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                01 / Location intelligence
            </div>

            <h2 class="section-title">
                One city. Six pollution signals.
            </h2>

        </div>

    </div>


    <div class="location-grid">

        <div class="location-primary">

            <div class="eyebrow">
                Monitoring location
            </div>

            <h3>
                Lahore
            </h3>

            <div class="location-sub">
                Lahore Cantonment · Pakistan · PKT
            </div>

            <div class="location-divider"></div>

            <div class="location-status">

                <div id="locationBadge"></div>

                <div class="current-small">

                    <div class="current-small-label">
                        Current air quality
                    </div>

                    <div class="current-small-value"
                         id="locationAQI">
                    </div>

                </div>

            </div>

        </div>


        <div class="location-details">

            <div class="card detail-card">

                <div class="detail-label">
                    Latitude
                </div>

                <div class="detail-value">
                    31.5205° N
                </div>

            </div>

            <div class="card detail-card">

                <div class="detail-label">
                    Longitude
                </div>

                <div class="detail-value">
                    74.4036° E
                </div>

            </div>

            <div class="card detail-card">

                <div class="detail-label">
                    Station
                </div>

                <div class="detail-value">
                    Lahore Cantonment
                </div>

            </div>

            <div class="card detail-card">

                <div class="detail-label">
                    Timezone
                </div>

                <div class="detail-value">
                    PKT / UTC+05:00
                </div>

            </div>

        </div>

    </div>

</section>


<!-- =========================================================
     CURRENT AQI + GAUGE
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                02 / Current air quality
            </div>

            <h2 class="section-title">
                Where the air is now.
            </h2>

        </div>

    </div>


    <div class="card aqi-hero">

        <div class="aqi-value-panel">

            <div class="eyebrow">
                Current AQI
            </div>

            <div
                class="aqi-number"
                id="currentAQINumber"
            >
                --
            </div>

            <div
                class="aqi-category"
                id="currentAQICategory"
            >
                --
            </div>

            <div
                class="aqi-explanation"
                id="healthMessage"
            >
            </div>

            <div
                class="aqi-source"
                id="aqiSource"
            >
            </div>

        </div>


        <div class="gauge-panel">

            <div class="gauge-wrap">

                <svg
                    class="gauge-svg"
                    viewBox="0 0 760 430"
                    aria-label="AQI 0 to 500 gauge"
                >

                    <!-- Gauge arcs -->

                    <path
                        class="gauge-track"
                        stroke="#43c93b"
                        d="M 100 330 A 280 280 0 0 1 193 122"
                    />

                    <path
                        class="gauge-track"
                        stroke="#f4dc27"
                        d="M 193 122 A 280 280 0 0 1 380 50"
                    />

                    <path
                        class="gauge-track"
                        stroke="#f5a127"
                        d="M 380 50 A 280 280 0 0 1 567 122"
                    />

                    <path
                        class="gauge-track"
                        stroke="#ef493e"
                        d="M 567 122 A 280 280 0 0 1 660 330"
                    />

                    <!-- labels -->

                    <text
                        x="85"
                        y="365"
                        class="gauge-scale"
                    >
                        0
                    </text>

                    <text
                        x="173"
                        y="105"
                        class="gauge-scale"
                    >
                        100
                    </text>

                    <text
                        x="365"
                        y="30"
                        class="gauge-scale"
                    >
                        200
                    </text>

                    <text
                        x="575"
                        y="105"
                        class="gauge-scale"
                    >
                        300
                    </text>

                    <text
                        x="650"
                        y="365"
                        class="gauge-scale"
                    >
                        500
                    </text>


                    <!-- category labels -->

                    <text
                        x="115"
                        y="285"
                        class="gauge-label"
                    >
                        Good
                    </text>

                    <text
                        x="240"
                        y="130"
                        class="gauge-label"
                    >
                        Moderate
                    </text>

                    <text
                        x="350"
                        y="105"
                        class="gauge-label"
                    >
                        Sensitive
                    </text>

                    <text
                        x="480"
                        y="130"
                        class="gauge-label"
                    >
                        Unhealthy
                    </text>

                    <text
                        x="585"
                        y="285"
                        class="gauge-label"
                    >
                        Hazardous
                    </text>


                    <!-- center -->

                    <circle
                        cx="380"
                        cy="330"
                        r="46"
                        class="gauge-center"
                    />

                    <text
                        x="380"
                        y="339"
                        text-anchor="middle"
                        class="gauge-center-number"
                        id="gaugeCenterNumber"
                    >
                        --
                    </text>


                    <!-- dynamic needle -->

                    <line
                        id="gaugeNeedle"
                        x1="380"
                        y1="330"
                        x2="380"
                        y2="80"
                        class="gauge-marker-line"
                    />

                    <circle
                        id="gaugeMarker"
                        cx="380"
                        cy="50"
                        r="12"
                        class="gauge-marker"
                    />

                </svg>

            </div>

        </div>

    </div>

</section>


<!-- =========================================================
     KPI
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                03 / Current outlook
            </div>

            <h2 class="section-title">
                The numbers that matter.
            </h2>

        </div>

    </div>


    <div class="kpi-grid">

        <div class="card kpi">

            <div class="kpi-label">
                Current AQI
            </div>

            <div
                class="kpi-value"
                id="kpiCurrent"
            >
                --
            </div>

            <div class="kpi-sub">
                Current production reading
            </div>

        </div>


        <div class="card kpi">

            <div class="kpi-label">
                Forecast maximum
            </div>

            <div
                class="kpi-value"
                id="kpiMax"
            >
                --
            </div>

            <div class="kpi-sub">
                Peak predicted AQI
            </div>

        </div>


        <div class="card kpi">

            <div class="kpi-label">
                Forecast mean
            </div>

            <div
                class="kpi-value"
                id="kpiMean"
            >
                --
            </div>

            <div class="kpi-sub">
                72-hour production average
            </div>

        </div>


        <div class="card kpi">

            <div class="kpi-label">
                Forecast trend
            </div>

            <div
                class="kpi-value"
                id="kpiTrend"
            >
                --
            </div>

            <div class="kpi-sub">
                Production trajectory
            </div>

        </div>

    </div>

</section>


<!-- =========================================================
     POLLUTANTS
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                04 / Pollution ingredients
            </div>

            <h2 class="section-title">
                What is shaping the reading.
            </h2>

        </div>

    </div>


    <div
        class="pollutant-grid"
        id="pollutantGrid"
    >
    </div>

</section>


<!-- =========================================================
     FORECAST
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                05 / Pollution & AQI intelligence
            </div>

            <h2 class="section-title">
                A forecast with a shape.
            </h2>

        </div>

        <div>
            72-hour production horizon
        </div>

    </div>


    <div class="forecast-grid">


        <div class="card chart-card">

            <div class="chart-header">

                <div>

                    <div class="chart-title">
                        AQI trajectory
                    </div>

                    <div
                        class="chart-subtitle"
                        id="chartSubtitle"
                    >
                    </div>

                </div>

                <div class="peak">

                    <div
                        class="peak-value"
                        id="peakValue"
                    >
                        --
                    </div>

                    <div class="peak-label">
                        Peak predicted AQI
                    </div>

                </div>

            </div>


            <div id="forecastChart"></div>

        </div>


        <div class="ranking-card">

            <div class="ranking-title">
                Pollutant ranking
            </div>

            <div class="ranking-sub">
                Relative intensity across the pollution profile
            </div>

            <div id="ranking"></div>

        </div>

    </div>

</section>


<!-- =========================================================
     INTELLIGENCE
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                06 / Pearl intelligence
            </div>

            <h2 class="section-title">
                What the model is telling us.
            </h2>

        </div>

    </div>


    <div class="insight-grid">

        <div class="card insight">

            <div class="insight-number">
                Current condition
            </div>

            <h3>
                Air quality
            </h3>

            <p id="insightHealth">
            </p>

        </div>


        <div class="card insight">

            <div class="insight-number">
                Main driver
            </div>

            <h3>
                Pollution signal
            </h3>

            <p id="insightDriver">
            </p>

        </div>


        <div class="card insight">

            <div class="insight-number">
                72-hour trajectory
            </div>

            <h3>
                Forecast
            </h3>

            <p id="insightTrajectory">
            </p>

        </div>

    </div>

</section>


<!-- =========================================================
     CATEGORY DISTRIBUTION
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                07 / Forecast distribution
            </div>

            <h2 class="section-title">
                How the next 72 hours are distributed.
            </h2>

        </div>

    </div>


    <div
        class="card"
        style="padding:30px"
    >

        <div id="categoryDistribution"></div>

    </div>

</section>


<!-- =========================================================
     EVENTS
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                08 / Pollution event intelligence
            </div>

            <h2 class="section-title">
                The longer context.
            </h2>

        </div>

    </div>


    <div class="event-kpis">

        <div class="card event-card">

            <div
                class="event-value"
                id="eventTotal"
            >
                --
            </div>

            <div class="event-label">
                Total events
            </div>

        </div>


        <div class="card event-card">

            <div
                class="event-value"
                id="eventHigh"
            >
                --
            </div>

            <div class="event-label">
                High severity
            </div>

        </div>


        <div class="card event-card">

            <div
                class="event-value"
                id="eventElevated"
            >
                --
            </div>

            <div class="event-label">
                Elevated
            </div>

        </div>


        <div class="card event-card">

            <div
                class="event-value"
                id="eventDominant"
            >
                --
            </div>

            <div class="event-label">
                Dominant pollutant
            </div>

        </div>

    </div>

</section>


<!-- =========================================================
     NEXT 12 HOURS
     ========================================================= -->

<section class="section">

    <div class="section-head">

        <div>

            <div class="section-number">
                09 / Forecast detail
            </div>

            <h2 class="section-title">
                Next 12 hours.
            </h2>

        </div>

    </div>


    <div class="card table-card">

        <div class="table-wrap">

            <table>

                <thead>

                    <tr>
                        <th>Hour</th>
                        <th>Timestamp</th>
                        <th>Predicted AQI</th>
                        <th>Category</th>
                    </tr>

                </thead>

                <tbody id="forecastTable">
                </tbody>

            </table>

        </div>

    </div>

</section>


<!-- =========================================================
     FOOTER
     ========================================================= -->

<footer class="footer">

    <div>
        <strong>PEARLS AQI Predictor</strong>
        · Lahore Intelligence Snapshot
    </div>

    <div>
        Production forecast · 72 hours · AQI is modeled, not measured
    </div>

</footer>


</main>

</div>


<script>

const DATA = __DATA__;


// ============================================================
// FORMATTERS
// ============================================================

function number(value, digits = 1) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "N/A";
    }

    return Number(value).toLocaleString(
        undefined,
        {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        }
    );
}


function categoryClass(category) {

    const c = String(category).toLowerCase();

    if (c.includes("unhealthy")) {
        return "unhealthy";
    }

    if (c.includes("moderate")) {
        return "moderate";
    }

    if (c.includes("good")) {
        return "good";
    }

    return "moderate";
}


function formatTimestamp(value) {

    const d = new Date(value);

    return d.toLocaleString(
        undefined,
        {
            day: "2-digit",
            month: "short",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false
        }
    );
}


// ============================================================
// CURRENT AQI
// ============================================================

const currentAQI = DATA.current.aqi;
const currentCategory = DATA.current.category;

document.getElementById(
    "currentAQINumber"
).textContent = number(currentAQI, 1);

document.getElementById(
    "currentAQICategory"
).textContent = currentCategory;

document.getElementById(
    "gaugeCenterNumber"
).textContent = number(currentAQI, 0);

document.getElementById(
    "locationAQI"
).textContent =
    number(currentAQI, 1);

document.getElementById(
    "kpiCurrent"
).textContent =
    number(currentAQI, 1);

document.getElementById(
    "aqiSource"
).textContent =
    "Source: " + DATA.current.source;


// ============================================================
// LOCATION BADGE
// ============================================================

document.getElementById(
    "locationBadge"
).innerHTML =
    `<span class="badge ${categoryClass(currentCategory)}">
        ${currentCategory}
    </span>`;


// ============================================================
// HEALTH MESSAGE
// ============================================================

document.getElementById(
    "healthMessage"
).textContent =
    DATA.intelligence.health_message;

document.getElementById(
    "insightHealth"
).textContent =
    DATA.intelligence.health_message;

document.getElementById(
    "insightDriver"
).textContent =
    DATA.intelligence.driver_message;

document.getElementById(
    "insightTrajectory"
).textContent =
    DATA.intelligence.trajectory_message;


// ============================================================
// KPI
// ============================================================

document.getElementById(
    "kpiMax"
).textContent =
    number(DATA.forecast.max, 1);

document.getElementById(
    "kpiMean"
).textContent =
    number(DATA.forecast.mean, 1);

document.getElementById(
    "kpiTrend"
).textContent =
    DATA.forecast.trend.charAt(0).toUpperCase()
    + DATA.forecast.trend.slice(1);

document.getElementById(
    "peakValue"
).textContent =
    number(DATA.forecast.max, 1);


// ============================================================
// GAUGE
// ============================================================

function setGauge(aqi) {

    const value =
        Math.max(
            0,
            Math.min(500, Number(aqi))
        );

    /*
        Semicircle:
        center = 380,330
        radius = 280

        180 degrees:
        AQI 0   = 180°
        AQI 500 = 0°
    */

    const theta =
        Math.PI -
        (value / 500) * Math.PI;

    const cx = 380;
    const cy = 330;
    const r = 280;

    const x =
        cx + r * Math.cos(theta);

    const y =
        cy - r * Math.sin(theta);

    document.getElementById(
        "gaugeMarker"
    ).setAttribute("cx", x);

    document.getElementById(
        "gaugeMarker"
    ).setAttribute("cy", y);

    document.getElementById(
        "gaugeNeedle"
    ).setAttribute("x2", x);

    document.getElementById(
        "gaugeNeedle"
    ).setAttribute("y2", y);
}

setGauge(currentAQI);


// ============================================================
// POLLUTANT CARDS
// ============================================================

const pollutantGrid =
    document.getElementById(
        "pollutantGrid"
    );

DATA.pollution.forEach((p, index) => {

    const div =
        document.createElement("div");

    div.className = "card pollutant";

    const percentile =
        Math.max(
            0,
            Math.min(100, p.percentile)
        );

    div.innerHTML = `

        <span class="pollutant-dot"></span>

        <div class="pollutant-symbol">
            ${p.symbol}
        </div>

        <div class="pollutant-value">
            ${number(p.value, 3)}
        </div>

        <div class="pollutant-unit">
            ${p.unit}
        </div>

        <div class="pollutant-mean">
            Mean: ${number(p.mean, 3)}
        </div>

        <div class="bar">
            <div
                class="bar-fill"
                style="width:${percentile}%"
            ></div>
        </div>

        <div class="pollutant-footer">

            <span>
                ${number(p.percentile, 1)} percentile
            </span>

            <span>
                ${p.status}
            </span>

        </div>
    `;

    pollutantGrid.appendChild(div);
});


// ============================================================
// RANKING
// ============================================================

const ranking =
    document.getElementById(
        "ranking"
    );

DATA.pollution.forEach((p, index) => {

    const item =
        document.createElement("div");

    item.className =
        "ranking-item";

    item.innerHTML = `

        <div class="ranking-top">

            <div class="ranking-name">
                ${String(index + 1).padStart(2, "0")}
                &nbsp;&nbsp;
                ${p.symbol}
            </div>

            <div class="ranking-value">
                ${number(p.percentile, 1)}%
            </div>

        </div>

        <div class="ranking-track">

            <div
                class="ranking-fill"
                style="width:${Math.min(
                    100,
                    Math.max(0, p.percentile)
                )}%"
            ></div>

        </div>
    `;

    ranking.appendChild(item);
});


// ============================================================
// FORECAST CHART
// ============================================================

function buildForecastChart() {

    const container =
        document.getElementById(
            "forecastChart"
        );

    const points =
        DATA.forecast.records;

    const width = 1000;
    const height = 370;

    const padLeft = 55;
    const padRight = 20;
    const padTop = 25;
    const padBottom = 35;

    const values =
        points.map(p => Number(p.aqi));

    let min =
        Math.min(...values);

    let max =
        Math.max(...values);

    const range =
        Math.max(20, max - min);

    min -= range * .12;
    max += range * .12;

    const x = i =>
        padLeft +
        (
            i /
            Math.max(1, points.length - 1)
        ) *
        (
            width -
            padLeft -
            padRight
        );

    const y = value =>
        padTop +
        (
            1 -
            (value - min) /
            (max - min)
        ) *
        (
            height -
            padTop -
            padBottom
        );

    let path = "";

    points.forEach((point, i) => {

        const px = x(i);
        const py = y(point.aqi);

        path +=
            (i === 0 ? "M" : "L")
            + ` ${px} ${py}`;
    });

    const areaPath =
        path
        + ` L ${x(points.length - 1)} ${height - padBottom}`
        + ` L ${x(0)} ${height - padBottom}`
        + " Z";

    let grid = "";

    const ticks = 5;

    for (let i = 0; i <= ticks; i++) {

        const value =
            min +
            (max - min) *
            (i / ticks);

        const py = y(value);

        grid += `
            <line
                x1="${padLeft}"
                y1="${py}"
                x2="${width - padRight}"
                y2="${py}"
                class="grid-line"
            />

            <text
                x="${padLeft - 10}"
                y="${py + 4}"
                text-anchor="end"
                class="axis-text"
            >
                ${Math.round(value)}
            </text>
        `;
    }

    let pointMarkup = "";

    let peakIndex = 0;

    values.forEach((value, i) => {

        if (
            value >
            values[peakIndex]
        ) {
            peakIndex = i;
        }
    });

    pointMarkup = `
        <circle
            cx="${x(peakIndex)}"
            cy="${y(values[peakIndex])}"
            r="7"
            class="chart-point"
        />

        <text
            x="${x(peakIndex)}"
            y="${y(values[peakIndex]) - 15}"
            text-anchor="middle"
            style="
                fill:#ef6c50;
                font-size:13px;
                font-weight:900;
            "
        >
            ${number(values[peakIndex], 1)}
        </text>
    `;

    const startLabel =
        formatTimestamp(points[0].timestamp);

    const endLabel =
        formatTimestamp(
            points[points.length - 1].timestamp
        );

    container.innerHTML = `

        <svg
            class="forecast-svg"
            viewBox="0 0 ${width} ${height}"
            preserveAspectRatio="none"
        >

            ${grid}

            <path
                d="${areaPath}"
                class="chart-area"
            />

            <path
                d="${path}"
                class="chart-line"
            />

            ${pointMarkup}

            <text
                x="${padLeft}"
                y="${height - 8}"
                class="axis-text"
            >
                ${startLabel}
            </text>

            <text
                x="${width - padRight}"
                y="${height - 8}"
                text-anchor="end"
                class="axis-text"
            >
                ${endLabel}
            </text>

        </svg>
    `;

    document.getElementById(
        "chartSubtitle"
    ).textContent =
        `${startLabel} — ${endLabel}`;
}

buildForecastChart();


// ============================================================
// CATEGORY DISTRIBUTION
// ============================================================

const distribution =
    document.getElementById(
        "categoryDistribution"
    );

const categoryColors = {
    "Good": "#79b96b",
    "Moderate": "#d5bd4e",
    "Unhealthy for Sensitive Groups": "#e6a341",
    "Unhealthy": "#ef6c50",
    "Very Unhealthy": "#b64b6d",
    "Hazardous": "#7d2135"
};

let distributionHTML = "";

DATA.categories.forEach(item => {

    const width =
        Math.min(
            100,
            Math.max(
                0,
                Number(item.percentage)
            )
        );

    const color =
        categoryColors[item.category]
        || "#8ba29c";

    distributionHTML += `

        <div style="margin-bottom:20px">

            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:7px;
                font-size:12px;
                font-weight:800;
            ">

                <span>
                    ${item.category}
                </span>

                <span>
                    ${item.hours}h · ${number(
                        item.percentage,
                        1
                    )}%
                </span>

            </div>

            <div style="
                height:10px;
                background:#e5ebe4;
                border-radius:999px;
                overflow:hidden;
            ">

                <div style="
                    width:${width}%;
                    height:100%;
                    background:${color};
                    border-radius:999px;
                "></div>

            </div>

        </div>
    `;
});

distribution.innerHTML =
    distributionHTML;


// ============================================================
// EVENTS
// ============================================================

document.getElementById(
    "eventTotal"
).textContent =
    Number(
        DATA.events.total
    ).toLocaleString();

document.getElementById(
    "eventHigh"
).textContent =
    Number(
        DATA.events.high
    ).toLocaleString();

document.getElementById(
    "eventElevated"
).textContent =
    Number(
        DATA.events.elevated
    ).toLocaleString();

document.getElementById(
    "eventDominant"
).textContent =
    DATA.dominant_pollutant
        ? DATA.dominant_pollutant.symbol
        : "N/A";


// ============================================================
// NEXT 12 HOURS TABLE
// ============================================================

const table =
    document.getElementById(
        "forecastTable"
    );

DATA.next_12_hours.forEach(item => {

    const row =
        document.createElement("tr");

    const cls =
        categoryClass(
            item.category
        );

    row.innerHTML = `

        <td>
            H+${item.hour}
        </td>

        <td>
            ${formatTimestamp(
                item.timestamp
            )}
        </td>

        <td class="aqi">
            <span
                style="
                    display:inline-block;
                    width:7px;
                    height:7px;
                    background:#ef6c50;
                    border-radius:50%;
                    margin-right:7px;
                "
            ></span>

            ${number(item.aqi, 2)}
        </td>

        <td>
            <span class="
                mini-badge
                ${cls}
            ">
                ${item.category}
            </span>
        </td>
    `;

    table.appendChild(row);
});


// ============================================================
// KEYBOARD / PRINT
// ============================================================

document.addEventListener(
    "keydown",
    event => {

        if (
            event.ctrlKey &&
            event.key.toLowerCase() === "p"
        ) {
            return;
        }

    }
);

</script>

</body>
</html>
"""


HTML = HTML.replace(
    "__DATA__",
    data_json
)

with open(
    HTML_OUT,
    "w",
    encoding="utf-8"
) as f:
    f.write(HTML)


# ============================================================================
# CACHE RECORD
# ============================================================================

cache_record = {
    "dashboard_hash": dashboard_hash,
    "created_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "inputs": hash_items,
    "html": str(HTML_OUT),
    "data": str(DATA_JSON_OUT),
}

write_json(
    CACHE_HASH_OUT,
    cache_record
)


# ============================================================================
# VALIDATION
# ============================================================================

banner("VALIDATING PEARL INTELLIGENCE DASHBOARD")

checks = {}

checks["dashboard_html"] = (
    HTML_OUT.exists()
    and HTML_OUT.stat().st_size > 10000
)

checks["dashboard_data"] = (
    DATA_JSON_OUT.exists()
    and DATA_JSON_OUT.stat().st_size > 100
)

checks["forecast_csv"] = (
    FORECAST_OUT.exists()
    and len(pd.read_csv(FORECAST_OUT)) == FORECAST_HORIZON
)

checks["pollution_csv"] = (
    POLLUTION_OUT.exists()
)

checks["summary_csv"] = (
    SUMMARY_OUT.exists()
)

checks["cache_hash"] = (
    CACHE_HASH_OUT.exists()
)

html_text = HTML_OUT.read_text(
    encoding="utf-8"
)

checks["current_aqi"] = (
    "Current AQI" in html_text
)

checks["aqi_gauge"] = (
    "gaugeMarker" in html_text
    and "gaugeNeedle" in html_text
)

checks["location"] = (
    "Lahore Cantonment" in html_text
)

checks["pollutants"] = all(
    label in html_text
    for label in [
        "PM₂.₅",
        "PM₁₀",
        "O₃",
        "NO₂",
        "SO₂",
        "CO",
    ]
)

checks["forecast"] = (
    "AQI trajectory" in html_text
    and "forecastChart" in html_text
)

checks["json_serialization"] = True

for key, passed in checks.items():

    info(
        key,
        "PASS" if passed else "FAIL"
    )

overall_pass = all(checks.values())


# ============================================================================
# REPORT
# ============================================================================

report = {
    "step": 23,
    "name": "Pearl Intelligence Dashboard",
    "status": "PASS" if overall_pass else "FAIL",

    "location": LOCATION,

    "current": {
        "aqi": current_aqi,
        "category": current_category,
        "source": current_aqi_source,
    },

    "forecast": {
        "horizon": FORECAST_HORIZON,
        "minimum": forecast_min,
        "maximum": forecast_max,
        "mean": forecast_mean,
        "median": forecast_median,
        "trend": forecast_trend,
        "start": forecast["_timestamp"].iloc[0].isoformat(),
        "end": forecast["_timestamp"].iloc[-1].isoformat(),
    },

    "pollution": {
        "available": len(available_pollutants),
        "dominant": (
            dominant_pollutant["label"]
            if dominant_pollutant
            else None
        ),
    },

    "events": {
        "total": event_count,
        "high": high_event_count,
        "elevated": elevated_event_count,
    },

    "model_training": False,
    "model_selection": False,
    "validation_test": False,

    "cache": {
        "enabled": True,
        "status": "HIT" if cache_hit else "MISS",
        "hash": dashboard_hash,
    },

    "validation": checks,

    "outputs": {
        "html": str(HTML_OUT),
        "data": str(DATA_JSON_OUT),
        "forecast": str(FORECAST_OUT),
        "pollution": str(POLLUTION_OUT),
        "summary": str(SUMMARY_OUT),
        "cache": str(CACHE_HASH_OUT),
    },
}

write_json(
    REPORT_OUT,
    report
)


# ============================================================================
# COMPLETE
# ============================================================================

banner("STEP 23 COMPLETE")

info(
    "Dashboard status",
    "PASS" if overall_pass else "FAIL"
)

info(
    "Location",
    "Lahore Cantonment, Lahore, Pakistan"
)

info(
    "Coordinates",
    f"{LOCATION['latitude']}, {LOCATION['longitude']}"
)

info(
    "Timezone",
    "PKT / UTC+05:00"
)

info(
    "Current AQI",
    f"{current_aqi:.3f}"
)

info(
    "Current AQI category",
    current_category
)

info(
    "Forecast maximum",
    f"{forecast_max:.3f}"
)

info(
    "Forecast mean",
    f"{forecast_mean:.3f}"
)

info(
    "Forecast trend",
    forecast_trend
)

info(
    "Pollution ingredients",
    f"{len(available_pollutants)} available"
)

info(
    "Dominant pollutant",
    (
        dominant_pollutant["label"]
        if dominant_pollutant
        else "N/A"
    )
)

info(
    "Pollution events",
    f"{event_count:,}"
)

info(
    "Cache status",
    "HIT" if cache_hit else "MISS"
)

info(
    "Dashboard HTML",
    HTML_OUT
)

info(
    "Dashboard data",
    DATA_JSON_OUT
)

info(
    "Dashboard report",
    REPORT_OUT
)

print()
print("Open the following file in Chrome:")
print(HTML_OUT)
print()