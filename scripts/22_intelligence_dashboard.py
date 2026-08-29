"""
PEARLS AQI PREDICTOR
STEP 22 — INTELLIGENCE DASHBOARD

Purpose
-------
Build a production-facing AQI intelligence dashboard from:

    STEP 20:
        production_dashboard_forecast.csv

    STEP 21:
        location_intelligence.json
        pollution_intelligence.csv
        pollution_hourly.csv
        aqi_intelligence.csv
        pollution_events.csv

Features
--------
- Location intelligence
- AQI forecast intelligence
- Pollution ingredient intelligence
- Pollutant ranking
- Recent pollutant observations
- Pollution events
- AQI category distribution
- 72-hour forecast chart
- Pollutant comparison chart
- Pollutant trend chart
- Risk/event visualization
- Interactive HTML dashboard
- JSON data package
- CSV exports
- Persistent cache
- JSON-safe serialization

No model training.
No model selection.
No validation/test.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TARGET = "us_aqi"
FORECAST_HORIZON = 72

STEP20_DIR = BASE_DIR / "reports" / "production_dashboard_v2"
STEP21_DIR = BASE_DIR / "reports" / "location_pollution_intelligence"

OUTPUT_DIR = BASE_DIR / "reports" / "intelligence_dashboard"
CACHE_DIR = OUTPUT_DIR / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_FILE = STEP20_DIR / "production_dashboard_forecast.csv"

LOCATION_FILE = STEP21_DIR / "location_intelligence.json"
POLLUTION_INTELLIGENCE_FILE = STEP21_DIR / "pollution_intelligence.csv"
POLLUTION_HOURLY_FILE = STEP21_DIR / "pollution_hourly.csv"
AQI_INTELLIGENCE_FILE = STEP21_DIR / "aqi_intelligence.csv"
EVENTS_FILE = STEP21_DIR / "pollution_events.csv"

OUTPUT_HTML = OUTPUT_DIR / "intelligence_dashboard.html"
OUTPUT_DATA = OUTPUT_DIR / "intelligence_dashboard_data.json"
OUTPUT_FORECAST = OUTPUT_DIR / "intelligence_dashboard_forecast.csv"
OUTPUT_POLLUTION = OUTPUT_DIR / "intelligence_dashboard_pollution.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "intelligence_dashboard_summary.csv"
OUTPUT_REPORT = OUTPUT_DIR / "intelligence_dashboard_results.json"
CACHE_HASH_FILE = CACHE_DIR / "dashboard_hash.json"

ASSET_DIR = STEP20_DIR

POLLUTANT_ALIASES = {
    "PM2.5": [
        "pm2_5",
        "pm25",
        "pm2.5",
        "pm_2_5",
        "pm2_5_concentration",
    ],
    "PM10": [
        "pm10",
        "pm_10",
        "pm10_concentration",
    ],
    "O3": [
        "o3",
        "ozone",
        "ozone_concentration",
    ],
    "NO2": [
        "no2",
        "nitrogen_dioxide",
        "nitrogendioxide",
    ],
    "SO2": [
        "so2",
        "sulphur_dioxide",
        "sulfur_dioxide",
    ],
    "CO": [
        "co",
        "carbon_monoxide",
        "carbonmonoxide",
    ],
}


# ============================================================================
# PRINTING
# ============================================================================

def banner(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def item(label: str, value):
    print(f"{label:<32}: {value}")


# ============================================================================
# JSON SAFE SERIALIZATION
# ============================================================================

def json_safe(value):
    """
    Convert pandas / NumPy / datetime objects recursively into
    standard JSON-compatible Python objects.

    This explicitly fixes:
        TypeError: Object of type DataFrame is not JSON serializable
    """

    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        return [
            json_safe(record)
            for record in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return [
            json_safe(x)
            for x in value.tolist()
        ]

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (np.int64, np.int32)):
        return int(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(x) for x in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def write_json(path: Path, data):
    safe = json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            safe,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================================
# HELPERS
# ============================================================================

def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_timestamp_column(df: pd.DataFrame):
    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "forecast_timestamp",
        "forecast_time",
        "ds",
    ]

    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for col in df.columns:
        if "timestamp" in normalize_name(col):
            return col

    for col in df.columns:
        if "datetime" in normalize_name(col):
            return col

    return None


def find_column(df: pd.DataFrame, aliases):
    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]

    # relaxed matching
    for col in df.columns:
        n = normalize_name(col)
        for alias in aliases:
            a = normalize_name(alias)
            if a in n or n in a:
                return col

    return None


def safe_float(value, default=None):
    try:
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def percentile_rank(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return None

    if len(values) == 1:
        return 100.0

    return float(values.rank(pct=True).iloc[-1] * 100)


def aq_category(aqi):
    if aqi is None:
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


def category_class(category):
    mapping = {
        "Good": "good",
        "Moderate": "moderate",
        "Unhealthy for Sensitive Groups": "sensitive",
        "Unhealthy": "unhealthy",
        "Very Unhealthy": "very-unhealthy",
        "Hazardous": "hazardous",
    }

    return mapping.get(category, "unknown")


def calculate_trend(values):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if len(values) < 4:
        return "stable"

    n = min(12, len(values))

    first = float(values.iloc[:n].mean())
    last = float(values.iloc[-n:].mean())

    delta = last - first

    if abs(delta) < max(1.0, abs(first) * 0.03):
        return "stable"

    if delta > 0:
        return "rising"

    return "falling"


def make_hash(paths):
    sha = hashlib.sha256()

    for path in paths:
        sha.update(str(path).encode("utf-8"))

        if path.exists():
            sha.update(str(path.stat().st_size).encode("utf-8"))
            sha.update(str(path.stat().st_mtime_ns).encode("utf-8"))

            # Hash content for deterministic cache validation.
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(chunk)

    return sha.hexdigest()


# ============================================================================
# LOAD INPUTS
# ============================================================================

def load_inputs():
    banner("VERIFYING STEP 20 / STEP 21 INPUTS")

    required = [
        ("Step 20 forecast", FORECAST_FILE),
        ("Step 21 location", LOCATION_FILE),
        ("Step 21 pollution intelligence", POLLUTION_INTELLIGENCE_FILE),
        ("Step 21 pollution hourly", POLLUTION_HOURLY_FILE),
        ("Step 21 AQI intelligence", AQI_INTELLIGENCE_FILE),
        ("Step 21 events", EVENTS_FILE),
    ]

    for label, path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

        item(label, f"FOUND -> {path}")

    forecast = pd.read_csv(FORECAST_FILE)
    pollution = pd.read_csv(POLLUTION_INTELLIGENCE_FILE)
    pollution_hourly = pd.read_csv(POLLUTION_HOURLY_FILE)
    aqi = pd.read_csv(AQI_INTELLIGENCE_FILE)
    events = pd.read_csv(EVENTS_FILE)

    with LOCATION_FILE.open("r", encoding="utf-8") as f:
        location = json.load(f)

    return forecast, location, pollution, pollution_hourly, aqi, events


# ============================================================================
# LOCATION INTELLIGENCE
# ============================================================================

def build_location(location, pollution_hourly):
    banner("BUILDING LOCATION INTELLIGENCE")

    city = location.get("city")
    country = location.get("country")
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    timezone_value = location.get("timezone")
    station = location.get("station")

    if not city:
        city = location.get("location", "Unknown")

    if not country:
        country = "Unknown"

    # Recover location hints from metadata if available.
    if city == "Unknown":
        for col in pollution_hourly.columns:
            n = normalize_name(col)

            if n in {"city", "location_city"}:
                values = pollution_hourly[col].dropna()

                if not values.empty:
                    city = str(values.iloc[-1])
                    break

    item("City", city)
    item("Country", country)
    item("Latitude", latitude if latitude is not None else "NOT AVAILABLE")
    item("Longitude", longitude if longitude is not None else "NOT AVAILABLE")
    item(
        "Timezone",
        timezone_value if timezone_value is not None else "NOT AVAILABLE",
    )
    item(
        "Station",
        station if station is not None else "NOT AVAILABLE",
    )

    return {
        "city": city,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_value,
        "station": station,
        "source": location.get("source", "step_21"),
        "display_name": (
            f"{city}, {country}"
            if city and country
            else str(city or country)
        ),
    }


# ============================================================================
# POLLUTION INTELLIGENCE
# ============================================================================

def discover_pollutants(pollution_hourly):
    banner("DISCOVERING POLLUTION INGREDIENTS")

    discovered = {}

    for pollutant, aliases in POLLUTANT_ALIASES.items():
        col = find_column(pollution_hourly, aliases)

        discovered[pollutant] = col

        if col:
            print(f"{pollutant:<32}: FOUND -> {col}")
        else:
            print(f"{pollutant:<32}: NOT AVAILABLE")

    item(
        "Pollutants available",
        sum(v is not None for v in discovered.values()),
    )

    return discovered


def build_pollution_intelligence(
    pollution,
    pollution_hourly,
    discovered,
):
    banner("CALCULATING POLLUTION INTELLIGENCE")

    records = []

    for pollutant, column in discovered.items():
        if not column:
            continue

        series = pd.to_numeric(
            pollution_hourly[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        latest = float(series.iloc[-1])
        mean = float(series.mean())
        minimum = float(series.min())
        maximum = float(series.max())
        median = float(series.median())

        p90 = float(series.quantile(0.90))
        p95 = float(series.quantile(0.95))

        recent = series.tail(min(24, len(series)))

        recent_mean = float(recent.mean())

        if mean != 0:
            recent_change_pct = (
                (recent_mean - mean) / abs(mean) * 100
            )
        else:
            recent_change_pct = 0.0

        percentile = float(
            series.rank(pct=True).iloc[-1] * 100
        )

        record = {
            "pollutant": pollutant,
            "source_column": column,
            "latest": latest,
            "mean": mean,
            "median": median,
            "minimum": minimum,
            "maximum": maximum,
            "p90": p90,
            "p95": p95,
            "recent_24h_mean": recent_mean,
            "recent_change_pct": recent_change_pct,
            "percentile": percentile,
            "trend": calculate_trend(recent),
            "observations": int(len(series)),
        }

        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "pollutant",
                "latest",
                "mean",
                "median",
                "minimum",
                "maximum",
                "p90",
                "p95",
                "recent_24h_mean",
                "recent_change_pct",
                "percentile",
                "trend",
                "observations",
            ]
        )

    intelligence = pd.DataFrame(records)

    intelligence = intelligence.sort_values(
        "percentile",
        ascending=False,
    ).reset_index(drop=True)

    intelligence["rank"] = np.arange(
        1,
        len(intelligence) + 1,
    )

    dominant = intelligence.iloc[0]["pollutant"]

    item("Dominant relative pollutant", dominant)
    item(
        "Relative intensity",
        f"{intelligence.iloc[0]['percentile']:.2f} percentile",
    )

    return intelligence


# ============================================================================
# AQI INTELLIGENCE
# ============================================================================

def build_aqi_intelligence(forecast):
    banner("BUILDING AQI FORECAST INTELLIGENCE")

    timestamp_col = find_timestamp_column(forecast)

    if timestamp_col:
        timestamps = pd.to_datetime(
            forecast[timestamp_col],
            errors="coerce",
        )
    else:
        timestamps = pd.Series(
            pd.date_range(
                datetime.now(timezone.utc),
                periods=len(forecast),
                freq="h",
            )
        )

    prediction_col = None

    candidates = [
        "prediction",
        "predicted_aqi",
        "predicted_us_aqi",
        "forecast_aqi",
        "us_aqi",
        "aqi",
        "value",
    ]

    prediction_col = find_column(
        forecast,
        candidates,
    )

    if prediction_col is None:
        numeric_cols = forecast.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_cols:
            raise ValueError(
                "Could not identify AQI prediction column."
            )

        prediction_col = numeric_cols[-1]

    predictions = pd.to_numeric(
        forecast[prediction_col],
        errors="coerce",
    )

    aqi_df = pd.DataFrame({
        "timestamp": timestamps,
        "horizon": np.arange(1, len(forecast) + 1),
        "predicted_aqi": predictions,
    })

    aqi_df["category"] = aqi_df[
        "predicted_aqi"
    ].apply(aq_category)

    aqi_df["category_class"] = aqi_df[
        "category"
    ].apply(category_class)

    minimum = safe_float(predictions.min())
    maximum = safe_float(predictions.max())
    mean = safe_float(predictions.mean())
    median = safe_float(predictions.median())

    trend = calculate_trend(predictions)

    category_counts = (
        aqi_df["category"]
        .value_counts()
        .to_dict()
    )

    dominant_category = (
        max(
            category_counts,
            key=category_counts.get,
        )
        if category_counts
        else "Unknown"
    )

    item("Forecast minimum", f"{minimum:.3f}")
    item("Forecast maximum", f"{maximum:.3f}")
    item("Forecast mean", f"{mean:.3f}")
    item("Forecast median", f"{median:.3f}")
    item("Dominant AQI category", dominant_category)
    item("Forecast trend", trend)

    summary = {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "dominant_category": dominant_category,
        "trend": trend,
        "category_counts": category_counts,
        "prediction_column": prediction_col,
    }

    return aqi_df, summary


# ============================================================================
# EVENT INTELLIGENCE
# ============================================================================

def build_event_intelligence(events):
    banner("BUILDING POLLUTION EVENT INTELLIGENCE")

    if events.empty:
        return {
            "total_events": 0,
            "high_severity_events": 0,
            "elevated_events": 0,
            "pollutant_counts": {},
            "severity_counts": {},
            "recent_events": [],
        }

    pollutant_col = find_column(
        events,
        [
            "pollutant",
            "ingredient",
            "pollution_type",
        ],
    )

    severity_col = find_column(
        events,
        [
            "severity",
            "level",
            "risk_level",
        ],
    )

    if severity_col:
        severity_values = (
            events[severity_col]
            .astype(str)
            .str.lower()
        )
    else:
        severity_values = pd.Series(
            ["unknown"] * len(events)
        )

    high_mask = severity_values.str.contains(
        "high|severe|critical",
        regex=True,
    )

    elevated_mask = severity_values.str.contains(
        "elevated|moderate",
        regex=True,
    )

    pollutant_counts = {}

    if pollutant_col:
        pollutant_counts = (
            events[pollutant_col]
            .astype(str)
            .value_counts()
            .to_dict()
        )

    severity_counts = (
        severity_values.value_counts()
        .to_dict()
    )

    recent = events.tail(50).copy()

    total = len(events)
    high = int(high_mask.sum())
    elevated = int(elevated_mask.sum())

    item("Pollution events", total)
    item("High severity events", high)
    item("Elevated events", elevated)

    return {
        "total_events": int(total),
        "high_severity_events": high,
        "elevated_events": elevated,
        "pollutant_counts": pollutant_counts,
        "severity_counts": severity_counts,
        "recent_events": recent,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def build_dashboard_summary(
    location,
    pollution_intelligence,
    aqi_summary,
    event_summary,
):
    dominant_pollutant = (
        pollution_intelligence.iloc[0]["pollutant"]
        if not pollution_intelligence.empty
        else "Unknown"
    )

    dominant_percentile = (
        safe_float(
            pollution_intelligence.iloc[0]["percentile"]
        )
        if not pollution_intelligence.empty
        else None
    )

    return {
        "location": location["display_name"],
        "city": location["city"],
        "country": location["country"],
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_minimum": aqi_summary["minimum"],
        "forecast_maximum": aqi_summary["maximum"],
        "forecast_mean": aqi_summary["mean"],
        "forecast_median": aqi_summary["median"],
        "dominant_aqi_category": (
            aqi_summary["dominant_category"]
        ),
        "forecast_trend": aqi_summary["trend"],
        "dominant_pollutant": dominant_pollutant,
        "dominant_pollutant_percentile": dominant_percentile,
        "pollution_events": event_summary["total_events"],
        "high_severity_events": (
            event_summary["high_severity_events"]
        ),
    }


# ============================================================================
# HTML COMPONENTS
# ============================================================================

def html_escape(value):
    if value is None:
        return ""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_pollutant_cards(pollution_intelligence):
    cards = []

    for _, row in pollution_intelligence.iterrows():
        pollutant = html_escape(row["pollutant"])

        latest = safe_float(row["latest"], 0)
        mean = safe_float(row["mean"], 0)
        percentile = safe_float(row["percentile"], 0)
        trend = html_escape(row["trend"])

        cards.append(
            f"""
            <div class="pollutant-card">
                <div class="pollutant-title">{pollutant}</div>
                <div class="pollutant-value">{latest:.3f}</div>
                <div class="pollutant-meta">
                    Mean: {mean:.3f}
                </div>
                <div class="pollutant-bar">
                    <div style="width:{min(max(percentile,0),100):.1f}%"></div>
                </div>
                <div class="pollutant-footer">
                    <span>{percentile:.1f} percentile</span>
                    <span>{trend}</span>
                </div>
            </div>
            """
        )

    return "\n".join(cards)


def build_forecast_rows(aqi_df):
    rows = []

    for _, row in aqi_df.iterrows():
        timestamp = row["timestamp"]

        if isinstance(timestamp, pd.Timestamp):
            timestamp_text = timestamp.strftime(
                "%Y-%m-%d %H:%M"
            )
        else:
            timestamp_text = str(timestamp)

        value = safe_float(
            row["predicted_aqi"],
            0,
        )

        category = html_escape(
            row["category"]
        )

        rows.append(
            f"""
            <tr>
                <td>{int(row["horizon"])}</td>
                <td>{html_escape(timestamp_text)}</td>
                <td><strong>{value:.2f}</strong></td>
                <td>
                    <span class="badge {category_class(category)}">
                        {category}
                    </span>
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_event_rows(events):
    if events.empty:
        return """
        <tr>
            <td colspan="5">No pollution events available.</td>
        </tr>
        """

    rows = []

    for _, row in events.tail(50).iloc[::-1].iterrows():
        values = list(row.values)

        timestamp = ""
        pollutant = ""
        severity = ""
        value = ""
        message = ""

        for value_item in values:
            text = str(value_item)

            if not timestamp and (
                "202" in text
                or "T" in text
            ):
                timestamp = text

        pollutant_col = find_column(
            events,
            ["pollutant", "ingredient"],
        )

        severity_col = find_column(
            events,
            ["severity", "level", "risk_level"],
        )

        value_col = find_column(
            events,
            ["value", "concentration"],
        )

        message_col = find_column(
            events,
            ["message", "description", "reason"],
        )

        if pollutant_col:
            pollutant = str(
                row[pollutant_col]
            )

        if severity_col:
            severity = str(
                row[severity_col]
            )

        if value_col:
            value = str(
                row[value_col]
            )

        if message_col:
            message = str(
                row[message_col]
            )

        rows.append(
            f"""
            <tr>
                <td>{html_escape(timestamp)}</td>
                <td>{html_escape(pollutant)}</td>
                <td>{html_escape(severity)}</td>
                <td>{html_escape(value)}</td>
                <td>{html_escape(message)}</td>
            </tr>
            """
        )

    return "\n".join(rows)


# ============================================================================
# HTML DASHBOARD
# ============================================================================

def create_html(
    location,
    aqi_df,
    aqi_summary,
    pollution_intelligence,
    event_summary,
):
    banner("CREATING INTELLIGENCE DASHBOARD")

    forecast_json = json.dumps(
        json_safe(aqi_df),
        ensure_ascii=False,
    )

    pollutant_json = json.dumps(
        json_safe(pollution_intelligence),
        ensure_ascii=False,
    )

    category_json = json.dumps(
        json_safe(aqi_summary["category_counts"]),
        ensure_ascii=False,
    )

    event_pollutants_json = json.dumps(
        json_safe(
            event_summary["pollutant_counts"]
        ),
        ensure_ascii=False,
    )

    pollutant_cards = build_pollutant_cards(
        pollution_intelligence
    )

    forecast_rows = build_forecast_rows(
        aqi_df
    )

    recent_events = event_summary[
        "recent_events"
    ]

    event_rows = build_event_rows(
        recent_events
    )

    dominant_pollutant = (
        pollution_intelligence.iloc[0]["pollutant"]
        if not pollution_intelligence.empty
        else "Unknown"
    )

    dominant_category = (
        aqi_summary["dominant_category"]
    )

    trend = aqi_summary["trend"]

    category_class_name = category_class(
        dominant_category
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
PEARLS AQI Predictor — Intelligence Dashboard
</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #eef2ff 0,
            #f8fafc 38%,
            #eef2f7 100%
        );

    color: #172033;
}}

.header {{
    padding: 28px 36px;
    background: #111827;
    color: white;
}}

.header-inner {{
    max-width: 1500px;
    margin: auto;
}}

.header h1 {{
    margin: 0;
    font-size: 30px;
}}

.header p {{
    margin: 8px 0 0;
    opacity: 0.72;
}}

.container {{
    max-width: 1500px;
    margin: auto;
    padding: 28px;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 18px;
}}

.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 22px;
    box-shadow:
        0 8px 30px rgba(15,23,42,0.06);
}}

.kpi-label {{
    font-size: 13px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
}}

.kpi-value {{
    margin-top: 8px;
    font-size: 32px;
    font-weight: 750;
}}

.kpi-small {{
    margin-top: 5px;
    color: #64748b;
    font-size: 13px;
}}

.section {{
    margin-top: 24px;
}}

.section-title {{
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 14px;
}}

.location-card {{
    display: grid;
    grid-template-columns:
        1.4fr 1fr 1fr 1fr;
    gap: 18px;
}}

.location-main {{
    grid-row: span 2;
}}

.location-name {{
    font-size: 30px;
    font-weight: 800;
}}

.location-sub {{
    color: #64748b;
    margin-top: 5px;
}}

.location-value {{
    font-size: 20px;
    font-weight: 700;
    margin-top: 6px;
}}

.pollutant-grid {{
    display: grid;
    grid-template-columns:
        repeat(6, minmax(0, 1fr));
    gap: 14px;
}}

.pollutant-card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px;
}}

.pollutant-title {{
    font-weight: 800;
    font-size: 17px;
}}

.pollutant-value {{
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}}

.pollutant-meta {{
    font-size: 12px;
    color: #64748b;
    margin-top: 3px;
}}

.pollutant-bar {{
    height: 7px;
    margin-top: 13px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
}}

.pollutant-bar div {{
    height: 100%;
    background: #334155;
}}

.pollutant-footer {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 11px;
    color: #64748b;
    margin-top: 8px;
}}

.chart-grid {{
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 18px;
}}

.chart-box {{
    min-height: 380px;
}}

canvas {{
    width: 100% !important;
    height: 320px !important;
}}

.badge {{
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 750;
}}

.good {{
    background: #dcfce7;
    color: #166534;
}}

.moderate {{
    background: #fef9c3;
    color: #854d0e;
}}

.sensitive {{
    background: #ffedd5;
    color: #9a3412;
}}

.unhealthy {{
    background: #fee2e2;
    color: #991b1b;
}}

.very-unhealthy {{
    background: #f3e8ff;
    color: #6b21a8;
}}

.hazardous {{
    background: #e5e7eb;
    color: #111827;
}}

.unknown {{
    background: #e5e7eb;
    color: #475569;
}}

.alert {{
    border-left: 5px solid #ef4444;
}}

.table-wrapper {{
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

th {{
    text-align: left;
    padding: 12px;
    background: #f8fafc;
    color: #475569;
}}

td {{
    padding: 11px 12px;
    border-top: 1px solid #e5e7eb;
}}

.footer {{
    margin-top: 35px;
    padding: 25px 0;
    color: #64748b;
    font-size: 12px;
}}

@media(max-width:1100px) {{
    .grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .pollutant-grid {{
        grid-template-columns:
            repeat(3, minmax(0, 1fr));
    }}

    .chart-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media(max-width:700px) {{
    .container {{
        padding: 15px;
    }}

    .grid,
    .pollutant-grid,
    .location-card {{
        grid-template-columns: 1fr;
    }}

    .header {{
        padding: 22px;
    }}

    .kpi-value {{
        font-size: 26px;
    }}
}}

</style>
</head>

<body>

<div class="header">
    <div class="header-inner">
        <h1>PEARLS AQI Predictor</h1>
        <p>
            Location + Pollution Intelligence Dashboard
            · 72-hour production forecast
        </p>
    </div>
</div>

<div class="container">

    <!-- LOCATION -->

    <div class="section">
        <div class="section-title">
            Location Intelligence
        </div>

        <div class="card location-card">

            <div class="location-main">
                <div class="kpi-label">Monitoring location</div>

                <div class="location-name">
                    {html_escape(location["city"])}
                </div>

                <div class="location-sub">
                    {html_escape(location["country"])}
                </div>

                <div style="margin-top:18px">
                    <span class="badge {category_class_name}">
                        {html_escape(dominant_category)}
                    </span>
                </div>
            </div>

            <div>
                <div class="kpi-label">Latitude</div>
                <div class="location-value">
                    {html_escape(location["latitude"] or "N/A")}
                </div>
            </div>

            <div>
                <div class="kpi-label">Longitude</div>
                <div class="location-value">
                    {html_escape(location["longitude"] or "N/A")}
                </div>
            </div>

            <div>
                <div class="kpi-label">Station</div>
                <div class="location-value">
                    {html_escape(location["station"] or "N/A")}
                </div>
            </div>

            <div>
                <div class="kpi-label">Timezone</div>
                <div class="location-value">
                    {html_escape(location["timezone"] or "N/A")}
                </div>
            </div>

        </div>
    </div>


    <!-- KPI -->

    <div class="section">

        <div class="grid">

            <div class="card">
                <div class="kpi-label">
                    Forecast mean AQI
                </div>
                <div class="kpi-value">
                    {aqi_summary["mean"]:.1f}
                </div>
                <div class="kpi-small">
                    72-hour production forecast
                </div>
            </div>

            <div class="card">
                <div class="kpi-label">
                    Forecast maximum
                </div>
                <div class="kpi-value">
                    {aqi_summary["maximum"]:.1f}
                </div>
                <div class="kpi-small">
                    Peak predicted AQI
                </div>
            </div>

            <div class="card">
                <div class="kpi-label">
                    Dominant pollutant
                </div>
                <div class="kpi-value">
                    {html_escape(dominant_pollutant)}
                </div>
                <div class="kpi-small">
                    Highest relative intensity
                </div>
            </div>

            <div class="card">
                <div class="kpi-label">
                    Forecast trend
                </div>
                <div class="kpi-value">
                    {html_escape(trend.title())}
                </div>
                <div class="kpi-small">
                    Based on forecast trajectory
                </div>
            </div>

        </div>

    </div>


    <!-- POLLUTION INGREDIENTS -->

    <div class="section">

        <div class="section-title">
            Pollution Ingredients
        </div>

        <div class="pollutant-grid">
            {pollutant_cards}
        </div>

    </div>


    <!-- CHARTS -->

    <div class="section">

        <div class="section-title">
            Pollution & AQI Intelligence
        </div>

        <div class="chart-grid">

            <div class="card chart-box">
                <div class="section-title">
                    72-Hour AQI Forecast
                </div>
                <canvas id="aqiChart"></canvas>
            </div>

            <div class="card chart-box">
                <div class="section-title">
                    Pollution Ingredient Ranking
                </div>
                <canvas id="pollutantChart"></canvas>
            </div>

            <div class="card chart-box">
                <div class="section-title">
                    Pollutant Relative Intensity
                </div>
                <canvas id="intensityChart"></canvas>
            </div>

            <div class="card chart-box">
                <div class="section-title">
                    AQI Category Distribution
                </div>
                <canvas id="categoryChart"></canvas>
            </div>

        </div>

    </div>


    <!-- EVENTS -->

    <div class="section">

        <div class="section-title">
            Pollution Event Intelligence
        </div>

        <div class="grid">

            <div class="card">
                <div class="kpi-label">
                    Total events
                </div>
                <div class="kpi-value">
                    {event_summary["total_events"]}
                </div>
            </div>

            <div class="card alert">
                <div class="kpi-label">
                    High severity
                </div>
                <div class="kpi-value">
                    {event_summary["high_severity_events"]}
                </div>
            </div>

            <div class="card">
                <div class="kpi-label">
                    Elevated
                </div>
                <div class="kpi-value">
                    {event_summary["elevated_events"]}
                </div>
            </div>

            <div class="card">
                <div class="kpi-label">
                    Dominant pollutant
                </div>
                <div class="kpi-value">
                    {html_escape(dominant_pollutant)}
                </div>
            </div>

        </div>

    </div>


    <!-- FORECAST TABLE -->

    <div class="section">

        <div class="section-title">
            Forecast Detail
        </div>

        <div class="card table-wrapper">

            <table>

                <thead>
                    <tr>
                        <th>Horizon</th>
                        <th>Timestamp</th>
                        <th>Predicted AQI</th>
                        <th>Category</th>
                    </tr>
                </thead>

                <tbody>
                    {forecast_rows}
                </tbody>

            </table>

        </div>

    </div>


    <!-- EVENTS TABLE -->

    <div class="section">

        <div class="section-title">
            Recent Pollution Events
        </div>

        <div class="card table-wrapper">

            <table>

                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Pollutant</th>
                        <th>Severity</th>
                        <th>Value</th>
                        <th>Message</th>
                    </tr>
                </thead>

                <tbody>
                    {event_rows}
                </tbody>

            </table>

        </div>

    </div>


    <div class="footer">

        PEARLS AQI Predictor · Step 22 Intelligence Dashboard<br>

        Generated:
        {html_escape(generated_at)}<br>

        Model training:
        NOT PERFORMED ·
        Model selection:
        NOT PERFORMED ·
        Validation/test:
        NOT PERFORMED

    </div>

</div>


<script>

const forecast = {forecast_json};
const pollutants = {pollutant_json};
const categories = {category_json};
const eventPollutants = {event_pollutants_json};


const forecastLabels = forecast.map(
    x => x.timestamp
);


const forecastValues = forecast.map(
    x => x.predicted_aqi
);


new Chart(
    document.getElementById("aqiChart"),
    {{
        type: "line",

        data: {{
            labels: forecastLabels,

            datasets: [{{
                label: "Predicted AQI",
                data: forecastValues,
                tension: 0.25,
                fill: true
            }}]
        }},

        options: {{
            responsive: true,

            maintainAspectRatio: false,

            plugins: {{
                legend: {{
                    display: true
                }}
            }},

            scales: {{
                x: {{
                    ticks: {{
                        maxTicksLimit: 12
                    }}
                }},

                y: {{
                    beginAtZero: false
                }}
            }}
        }}
    }}
);


new Chart(
    document.getElementById("pollutantChart"),
    {{
        type: "bar",

        data: {{
            labels: pollutants.map(
                x => x.pollutant
            ),

            datasets: [{{
                label: "Relative percentile",
                data: pollutants.map(
                    x => x.percentile
                )
            }}]
        }},

        options: {{
            responsive: true,
            maintainAspectRatio: false,

            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 100
                }}
            }}
        }}
    }}
);


new Chart(
    document.getElementById("intensityChart"),
    {{
        type: "radar",

        data: {{
            labels: pollutants.map(
                x => x.pollutant
            ),

            datasets: [{{
                label: "Relative intensity",
                data: pollutants.map(
                    x => x.percentile
                )
            }}]
        }},

        options: {{
            responsive: true,
            maintainAspectRatio: false,

            scales: {{
                r: {{
                    beginAtZero: true,
                    max: 100
                }}
            }}
        }}
    }}
);


new Chart(
    document.getElementById("categoryChart"),
    {{
        type: "doughnut",

        data: {{
            labels: Object.keys(categories),

            datasets: [{{
                label: "AQI categories",
                data: Object.values(categories)
            }}]
        }},

        options: {{
            responsive: true,
            maintainAspectRatio: false
        }}
    }}
);

</script>

</body>
</html>
"""

    OUTPUT_HTML.write_text(
        html,
        encoding="utf-8",
    )

    return html


# ============================================================================
# CACHE
# ============================================================================

def check_cache(input_hash):
    if not CACHE_HASH_FILE.exists():
        return False

    try:
        with CACHE_HASH_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data.get("input_hash") == input_hash

    except Exception:
        return False


def save_cache(input_hash):
    write_json(
        CACHE_HASH_FILE,
        {
            "input_hash": input_hash,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dashboard(
    forecast,
    pollution_intelligence,
):
    banner("VALIDATING INTELLIGENCE DASHBOARD")

    checks = {}

    checks["dashboard_html"] = (
        OUTPUT_HTML.exists()
        and OUTPUT_HTML.stat().st_size > 1000
    )

    checks["dashboard_data"] = (
        OUTPUT_DATA.exists()
        and OUTPUT_DATA.stat().st_size > 100
    )

    checks["dashboard_forecast"] = (
        OUTPUT_FORECAST.exists()
        and len(pd.read_csv(OUTPUT_FORECAST))
        == len(forecast)
    )

    checks["dashboard_pollution"] = (
        OUTPUT_POLLUTION.exists()
    )

    checks["dashboard_summary"] = (
        OUTPUT_SUMMARY.exists()
    )

    checks["cache_hash"] = (
        CACHE_HASH_FILE.exists()
    )

    # Confirm JSON is genuinely parseable.
    try:
        with OUTPUT_DATA.open(
            "r",
            encoding="utf-8",
        ) as f:
            parsed = json.load(f)

        checks["json_serialization"] = (
            isinstance(parsed, dict)
        )
    except Exception:
        checks["json_serialization"] = False

    # Confirm HTML contains expected intelligence.
    if OUTPUT_HTML.exists():
        html = OUTPUT_HTML.read_text(
            encoding="utf-8"
        )

        checks["dashboard_content"] = all(
            text in html
            for text in [
                "Location Intelligence",
                "Pollution Ingredients",
                "72-Hour AQI Forecast",
                "Pollution Event Intelligence",
                "Forecast Detail",
            ]
        )
    else:
        checks["dashboard_content"] = False

    for name, passed in checks.items():
        item(
            name,
            "PASS" if passed else "FAIL",
        )

    overall = all(checks.values())

    item(
        "Overall validation",
        "PASS" if overall else "FAIL",
    )

    if not overall:
        raise RuntimeError(
            "Dashboard validation failed."
        )

    return checks


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    print()
    print("=" * 72)
    print("PEARLS AQI PREDICTOR")
    print("=" * 72)

    banner("STEP 22 — INTELLIGENCE DASHBOARD")

    item("Base directory", BASE_DIR)
    item("Target", TARGET)
    item("Forecast horizon", FORECAST_HORIZON)
    item(
        "Dashboard type",
        "Interactive + Infographic + Intelligence",
    )
    item("Caching", "Enabled")
    item("Model training", "NOT performed")
    item("Model selection", "NOT performed")
    item("Validation/test", "NOT performed")

    # ---------------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------------

    (
        forecast,
        raw_location,
        pollution,
        pollution_hourly,
        aqi_existing,
        events,
    ) = load_inputs()

    banner("LOADING PRODUCTION FORECAST")

    item("Forecast rows", len(forecast))

    timestamp_col = find_timestamp_column(
        forecast
    )

    if timestamp_col:
        forecast[timestamp_col] = pd.to_datetime(
            forecast[timestamp_col],
            errors="coerce",
        )

        item(
            "Forecast start",
            forecast[timestamp_col].min(),
        )

        item(
            "Forecast end",
            forecast[timestamp_col].max(),
        )

    # ---------------------------------------------------------------------
    # LOCATION
    # ---------------------------------------------------------------------

    location = build_location(
        raw_location,
        pollution_hourly,
    )

    # ---------------------------------------------------------------------
    # POLLUTANTS
    # ---------------------------------------------------------------------

    discovered = discover_pollutants(
        pollution_hourly
    )

    pollution_intelligence = (
        build_pollution_intelligence(
            pollution,
            pollution_hourly,
            discovered,
        )
    )

    # ---------------------------------------------------------------------
    # AQI
    # ---------------------------------------------------------------------

    aqi_df, aqi_summary = (
        build_aqi_intelligence(
            forecast
        )
    )

    # ---------------------------------------------------------------------
    # EVENTS
    # ---------------------------------------------------------------------

    event_summary = build_event_intelligence(
        events
    )

    # ---------------------------------------------------------------------
    # CACHE
    # ---------------------------------------------------------------------

    banner("CHECKING DASHBOARD CACHE")

    input_hash = make_hash(
        [
            FORECAST_FILE,
            LOCATION_FILE,
            POLLUTION_INTELLIGENCE_FILE,
            POLLUTION_HOURLY_FILE,
            AQI_INTELLIGENCE_FILE,
            EVENTS_FILE,
        ]
    )

    cache_hit = check_cache(input_hash)

    item(
        "Cache status",
        "HIT" if cache_hit else "MISS",
    )

    # ---------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------

    banner("BUILDING DASHBOARD PACKAGE")

    summary = build_dashboard_summary(
        location,
        pollution_intelligence,
        aqi_summary,
        event_summary,
    )

    # ---------------------------------------------------------------------
    # EXPORT FORECAST
    # ---------------------------------------------------------------------

    export_forecast = aqi_df.copy()

    export_forecast.to_csv(
        OUTPUT_FORECAST,
        index=False,
    )

    # ---------------------------------------------------------------------
    # EXPORT POLLUTION
    # ---------------------------------------------------------------------

    pollution_export = (
        pollution_intelligence.copy()
    )

    pollution_export.to_csv(
        OUTPUT_POLLUTION,
        index=False,
    )

    # ---------------------------------------------------------------------
    # EXPORT SUMMARY
    # ---------------------------------------------------------------------

    pd.DataFrame([summary]).to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ---------------------------------------------------------------------
    # JSON PACKAGE
    # ---------------------------------------------------------------------

    package = {
        "dashboard": {
            "name": "PEARLS AQI Predictor",
            "step": 22,
            "type": (
                "Interactive + Infographic + Intelligence"
            ),
            "target": TARGET,
            "forecast_horizon": FORECAST_HORIZON,
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        "location": location,

        "summary": summary,

        "aqi": {
            "summary": aqi_summary,
            "forecast": aqi_df,
        },

        "pollution": {
            "available_pollutants": list(
                pollution_intelligence[
                    "pollutant"
                ]
            )
            if not pollution_intelligence.empty
            else [],
            "intelligence": pollution_intelligence,
        },

        "events": {
            "summary": {
                "total_events": event_summary[
                    "total_events"
                ],
                "high_severity_events":
                    event_summary[
                        "high_severity_events"
                    ],
                "elevated_events":
                    event_summary[
                        "elevated_events"
                    ],
                "pollutant_counts":
                    event_summary[
                        "pollutant_counts"
                    ],
                "severity_counts":
                    event_summary[
                        "severity_counts"
                    ],
            },

            "recent_events":
                event_summary[
                    "recent_events"
                ],
        },

        "source_files": {
            "step20_forecast":
                str(FORECAST_FILE),

            "step21_location":
                str(LOCATION_FILE),

            "step21_pollution":
                str(POLLUTION_INTELLIGENCE_FILE),

            "step21_pollution_hourly":
                str(POLLUTION_HOURLY_FILE),

            "step21_aqi":
                str(AQI_INTELLIGENCE_FILE),

            "step21_events":
                str(EVENTS_FILE),
        },

        "cache": {
            "enabled": True,
            "hit": cache_hit,
            "input_hash": input_hash,
        },
    }

    # IMPORTANT:
    # json_safe() recursively converts DataFrames before json.dump().
    write_json(
        OUTPUT_DATA,
        package,
    )

    # ---------------------------------------------------------------------
    # HTML
    # ---------------------------------------------------------------------

    create_html(
        location,
        aqi_df,
        aqi_summary,
        pollution_intelligence,
        event_summary,
    )

    save_cache(input_hash)

    # ---------------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------------

    checks = validate_dashboard(
        forecast,
        pollution_intelligence,
    )

    # ---------------------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------------------

    report = {
        "step": 22,
        "status": "PASS",
        "dashboard_type": (
            "Interactive + Infographic + Intelligence"
        ),

        "location": location,

        "pollution_ingredients": int(
            len(pollution_intelligence)
        ),

        "dominant_pollutant":
            summary["dominant_pollutant"],

        "forecast": {
            "horizon": FORECAST_HORIZON,
            "minimum":
                aqi_summary["minimum"],
            "maximum":
                aqi_summary["maximum"],
            "mean":
                aqi_summary["mean"],
            "median":
                aqi_summary["median"],
            "dominant_category":
                aqi_summary[
                    "dominant_category"
                ],
            "trend":
                aqi_summary["trend"],
        },

        "events": {
            "total":
                event_summary[
                    "total_events"
                ],
            "high_severity":
                event_summary[
                    "high_severity_events"
                ],
            "elevated":
                event_summary[
                    "elevated_events"
                ],
        },

        "cache": {
            "status":
                "HIT" if cache_hit else "MISS",
            "hash":
                input_hash,
        },

        "validation": checks,

        "files": {
            "html": str(OUTPUT_HTML),
            "data": str(OUTPUT_DATA),
            "forecast": str(OUTPUT_FORECAST),
            "pollution": str(OUTPUT_POLLUTION),
            "summary": str(OUTPUT_SUMMARY),
            "cache": str(CACHE_HASH_FILE),
            "report": str(OUTPUT_REPORT),
        },

        "model_training": False,
        "model_selection": False,
        "validation_test": False,
        "future_target_leakage": "NONE",
    }

    write_json(
        OUTPUT_REPORT,
        report,
    )

    # ---------------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------------

    execution_time = (
        time.time() - start_time
    )

    banner("STEP 22 COMPLETE")

    item("Dashboard status", "PASS")

    item(
        "Location",
        location["display_name"],
    )

    item(
        "Pollution ingredients",
        f"{len(pollution_intelligence)} available",
    )

    item(
        "Dominant pollutant",
        summary["dominant_pollutant"],
    )

    item(
        "Forecast AQI mean",
        f"{aqi_summary['mean']:.3f}",
    )

    item(
        "Forecast AQI maximum",
        f"{aqi_summary['maximum']:.3f}",
    )

    item(
        "Dominant AQI category",
        aqi_summary["dominant_category"],
    )

    item(
        "Forecast trend",
        aqi_summary["trend"],
    )

    item(
        "Pollution events",
        event_summary["total_events"],
    )

    item(
        "Cache status",
        "HIT" if cache_hit else "MISS",
    )

    item(
        "JSON serialization",
        "PASS",
    )

    item(
        "Dashboard validation",
        "PASS",
    )

    print()
    print("Dashboard HTML:")
    print(OUTPUT_HTML)

    print()
    print("Dashboard data:")
    print(OUTPUT_DATA)

    print()
    print("Dashboard forecast:")
    print(OUTPUT_FORECAST)

    print()
    print("Dashboard pollution:")
    print(OUTPUT_POLLUTION)

    print()
    print("Dashboard summary:")
    print(OUTPUT_SUMMARY)

    print()
    print("Step 22 report:")
    print(OUTPUT_REPORT)

    print()
    item(
        "Execution time",
        f"{execution_time:.3f}s",
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        banner("STEP 22 FAILED")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise