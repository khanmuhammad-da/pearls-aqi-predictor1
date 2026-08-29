"""
PEARLS AQI PREDICTOR
STEP 20 — PRODUCTION DASHBOARD

Purpose
-------
Build the actual user-facing production dashboard from the existing
Step 18 / Step 19 production artifacts.

This step DOES NOT:
    - train a model
    - tune a model
    - select a model
    - perform validation
    - perform test-set evaluation
    - modify predictions

This step DOES:
    - load the production 72-hour forecast
    - discover available pollution-ingredient data
    - discover location metadata
    - create a cached dashboard data package
    - create an interactive, self-contained HTML dashboard
    - create infographic-style visual sections
    - reuse existing Step 18 visualization assets
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"

STEP18_FORECAST = REPORTS_DIR / "dashboard_ready_forecast.csv"
STEP18_DATA = REPORTS_DIR / "visualizations" / "dashboard_data.json"
STEP18_REPORT = REPORTS_DIR / "production_visualization_results.json"

STEP19_DIR = REPORTS_DIR / "production_dashboard"
STEP19_FORECAST = STEP19_DIR / "production_dashboard_forecast.csv"
STEP19_HOURLY = STEP19_DIR / "production_dashboard_hourly.csv"
STEP19_SUMMARY = STEP19_DIR / "production_dashboard_summary.csv"
STEP19_PACKAGE = STEP19_DIR / "production_dashboard_package.json"

OUTPUT_DIR = REPORTS_DIR / "production_dashboard_v2"
CACHE_DIR = OUTPUT_DIR / "cache"
ASSET_DIR = OUTPUT_DIR / "assets"

DASHBOARD_HTML = OUTPUT_DIR / "production_dashboard.html"
DASHBOARD_DATA = OUTPUT_DIR / "production_dashboard_data.json"
DASHBOARD_SUMMARY = OUTPUT_DIR / "production_dashboard_summary.csv"
DASHBOARD_FORECAST = OUTPUT_DIR / "production_dashboard_forecast.csv"
DASHBOARD_REPORT = OUTPUT_DIR / "production_dashboard_v2_results.json"

CACHE_FILE = CACHE_DIR / "dashboard_cache.json"
CACHE_HASH_FILE = CACHE_DIR / "dashboard_hash.json"


# Candidate pollution/data files.
# The script searches these locations without assuming that all exist.
DATA_SEARCH_DIRS = [
    REPORTS_DIR,
    REPORTS_DIR / "predictions",
    REPORTS_DIR / "visualizations",
    BASE_DIR / "data",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data" / "features",
    BASE_DIR / "datasets",
    BASE_DIR / "output",
]

# Candidate column names for pollutant discovery.
POLLUTANT_ALIASES = {
    "pm25": [
        "pm25",
        "pm2_5",
        "pm2.5",
        "pm_2_5",
        "PM25",
        "PM2.5",
        "PM2_5",
        "particulate_matter_2_5",
    ],
    "pm10": [
        "pm10",
        "pm_10",
        "PM10",
        "particulate_matter_10",
    ],
    "o3": [
        "o3",
        "O3",
        "ozone",
        "Ozone",
    ],
    "no2": [
        "no2",
        "NO2",
        "nitrogen_dioxide",
        "NitrogenDioxide",
    ],
    "so2": [
        "so2",
        "SO2",
        "sulfur_dioxide",
        "sulphur_dioxide",
    ],
    "co": [
        "co",
        "CO",
        "carbon_monoxide",
        "CarbonMonoxide",
    ],
}

POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3": "O₃",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
}

POLLUTANT_UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "o3": "µg/m³",
    "no2": "µg/m³",
    "so2": "µg/m³",
    "co": "mg/m³",
}

LOCATION_ALIASES = {
    "city": [
        "city",
        "location",
        "station_name",
        "station",
        "site_name",
        "site",
    ],
    "country": [
        "country",
        "country_name",
    ],
    "latitude": [
        "latitude",
        "lat",
        "Latitude",
        "LAT",
    ],
    "longitude": [
        "longitude",
        "lon",
        "lng",
        "Longitude",
        "LON",
    ],
}


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================

def line(char: str = "=", width: int = 72) -> None:
    print(char * width)


def heading(text: str) -> None:
    print()
    line("=")
    print(text)
    line("=")


def info(label: str, value: Any) -> None:
    print(f"{label:<32}: {value}")


# ============================================================================
# HELPERS
# ============================================================================

def json_default(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-compatible values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return str(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def first_existing_column(
    columns: Iterable[str],
    aliases: Iterable[str],
) -> Optional[str]:
    """Find a column using case-insensitive matching."""
    columns = list(columns)
    normalized = {str(c).strip().lower(): c for c in columns}

    for alias in aliases:
        if alias in columns:
            return alias

        key = str(alias).strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def classify_aqi(aqi: float) -> str:
    """
    US AQI-style category.
    """
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


def risk_level(category: str) -> str:
    category = str(category).lower()

    if "hazardous" in category:
        return "EXTREME"
    if "very unhealthy" in category:
        return "VERY HIGH"
    if "unhealthy" in category:
        return "HIGH"
    if "sensitive" in category:
        return "ELEVATED"
    if "moderate" in category:
        return "MODERATE"

    return "LOW"


def category_short(category: str) -> str:
    category = str(category)

    if category == "Unhealthy for Sensitive Groups":
        return "USG"

    return category


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    """
    Stable-ish hash for cache invalidation.
    """
    payload = {
        "columns": [str(c) for c in df.columns],
        "rows": df.astype(str).fillna("").values.tolist(),
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=json_default,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FORECAST LOADING
# ============================================================================

def locate_forecast() -> Path:
    """
    Prefer Step 19 production dashboard forecast.
    Fall back to Step 18 dashboard-ready forecast.
    """
    candidates = [
        STEP19_FORECAST,
        STEP18_FORECAST,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No production forecast found. Expected one of:\n"
        f"  {STEP19_FORECAST}\n"
        f"  {STEP18_FORECAST}"
    )


def load_forecast() -> Tuple[pd.DataFrame, Path]:
    path = locate_forecast()

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Forecast file is empty: {path}")

    print(f"Forecast source                  : {path}")

    # ---------------------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------------------
    timestamp_col = first_existing_column(
        df.columns,
        [
            "timestamp",
            "datetime",
            "date_time",
            "forecast_time",
            "time",
        ],
    )

    if timestamp_col is None:
        raise ValueError(
            "Forecast does not contain a timestamp column."
        )

    df["timestamp"] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------------------
    # Horizon
    # ---------------------------------------------------------------------
    horizon_col = first_existing_column(
        df.columns,
        [
            "horizon",
            "horizon_hour",
            "forecast_horizon",
            "step",
        ],
    )

    if horizon_col is not None:
        df["horizon"] = pd.to_numeric(
            df[horizon_col],
            errors="coerce",
        )
    else:
        df["horizon"] = range(1, len(df) + 1)

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    prediction_col = first_existing_column(
        df.columns,
        [
            "predicted_us_aqi",
            "predicted_aqi",
            "prediction",
            "predicted",
            "forecast_aqi",
            "us_aqi",
            "aqi",
        ],
    )

    if prediction_col is None:
        raise ValueError(
            "Could not identify AQI prediction column.\n"
            f"Available columns: {list(df.columns)}"
        )

    df["predicted_aqi"] = pd.to_numeric(
        df[prediction_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "predicted_aqi"]
    ).copy()

    if df.empty:
        raise ValueError("Forecast contains no valid rows.")

    # ---------------------------------------------------------------------
    # Category
    # ---------------------------------------------------------------------
    category_col = first_existing_column(
        df.columns,
        [
            "aqi_category",
            "category",
            "predicted_category",
            "forecast_category",
        ],
    )

    if category_col is not None:
        df["category"] = df[category_col].astype(str)
    else:
        df["category"] = df["predicted_aqi"].apply(classify_aqi)

    df["risk"] = df["category"].apply(risk_level)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df, path


# ============================================================================
# POLLUTION INGREDIENT DISCOVERY
# ============================================================================

def discover_csv_files() -> List[Path]:
    files: List[Path] = []
    seen = set()

    for directory in DATA_SEARCH_DIRS:
        if not directory.exists():
            continue

        try:
            candidates = directory.rglob("*.csv")
        except Exception:
            continue

        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path

            if resolved in seen:
                continue

            seen.add(resolved)
            files.append(path)

    return sorted(
        files,
        key=lambda p: str(p).lower(),
    )


def score_pollution_file(
    path: Path,
) -> Tuple[int, Dict[str, str], Dict[str, str]]:
    """
    Score a CSV based on pollutant and location columns.

    Returns:
        score, pollutant_column_mapping, location_column_mapping
    """
    try:
        columns = pd.read_csv(
            path,
            nrows=0,
        ).columns.tolist()
    except Exception:
        return 0, {}, {}

    pollutant_columns: Dict[str, str] = {}

    for pollutant, aliases in POLLUTANT_ALIASES.items():
        column = first_existing_column(
            columns,
            aliases,
        )

        if column is not None:
            pollutant_columns[pollutant] = column

    location_columns: Dict[str, str] = {}

    for key, aliases in LOCATION_ALIASES.items():
        column = first_existing_column(
            columns,
            aliases,
        )

        if column is not None:
            location_columns[key] = column

    score = len(pollutant_columns) * 10
    score += len(location_columns) * 2

    filename = path.name.lower()

    if "feature" in filename:
        score += 3

    if "historical" in filename:
        score += 2

    if "dashboard" in filename:
        score -= 10

    if "prediction" in filename:
        score -= 10

    return score, pollutant_columns, location_columns


def discover_pollution_source(
    forecast_source: Path,
) -> Tuple[
    Optional[Path],
    Dict[str, str],
    Dict[str, str],
]:
    """
    Find the best available source containing pollutant ingredients
    and/or location metadata.
    """
    candidates = discover_csv_files()

    ranked = []

    for path in candidates:
        if path.resolve() == forecast_source.resolve():
            continue

        score, pollutant_columns, location_columns = score_pollution_file(
            path
        )

        if score > 0:
            ranked.append(
                (
                    score,
                    path,
                    pollutant_columns,
                    location_columns,
                )
            )

    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1]).lower(),
        )
    )

    if not ranked:
        return None, {}, {}

    score, path, pollutant_columns, location_columns = ranked[0]

    print(f"Pollution data source             : {path}")
    print(f"Pollutant fields discovered       : {len(pollutant_columns)}")
    print(f"Location fields discovered        : {len(location_columns)}")

    return path, pollutant_columns, location_columns


# ============================================================================
# POLLUTION DATA PROCESSING
# ============================================================================

def load_pollution_data(
    source: Optional[Path],
    pollutant_columns: Dict[str, str],
    location_columns: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load latest available pollutant observations.

    If timestamp exists, choose the most recent record.

    This is intentionally observation-oriented. It does NOT fabricate
    future pollutant concentrations from the AQI model.
    """
    metadata: Dict[str, Any] = {
        "available": False,
        "source": None,
        "pollutants": {},
        "location": {},
        "timestamp": None,
        "note": None,
    }

    if source is None:
        metadata["note"] = (
            "No pollution-ingredient source was discovered. "
            "Dashboard displays AQI forecast without pollutant "
            "observation cards."
        )
        return pd.DataFrame(), metadata

    try:
        df = pd.read_csv(source)
    except Exception as exc:
        metadata["note"] = f"Could not read pollution source: {exc}"
        return pd.DataFrame(), metadata

    if df.empty:
        metadata["note"] = "Pollution source contains no rows."
        return pd.DataFrame(), metadata

    # Timestamp
    timestamp_col = first_existing_column(
        df.columns,
        [
            "timestamp",
            "datetime",
            "date_time",
            "time",
            "date",
        ],
    )

    if timestamp_col:
        timestamps = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            utc=True,
        )

        df["_dashboard_timestamp"] = timestamps

        valid = df.dropna(
            subset=["_dashboard_timestamp"]
        )

        if not valid.empty:
            row = valid.sort_values(
                "_dashboard_timestamp"
            ).iloc[-1]

            metadata["timestamp"] = (
                row["_dashboard_timestamp"].isoformat()
            )
        else:
            row = df.iloc[-1]
    else:
        row = df.iloc[-1]

    metadata["source"] = str(source)
    metadata["available"] = True

    # Pollutants
    for pollutant, column in pollutant_columns.items():
        value = safe_float(row.get(column))

        metadata["pollutants"][pollutant] = {
            "label": POLLUTANT_LABELS[pollutant],
            "value": value,
            "unit": POLLUTANT_UNITS[pollutant],
            "source_column": column,
        }

    # Location
    for key, column in location_columns.items():
        value = row.get(column)

        if pd.isna(value):
            continue

        if key in ("latitude", "longitude"):
            numeric = safe_float(value)

            if numeric is not None:
                metadata["location"][key] = numeric
        else:
            metadata["location"][key] = str(value)

    return df, metadata


# ============================================================================
# LOCATION
# ============================================================================

def enrich_location(
    location: Dict[str, Any],
    step18_data: Optional[Dict[str, Any]],
    step19_package: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Add location metadata from existing JSON artifacts if available.

    No external geocoding is performed.
    """
    result = dict(location)

    candidates = []

    if isinstance(step18_data, dict):
        candidates.extend([
            step18_data.get("location"),
            step18_data.get("metadata"),
        ])

    if isinstance(step19_package, dict):
        candidates.extend([
            step19_package.get("location"),
            step19_package.get("metadata"),
        ])

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        for key in (
            "city",
            "country",
            "latitude",
            "longitude",
        ):
            if key not in result and key in candidate:
                result[key] = candidate[key]

    return result


# ============================================================================
# EXISTING VISUALIZATION ASSETS
# ============================================================================

def copy_existing_visualizations() -> Dict[str, str]:
    """
    Copy Step 18 PNGs into the Step 20 asset directory.
    """
    candidates = {
        "forecast_png": (
            REPORTS_DIR
            / "visualizations"
            / "production_72h_forecast.png"
        ),
        "categories_png": (
            REPORTS_DIR
            / "visualizations"
            / "production_72h_categories.png"
        ),
        "risk_png": (
            REPORTS_DIR
            / "visualizations"
            / "production_72h_risk.png"
        ),
        "blocks_png": (
            REPORTS_DIR
            / "visualizations"
            / "production_24h_blocks.png"
        ),
    }

    copied = {}

    for key, source in candidates.items():
        if not source.exists():
            continue

        destination = ASSET_DIR / source.name

        try:
            shutil.copy2(
                source,
                destination,
            )
            copied[key] = str(
                destination.relative_to(OUTPUT_DIR)
            )
        except Exception as exc:
            print(
                f"Warning: could not copy {source}: {exc}"
            )

    return copied


# ============================================================================
# STEP 18 / STEP 19 JSON LOADING
# ============================================================================

def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


# ============================================================================
# FORECAST SUMMARY
# ============================================================================

def build_forecast_summary(
    forecast: pd.DataFrame,
) -> Dict[str, Any]:
    values = forecast["predicted_aqi"].astype(float)

    peak_idx = values.idxmax()
    min_idx = values.idxmin()

    peak_row = forecast.loc[peak_idx]
    min_row = forecast.loc[min_idx]

    categories = (
        forecast["category"]
        .value_counts()
        .to_dict()
    )

    risks = (
        forecast["risk"]
        .value_counts()
        .to_dict()
    )

    summary: Dict[str, Any] = {
        "rows": int(len(forecast)),
        "forecast_start": forecast["timestamp"].min().isoformat(),
        "forecast_end": forecast["timestamp"].max().isoformat(),
        "minimum_aqi": round(float(values.min()), 3),
        "maximum_aqi": round(float(values.max()), 3),
        "mean_aqi": round(float(values.mean()), 3),
        "median_aqi": round(float(values.median()), 3),
        "peak_timestamp": peak_row["timestamp"].isoformat(),
        "peak_aqi": round(float(peak_row["predicted_aqi"]), 3),
        "minimum_timestamp": min_row["timestamp"].isoformat(),
        "minimum_aqi_timestamp": min_row["timestamp"].isoformat(),
        "dominant_category": (
            forecast["category"].mode().iloc[0]
            if not forecast["category"].mode().empty
            else classify_aqi(float(values.mean()))
        ),
        "category_distribution": categories,
        "risk_distribution": risks,
    }

    # Time windows
    for hours in (24, 48, 72):
        subset = forecast.head(hours)

        if subset.empty:
            continue

        window_values = subset["predicted_aqi"].astype(float)

        summary[f"{hours}h"] = {
            "min": round(float(window_values.min()), 3),
            "max": round(float(window_values.max()), 3),
            "mean": round(float(window_values.mean()), 3),
            "category": (
                subset["category"].mode().iloc[0]
                if not subset["category"].mode().empty
                else classify_aqi(float(window_values.mean()))
            ),
        }

    # 24-hour blocks
    block_summaries = []

    for block_number, start in enumerate(
        range(0, min(len(forecast), 72), 24),
        start=1,
    ):
        subset = forecast.iloc[start:start + 24]

        vals = subset["predicted_aqi"].astype(float)

        if subset.empty:
            continue

        block_summaries.append({
            "block": block_number,
            "start": subset["timestamp"].iloc[0].isoformat(),
            "end": subset["timestamp"].iloc[-1].isoformat(),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "mean": round(float(vals.mean()), 3),
            "category": (
                subset["category"].mode().iloc[0]
                if not subset["category"].mode().empty
                else classify_aqi(float(vals.mean()))
            ),
        })

    summary["blocks"] = block_summaries

    return summary


# ============================================================================
# CACHE
# ============================================================================

def build_cache_key(
    forecast: pd.DataFrame,
    pollution_source: Optional[Path],
    location: Dict[str, Any],
) -> str:
    payload = {
        "forecast_hash": dataframe_hash(forecast),
        "pollution_file": (
            file_hash(pollution_source)
            if pollution_source and pollution_source.exists()
            else None
        ),
        "location": location,
        "dashboard_version": 2,
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=json_default,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def check_cache(cache_key: str) -> bool:
    if not CACHE_HASH_FILE.exists():
        return False

    try:
        with CACHE_HASH_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        return data.get("cache_key") == cache_key

    except Exception:
        return False


def save_cache_key(cache_key: str) -> None:
    payload = {
        "cache_key": cache_key,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with CACHE_HASH_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


# ============================================================================
# HTML GENERATION
# ============================================================================

def html_escape(value: Any) -> str:
    import html
    return html.escape(str(value))


def js_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        default=json_default,
    ).replace("</", "<\\/")


def build_dashboard_html(
    dashboard_data: Dict[str, Any],
) -> str:
    """
    Self-contained dashboard.

    No Plotly CDN.
    No external JavaScript.
    No external CSS.
    """

    payload = js_json(dashboard_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Pearls AQI Predictor — Production Dashboard</title>

<style>

:root {{
    --bg: #f4f6f8;
    --panel: #ffffff;
    --text: #17202a;
    --muted: #6b7280;
    --border: #e5e7eb;
    --good: #16a34a;
    --moderate: #ca8a04;
    --usg: #ea580c;
    --unhealthy: #dc2626;
    --very-unhealthy: #9333ea;
    --hazardous: #7f1d1d;
    --accent: #2563eb;
    --shadow: 0 8px 24px rgba(0,0,0,.07);
}}

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

    background: var(--bg);
    color: var(--text);
}}

.container {{
    width: min(1450px, 94vw);
    margin: 0 auto;
}}

header {{
    padding: 28px 0 18px;
}}

.header-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    box-shadow: var(--shadow);
}}

.brand {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}}

.brand h1 {{
    margin: 0;
    font-size: 28px;
}}

.brand p {{
    margin: 6px 0 0;
    color: var(--muted);
}}

.badge {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 800;
    background: #eef2ff;
    color: #3730a3;
}}

.location {{
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}

.location span {{
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 11px;
    background: #fafafa;
    color: #4b5563;
    font-size: 13px;
}}

.grid {{
    display: grid;
    gap: 16px;
}}

.kpi-grid {{
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
}}

.two-grid {{
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
}}

.three-grid {{
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
}}

.panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow);
}}

.panel h2 {{
    margin: 0 0 5px;
    font-size: 17px;
}}

.panel-subtitle {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 16px;
}}

.kpi {{
    min-height: 145px;
}}

.kpi-label {{
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-weight: 700;
}}

.kpi-value {{
    margin-top: 12px;
    font-size: 35px;
    font-weight: 850;
    letter-spacing: -.04em;
}}

.kpi-note {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 7px;
}}

.section {{
    margin: 16px 0;
}}

.gauge-wrap {{
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
}}

.gauge {{
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background:
        conic-gradient(
            var(--good) 0deg 36deg,
            var(--moderate) 36deg 72deg,
            var(--usg) 72deg 108deg,
            var(--unhealthy) 108deg 180deg,
            var(--very-unhealthy) 180deg 270deg,
            var(--hazardous) 270deg 360deg
        );
    position: relative;
}}

.gauge::after {{
    content: "";
    position: absolute;
    inset: 28px;
    border-radius: 50%;
    background: white;
}}

.gauge-center {{
    position: absolute;
    z-index: 2;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}

.gauge-number {{
    font-size: 42px;
    font-weight: 900;
}}

.gauge-category {{
    font-size: 13px;
    font-weight: 800;
    text-align: center;
    max-width: 150px;
}}

.ingredient-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 12px;
}}

.ingredient {{
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 14px;
    background: #fafafa;
}}

.ingredient-name {{
    font-weight: 850;
}}

.ingredient-value {{
    margin-top: 8px;
    font-size: 25px;
    font-weight: 850;
}}

.ingredient-unit {{
    color: var(--muted);
    font-size: 11px;
}}

.missing {{
    color: var(--muted);
    font-style: italic;
}}

.timeline {{
    width: 100%;
    overflow-x: auto;
}}

canvas {{
    width: 100%;
    height: 380px;
    display: block;
}}

.forecast-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

.forecast-table th,
.forecast-table td {{
    padding: 10px 9px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}}

.forecast-table th {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
}}

.category-pill {{
    display: inline-flex;
    padding: 5px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}}

.category-good {{
    background: #dcfce7;
    color: #166534;
}}

.category-moderate {{
    background: #fef9c3;
    color: #854d0e;
}}

.category-usg {{
    background: #ffedd5;
    color: #9a3412;
}}

.category-unhealthy {{
    background: #fee2e2;
    color: #991b1b;
}}

.category-very-unhealthy {{
    background: #f3e8ff;
    color: #6b21a8;
}}

.category-hazardous {{
    background: #450a0a;
    color: #ffffff;
}}

.block-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 12px;
}}

.block {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
}}

.block-number {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    font-weight: 800;
}}

.block-aqi {{
    font-size: 30px;
    font-weight: 900;
    margin: 6px 0;
}}

.stat-row {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}}

.stat-row:last-child {{
    border-bottom: 0;
}}

.stat-label {{
    color: var(--muted);
}}

.risk-banner {{
    border-radius: 14px;
    padding: 17px;
    background: #fef2f2;
    border: 1px solid #fecaca;
}}

.risk-title {{
    font-size: 22px;
    font-weight: 900;
}}

.health-list {{
    margin: 10px 0 0;
    padding-left: 20px;
    color: #4b5563;
    line-height: 1.65;
}}

.asset {{
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-top: 10px;
}}

footer {{
    padding: 24px 0 40px;
    color: var(--muted);
    font-size: 12px;
}}

@media (max-width: 1050px) {{
    .kpi-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .three-grid,
    .two-grid {{
        grid-template-columns: 1fr;
    }}

    .ingredient-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}
}}

@media (max-width: 650px) {{
    .kpi-grid,
    .ingredient-grid,
    .block-grid {{
        grid-template-columns: 1fr;
    }}

    .brand {{
        flex-direction: column;
        align-items: flex-start;
    }}

    .gauge {{
        width: 220px;
        height: 220px;
    }}
}}

</style>
</head>

<body>

<div class="container">

<header>
<div class="header-card">

<div class="brand">

<div>
<h1>PEARLS AQI PREDICTOR</h1>
<p>Production Air Quality Forecast Dashboard</p>
</div>

<div class="badge">
72-HOUR PRODUCTION FORECAST
</div>

</div>

<div id="location" class="location"></div>

</div>
</header>


<!-- KPI CARDS -->
<section class="section">
<div class="grid kpi-grid">

<div class="panel kpi">
<div class="kpi-label">Mean AQI</div>
<div id="meanAqi" class="kpi-value">—</div>
<div class="kpi-note">72-hour forecast average</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Peak AQI</div>
<div id="peakAqi" class="kpi-value">—</div>
<div id="peakTime" class="kpi-note">—</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Minimum AQI</div>
<div id="minAqi" class="kpi-value">—</div>
<div id="minTime" class="kpi-note">—</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Dominant Category</div>
<div id="dominantCategory" class="kpi-value"
     style="font-size:23px">—</div>
<div id="dominantRisk" class="kpi-note">—</div>
</div>

</div>
</section>


<!-- AQI GAUGE + RISK -->
<section class="section">
<div class="grid two-grid">

<div class="panel">
<h2>AQI Health Indicator</h2>
<div class="panel-subtitle">
Forecast mean AQI and corresponding category
</div>

<div class="gauge-wrap">
<div class="gauge">
<div class="gauge-center">
<div id="gaugeNumber" class="gauge-number">—</div>
<div id="gaugeCategory" class="gauge-category">—</div>
</div>
</div>
</div>
</div>


<div class="panel">
<h2>Air Quality Risk</h2>
<div class="panel-subtitle">
Operational interpretation of the forecast
</div>

<div class="risk-banner">
<div id="riskTitle" class="risk-title">—</div>
<div id="riskDescription">—</div>
</div>

<ul id="healthList" class="health-list"></ul>

<div style="margin-top:15px">

<div class="stat-row">
<span class="stat-label">Forecast period</span>
<strong id="forecastPeriod">—</strong>
</div>

<div class="stat-row">
<span class="stat-label">Forecast horizons</span>
<strong id="forecastRows">—</strong>
</div>

<div class="stat-row">
<span class="stat-label">Peak hour</span>
<strong id="peakHour">—</strong>
</div>

</div>
</div>

</div>
</section>


<!-- POLLUTION INGREDIENTS -->
<section class="section">
<div class="panel">

<h2>Pollution Ingredients</h2>

<div class="panel-subtitle">
Latest available pollutant observations from the project data.
These values are observations and are not fabricated from the AQI forecast.
</div>

<div id="ingredients"
     class="ingredient-grid">
</div>

</div>
</section>


<!-- FORECAST GRAPH -->
<section class="section">
<div class="panel">

<h2>72-Hour AQI Forecast</h2>

<div class="panel-subtitle">
Hourly production forecast
</div>

<div class="timeline">
<canvas id="forecastCanvas"></canvas>
</div>

</div>
</section>


<!-- TIME WINDOWS -->
<section class="section">
<div class="panel">

<h2>Forecast Windows</h2>

<div class="panel-subtitle">
24-hour blocks across the production horizon
</div>

<div id="blocks"
     class="block-grid">
</div>

</div>
</section>


<!-- CATEGORY / RISK -->
<section class="section">
<div class="grid two-grid">

<div class="panel">

<h2>AQI Category Distribution</h2>

<div class="panel-subtitle">
Number of forecast hours in each category
</div>

<div id="categoryDistribution"></div>

</div>


<div class="panel">

<h2>Risk Distribution</h2>

<div class="panel-subtitle">
Forecast-hour risk profile
</div>

<div id="riskDistribution"></div>

</div>

</div>
</section>


<!-- EXISTING STEP 18 VISUALIZATIONS -->
<section class="section">
<div class="panel">

<h2>Production Infographics</h2>

<div class="panel-subtitle">
Visual assets generated by the preceding production pipeline.
</div>

<div id="assets"></div>

</div>
</section>


<!-- HOURLY TABLE -->
<section class="section">
<div class="panel">

<h2>Hourly Forecast</h2>

<div class="panel-subtitle">
Complete 72-hour forecast
</div>

<div style="overflow-x:auto">

<table class="forecast-table">

<thead>
<tr>
<th>Horizon</th>
<th>Timestamp</th>
<th>AQI</th>
<th>Category</th>
<th>Risk</th>
</tr>
</thead>

<tbody id="forecastTable"></tbody>

</table>

</div>

</div>
</section>


<!-- MODEL / DATA INFO -->
<section class="section">
<div class="grid three-grid">

<div class="panel">
<h2>Production Model</h2>

<div class="stat-row">
<span class="stat-label">Model</span>
<strong id="modelName">XGBoost tuned</strong>
</div>

<div class="stat-row">
<span class="stat-label">Target</span>
<strong>us_aqi</strong>
</div>

<div class="stat-row">
<span class="stat-label">Horizon</span>
<strong>72 hours</strong>
</div>

</div>


<div class="panel">
<h2>Dashboard Status</h2>

<div class="stat-row">
<span class="stat-label">Cache</span>
<strong id="cacheStatus">—</strong>
</div>

<div class="stat-row">
<span class="stat-label">Pollution data</span>
<strong id="pollutionStatus">—</strong>
</div>

<div class="stat-row">
<span class="stat-label">Location</span>
<strong id="locationStatus">—</strong>
</div>

</div>


<div class="panel">
<h2>Last Updated</h2>

<div style="font-size:24px;font-weight:850;margin-top:10px"
     id="lastUpdated">
—
</div>

<div class="kpi-note">
Dashboard package generation time
</div>

</div>

</div>
</section>


<footer>
PEARLS AQI PREDICTOR · Production Dashboard ·
Forecast data generated by the production XGBoost pipeline.
</footer>

</div>


<script>

const DATA = {payload};


function esc(value) {{
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
}}


function categoryClass(category) {{

    const c = String(category).toLowerCase();

    if (c.includes("hazardous"))
        return "category-hazardous";

    if (c.includes("very unhealthy"))
        return "category-very-unhealthy";

    if (c.includes("unhealthy"))
        return "category-unhealthy";

    if (c.includes("sensitive"))
        return "category-usg";

    if (c.includes("moderate"))
        return "category-moderate";

    return "category-good";
}}


function fmt(value, digits=1) {{

    if (value === null || value === undefined)
        return "—";

    const number = Number(value);

    if (!Number.isFinite(number))
        return "—";

    return number.toFixed(digits);
}}


function fmtDate(value) {{

    if (!value)
        return "—";

    const date = new Date(value);

    if (Number.isNaN(date.getTime()))
        return value;

    return date.toLocaleString();
}}


function renderLocation() {{

    const location = DATA.location || {{}};

    const box = document.getElementById("location");

    const parts = [];

    if (location.city)
        parts.push("📍 " + location.city);

    if (location.country)
        parts.push(location.country);

    if (
        location.latitude !== undefined &&
        location.longitude !== undefined
    ) {{
        parts.push(
            "Coordinates: " +
            Number(location.latitude).toFixed(4) +
            ", " +
            Number(location.longitude).toFixed(4)
        );
    }}

    if (!parts.length)
        parts.push("📍 Location metadata not available");

    box.innerHTML = parts
        .map(x => "<span>" + esc(x) + "</span>")
        .join("");

    document.getElementById("locationStatus").textContent =
        parts.length ? "AVAILABLE" : "NOT AVAILABLE";
}}


function renderKpis() {{

    const s = DATA.summary;

    document.getElementById("meanAqi").textContent =
        fmt(s.mean_aqi);

    document.getElementById("peakAqi").textContent =
        fmt(s.maximum_aqi);

    document.getElementById("minAqi").textContent =
        fmt(s.minimum_aqi);

    document.getElementById("dominantCategory").textContent =
        s.dominant_category || "—";

    document.getElementById("dominantRisk").textContent =
        "Risk: " +
        (
            DATA.forecast[0]?.risk || "—"
        );

    document.getElementById("peakTime").textContent =
        "Peak: " + fmtDate(s.peak_timestamp);

    document.getElementById("minTime").textContent =
        "Minimum: " + fmtDate(s.minimum_timestamp);

    document.getElementById("gaugeNumber").textContent =
        fmt(s.mean_aqi);

    document.getElementById("gaugeCategory").textContent =
        s.dominant_category || "—";

    document.getElementById("forecastPeriod").textContent =
        fmtDate(s.forecast_start) +
        " → " +
        fmtDate(s.forecast_end);

    document.getElementById("forecastRows").textContent =
        s.rows + " hours";

    document.getElementById("peakHour").textContent =
        fmtDate(s.peak_timestamp);
}}


function renderRisk() {{

    const category =
        DATA.summary.dominant_category || "Unknown";

    let title = "LOW RISK";
    let description =
        "Forecast conditions are generally favorable.";

    if (category.includes("Hazardous")) {{
        title = "EXTREME RISK";
        description =
            "Very high pollution is forecast. Exposure should be minimized.";
    }}
    else if (category.includes("Very Unhealthy")) {{
        title = "VERY HIGH RISK";
        description =
            "Pollution levels may cause significant health effects.";
    }}
    else if (category.includes("Unhealthy")) {{
        title = "HIGH RISK";
        description =
            "Air quality is forecast to be unhealthy for the general population.";
    }}
    else if (category.includes("Sensitive")) {{
        title = "ELEVATED RISK";
        description =
            "Sensitive groups may experience health effects.";
    }}
    else if (category.includes("Moderate")) {{
        title = "MODERATE RISK";
        description =
            "Air quality is generally acceptable, but some individuals may be sensitive.";
    }}

    document.getElementById("riskTitle").textContent = title;

    document.getElementById("riskDescription").textContent =
        description;

    const list = document.getElementById("healthList");

    const items = [];

    if (category.includes("Unhealthy")) {{
        items.push(
            "Consider reducing prolonged or intense outdoor activity."
        );
        items.push(
            "Sensitive individuals should consider limiting exposure."
        );
    }}
    else if (category.includes("Moderate")) {{
        items.push(
            "Most people can continue normal outdoor activity."
        );
        items.push(
            "Unusually sensitive individuals may want to monitor symptoms."
        );
    }}
    else {{
        items.push(
            "Air quality conditions are generally favorable."
        );
    }}

    list.innerHTML = items
        .map(x => "<li>" + esc(x) + "</li>")
        .join("");
}}


function renderIngredients() {{

    const container =
        document.getElementById("ingredients");

    const pollutants =
        DATA.pollution?.pollutants || {{}};

    const keys = [
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co"
    ];

    let html = "";

    for (const key of keys) {{

        const item = pollutants[key];

        if (!item) {{
            html += `
                <div class="ingredient">
                    <div class="ingredient-name">
                        ${{key.toUpperCase()}}
                    </div>
                    <div class="missing">
                        Data unavailable
                    </div>
                </div>
            `;
            continue;
        }}

        html += `
            <div class="ingredient">
                <div class="ingredient-name">
                    ${{esc(item.label)}}
                </div>

                <div class="ingredient-value">
                    ${{item.value === null
                        ? "—"
                        : fmt(item.value)}}
                </div>

                <div class="ingredient-unit">
                    ${{esc(item.unit)}}
                </div>
            </div>
        `;
    }}

    container.innerHTML = html;

    document.getElementById("pollutionStatus").textContent =
        DATA.pollution?.available
        ? "AVAILABLE"
        : "NOT AVAILABLE";
}}


function renderBlocks() {{

    const container =
        document.getElementById("blocks");

    const blocks =
        DATA.summary.blocks || [];

    container.innerHTML = blocks.map(block => `

        <div class="block">

            <div class="block-number">
                Forecast block ${{block.block}}
            </div>

            <div class="block-aqi">
                ${{fmt(block.mean)}}
            </div>

            <div>
                <span class="category-pill
                    ${{categoryClass(block.category)}}">
                    ${{esc(block.category)}}
                </span>
            </div>

            <div style="margin-top:12px">

                <div class="stat-row">
                    <span class="stat-label">Minimum</span>
                    <strong>${{fmt(block.min)}}</strong>
                </div>

                <div class="stat-row">
                    <span class="stat-label">Maximum</span>
                    <strong>${{fmt(block.max)}}</strong>
                </div>

                <div class="stat-row">
                    <span class="stat-label">Start</span>
                    <strong>${{fmtDate(block.start)}}</strong>
                </div>

            </div>

        </div>

    `).join("");
}}


function renderDistribution(targetId, distribution) {{

    const container =
        document.getElementById(targetId);

    const entries =
        Object.entries(distribution || {{}})
            .sort((a,b) => b[1] - a[1]);

    if (!entries.length) {{
        container.innerHTML =
            '<div class="missing">No distribution data.</div>';
        return;
    }}

    const total =
        entries.reduce(
            (sum, item) => sum + Number(item[1]),
            0
        );

    container.innerHTML = entries.map(
        ([name, count]) => {{

            const percentage =
                total
                    ? (Number(count) / total) * 100
                    : 0;

            return `
                <div style="margin-bottom:15px">

                    <div class="stat-row">
                        <span>
                            <span class="category-pill
                                ${{categoryClass(name)}}">
                                ${{esc(name)}}
                            </span>
                        </span>

                        <strong>
                            ${{count}} h
                        </strong>
                    </div>

                    <div style="
                        height:9px;
                        background:#eef2f7;
                        border-radius:999px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:${{percentage}}%;
                            height:100%;
                            background:#2563eb;
                        "></div>

                    </div>

                </div>
            `;
        }}
    ).join("");
}}


function drawForecast() {{

    const canvas =
        document.getElementById("forecastCanvas");

    const ctx = canvas.getContext("2d");

    const rect =
        canvas.getBoundingClientRect();

    const dpr =
        window.devicePixelRatio || 1;

    canvas.width =
        rect.width * dpr;

    canvas.height =
        380 * dpr;

    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = 380;

    const padding = {{
        left: 48,
        right: 20,
        top: 25,
        bottom: 45
    }};

    const values =
        DATA.forecast.map(
            row => Number(row.predicted_aqi)
        );

    if (!values.length)
        return;

    const min =
        Math.min(...values);

    const max =
        Math.max(...values);

    const range =
        Math.max(1, max - min);

    const plotWidth =
        width - padding.left - padding.right;

    const plotHeight =
        height - padding.top - padding.bottom;


    // background
    ctx.clearRect(0, 0, width, height);


    // AQI bands
    const bands = [
        [0, 50],
        [50, 100],
        [100, 150],
        [150, 200],
        [200, 300],
        [300, 500]
    ];

    const bandNames = [
        "Good",
        "Moderate",
        "USG",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ];

    for (let i = 0; i < bands.length; i++) {{

        const lo = bands[i][0];
        const hi = bands[i][1];

        const y1 =
            padding.top +
            plotHeight -
            ((lo - min) / range) * plotHeight;

        const y2 =
            padding.top +
            plotHeight -
            ((hi - min) / range) * plotHeight;

        if (y1 < padding.top ||
            y2 > padding.top + plotHeight)
            continue;

        ctx.fillStyle =
            "rgba(100, 116, 139, 0.045)";

        ctx.fillRect(
            padding.left,
            Math.min(y1,y2),
            plotWidth,
            Math.abs(y2-y1)
        );
    }}


    // horizontal grid
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {{

        const y =
            padding.top +
            (plotHeight * i / 5);

        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        const value =
            max -
            (range * i / 5);

        ctx.fillStyle = "#6b7280";
        ctx.font = "11px system-ui";
        ctx.fillText(
            Math.round(value),
            8,
            y + 4
        );
    }}


    // line
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    ctx.beginPath();

    values.forEach((value, index) => {{

        const x =
            padding.left +
            (
                index /
                Math.max(1, values.length - 1)
            ) *
            plotWidth;

        const y =
            padding.top +
            plotHeight -
            (
                (value - min) /
                range
            ) *
            plotHeight;

        if (index === 0)
            ctx.moveTo(x,y);
        else
            ctx.lineTo(x,y);
    }});

    ctx.stroke();


    // points
    ctx.fillStyle = "#2563eb";

    values.forEach((value, index) => {{

        if (
            index !== 0 &&
            index !== values.length - 1 &&
            index % 6 !== 0
        )
            return;

        const x =
            padding.left +
            (
                index /
                Math.max(1, values.length - 1)
            ) *
            plotWidth;

        const y =
            padding.top +
            plotHeight -
            (
                (value - min) /
                range
            ) *
            plotHeight;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    }});


    // x labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "11px system-ui";

    const labelIndexes = [
        0,
        Math.floor(values.length / 4),
        Math.floor(values.length / 2),
        Math.floor(values.length * 3 / 4),
        values.length - 1
    ];

    labelIndexes.forEach(index => {{

        if (index < 0 || index >= DATA.forecast.length)
            return;

        const x =
            padding.left +
            (
                index /
                Math.max(1, values.length - 1)
            ) *
            plotWidth;

        const date =
            new Date(
                DATA.forecast[index].timestamp
            );

        const label =
            date.toLocaleString(
                [],
                {{
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit"
                }}
            );

        ctx.fillText(
            label,
            Math.max(0, x - 25),
            height - 12
        );
    }});
}}


function renderAssets() {{

    const container =
        document.getElementById("assets");

    const assets =
        DATA.assets || {{}};

    const order = [
        "forecast_png",
        "categories_png",
        "risk_png",
        "blocks_png"
    ];

    const labels = {{
        forecast_png: "72-Hour AQI Forecast",
        categories_png: "AQI Category Distribution",
        risk_png: "Risk Distribution",
        blocks_png: "24-Hour Forecast Blocks"
    }};

    let html = "";

    for (const key of order) {{

        if (!assets[key])
            continue;

        html += `
            <div style="margin-top:20px">
                <strong>${{labels[key]}}</strong>
                <img
                    class="asset"
                    src="${{esc(assets[key])}}"
                    alt="${{esc(labels[key])}}"
                >
            </div>
        `;
    }}

    if (!html)
        html =
            '<div class="missing">No existing infographic assets found.</div>';

    container.innerHTML = html;
}}


function renderTable() {{

    const tbody =
        document.getElementById("forecastTable");

    tbody.innerHTML =
        DATA.forecast.map(row => `

            <tr>

                <td>${{esc(row.horizon)}}</td>

                <td>${{fmtDate(row.timestamp)}}</td>

                <td><strong>
                    ${{fmt(row.predicted_aqi, 2)}}
                </strong></td>

                <td>
                    <span class="category-pill
                        ${{categoryClass(row.category)}}">
                        ${{esc(row.category)}}
                    </span>
                </td>

                <td>${{esc(row.risk)}}</td>

            </tr>

        `).join("");
}}


function renderStatus() {{

    document.getElementById("cacheStatus").textContent =
        DATA.cache?.status || "—";

    document.getElementById("lastUpdated").textContent =
        fmtDate(DATA.generated_at);
}}


function renderAll() {{

    renderLocation();
    renderKpis();
    renderRisk();
    renderIngredients();
    renderBlocks();

    renderDistribution(
        "categoryDistribution",
        DATA.summary.category_distribution
    );

    renderDistribution(
        "riskDistribution",
        DATA.summary.risk_distribution
    );

    renderAssets();
    renderTable();
    renderStatus();

    drawForecast();
}}


window.addEventListener(
    "resize",
    drawForecast
);

renderAll();

</script>

</body>
</html>
"""


# ============================================================================
# DATA PACKAGE
# ============================================================================

def build_dashboard_data(
    forecast: pd.DataFrame,
    pollution_metadata: Dict[str, Any],
    location: Dict[str, Any],
    assets: Dict[str, str],
    cache_status: str,
    forecast_source: Path,
    pollution_source: Optional[Path],
    model_name: str = "XGBoost tuned",
) -> Dict[str, Any]:

    summary = build_forecast_summary(
        forecast
    )

    forecast_records = []

    for _, row in forecast.iterrows():
        forecast_records.append({
            "timestamp": row["timestamp"].isoformat(),
            "horizon": (
                int(row["horizon"])
                if pd.notna(row["horizon"])
                else None
            ),
            "predicted_aqi": round(
                float(row["predicted_aqi"]),
                3,
            ),
            "category": str(row["category"]),
            "risk": str(row["risk"]),
        })

    data = {
        "dashboard_version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "project": {
            "name": "PEARLS AQI PREDICTOR",
            "target": "us_aqi",
            "forecast_horizon": 72,
            "model": model_name,
        },

        "location": location,

        "summary": summary,

        "forecast": forecast_records,

        "pollution": pollution_metadata,

        "assets": assets,

        "sources": {
            "forecast": str(forecast_source),
            "pollution": (
                str(pollution_source)
                if pollution_source
                else None
            ),
        },

        "cache": {
            "status": cache_status,
        },

        "limitations": [
            "Pollution ingredient cards use available observations "
            "from project data.",
            "Missing pollutant values are not inferred or fabricated.",
            "Location is shown only when available in project data.",
            "The dashboard does not perform model retraining.",
            "The dashboard does not perform model selection.",
            "The dashboard does not perform validation or test evaluation.",
        ],
    }

    return data


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_dashboard_outputs(
    forecast: pd.DataFrame,
    dashboard_data: Dict[str, Any],
) -> None:

    forecast.to_csv(
        DASHBOARD_FORECAST,
        index=False,
    )

    summary = build_forecast_summary(
        forecast
    )

    summary_rows = []

    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(
                value,
                ensure_ascii=False,
                default=json_default,
            )

        summary_rows.append({
            "metric": key,
            "value": value,
        })

    pd.DataFrame(summary_rows).to_csv(
        DASHBOARD_SUMMARY,
        index=False,
    )

    with DASHBOARD_DATA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            dashboard_data,
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

    html = build_dashboard_html(
        dashboard_data
    )

    with DASHBOARD_HTML.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(html)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_outputs() -> Dict[str, bool]:
    results = {}

    results["dashboard_html"] = (
        DASHBOARD_HTML.exists()
        and DASHBOARD_HTML.stat().st_size > 1000
    )

    results["dashboard_data"] = (
        DASHBOARD_DATA.exists()
        and DASHBOARD_DATA.stat().st_size > 100
    )

    results["dashboard_forecast"] = (
        DASHBOARD_FORECAST.exists()
        and DASHBOARD_FORECAST.stat().st_size > 50
    )

    results["dashboard_summary"] = (
        DASHBOARD_SUMMARY.exists()
        and DASHBOARD_SUMMARY.stat().st_size > 20
    )

    results["cache_hash"] = (
        CACHE_HASH_FILE.exists()
    )

    # Check that generated HTML contains important dashboard sections.
    if DASHBOARD_HTML.exists():
        try:
            html = DASHBOARD_HTML.read_text(
                encoding="utf-8"
            ).lower()

            required_terms = [
                "pollution ingredients",
                "72-hour aqi forecast",
                "forecast windows",
                "hourly forecast",
                "aqi health indicator",
            ]

            results["dashboard_content"] = all(
                term in html
                for term in required_terms
            )

        except Exception:
            results["dashboard_content"] = False

    else:
        results["dashboard_content"] = False

    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    started = datetime.now(
        timezone.utc
    )

    print()
    line()
    print("PEARLS AQI PREDICTOR")
    line()
    print("STEP 20 — PRODUCTION DASHBOARD")
    line()
    info("Base directory", BASE_DIR)
    info("Target", "us_aqi")
    info("Forecast horizon", 72)
    info("Dashboard type", "Interactive + Infographic")
    info("Caching", "Enabled")
    info("Model training", "NOT performed")
    info("Model selection", "NOT performed")
    info("Validation/test", "NOT performed")

    try:

        ensure_directories()

        # ---------------------------------------------------------------
        # Forecast
        # ---------------------------------------------------------------
        heading("LOADING PRODUCTION FORECAST")

        forecast, forecast_source = load_forecast()

        info("Forecast rows", len(forecast))
        info(
            "Forecast start",
            forecast["timestamp"].min().isoformat(),
        )
        info(
            "Forecast end",
            forecast["timestamp"].max().isoformat(),
        )

        # ---------------------------------------------------------------
        # Step 18 / Step 19 metadata
        # ---------------------------------------------------------------
        heading("LOADING EXISTING DASHBOARD METADATA")

        step18_data = load_json_if_exists(
            STEP18_DATA
        )

        step19_package = load_json_if_exists(
            STEP19_PACKAGE
        )

        info(
            "Step 18 dashboard data",
            "FOUND" if step18_data else "NOT FOUND",
        )

        info(
            "Step 19 dashboard package",
            "FOUND" if step19_package else "NOT FOUND",
        )

        # ---------------------------------------------------------------
        # Pollution source
        # ---------------------------------------------------------------
        heading("DISCOVERING POLLUTION INGREDIENT DATA")

        (
            pollution_source,
            pollutant_columns,
            location_columns,
        ) = discover_pollution_source(
            forecast_source
        )

        pollution_df, pollution_metadata = (
            load_pollution_data(
                pollution_source,
                pollutant_columns,
                location_columns,
            )
        )

        info(
            "Pollution source",
            pollution_source
            if pollution_source
            else "NOT FOUND",
        )

        info(
            "PM2.5",
            (
                "FOUND"
                if "pm25"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        info(
            "PM10",
            (
                "FOUND"
                if "pm10"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        info(
            "O3",
            (
                "FOUND"
                if "o3"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        info(
            "NO2",
            (
                "FOUND"
                if "no2"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        info(
            "SO2",
            (
                "FOUND"
                if "so2"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        info(
            "CO",
            (
                "FOUND"
                if "co"
                in pollution_metadata.get(
                    "pollutants",
                    {},
                )
                else "NOT FOUND"
            ),
        )

        # ---------------------------------------------------------------
        # Location
        # ---------------------------------------------------------------
        heading("BUILDING LOCATION CONTEXT")

        location = pollution_metadata.get(
            "location",
            {},
        )

        location = enrich_location(
            location,
            step18_data,
            step19_package,
        )

        if location:
            info("Location metadata", location)
        else:
            info(
                "Location metadata",
                "NOT AVAILABLE",
            )

        # ---------------------------------------------------------------
        # Existing visualizations
        # ---------------------------------------------------------------
        heading("IMPORTING STEP 18 INFOGRAPHICS")

        assets = copy_existing_visualizations()

        info(
            "Existing visualization assets",
            len(assets),
        )

        for key, value in assets.items():
            info(key, value)

        # ---------------------------------------------------------------
        # Cache
        # ---------------------------------------------------------------
        heading("CHECKING DASHBOARD CACHE")

        cache_key = build_cache_key(
            forecast,
            pollution_source,
            location,
        )

        cache_hit = check_cache(
            cache_key
        )

        if cache_hit:
            cache_status = "HIT"
            print(
                "Cache status                     : HIT"
            )
        else:
            cache_status = "MISS"
            print(
                "Cache status                     : MISS"
            )

        # We still regenerate the HTML every run because asset paths,
        # timestamps and source state are cheap to build and make the
        # dashboard deterministic. The cache records whether underlying
        # data changed.
        #
        # This provides useful production behavior without making the
        # dashboard stale merely because the cache was hit.

        # ---------------------------------------------------------------
        # Build data
        # ---------------------------------------------------------------
        heading("BUILDING DASHBOARD DATA PACKAGE")

        dashboard_data = build_dashboard_data(
            forecast=forecast,
            pollution_metadata=pollution_metadata,
            location=location,
            assets=assets,
            cache_status=cache_status,
            forecast_source=forecast_source,
            pollution_source=pollution_source,
        )

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------
        heading("CREATING PRODUCTION DASHBOARD")

        save_dashboard_outputs(
            forecast,
            dashboard_data,
        )

        save_cache_key(
            cache_key
        )

        info(
            "Dashboard HTML",
            DASHBOARD_HTML,
        )

        info(
            "Dashboard data",
            DASHBOARD_DATA,
        )

        info(
            "Dashboard forecast",
            DASHBOARD_FORECAST,
        )

        info(
            "Dashboard summary",
            DASHBOARD_SUMMARY,
        )

        info(
            "Cache",
            CACHE_HASH_FILE,
        )

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------
        heading("VALIDATING DASHBOARD")

        validation = validate_outputs()

        for name, passed in validation.items():
            print(
                f"{name:<32}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        all_passed = all(
            validation.values()
        )

        # ---------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------
        heading("SAVING STEP 20 REPORT")

        finished = datetime.now(
            timezone.utc
        )

        report = {
            "step": 20,
            "name": "production_dashboard",
            "status": (
                "PASS"
                if all_passed
                else "FAIL"
            ),
            "generated_at": finished.isoformat(),
            "execution_seconds": (
                finished - started
            ).total_seconds(),
            "forecast_source": str(
                forecast_source
            ),
            "pollution_source": (
                str(pollution_source)
                if pollution_source
                else None
            ),
            "forecast_rows": int(
                len(forecast)
            ),
            "forecast_start": forecast[
                "timestamp"
            ].min().isoformat(),
            "forecast_end": forecast[
                "timestamp"
            ].max().isoformat(),
            "mean_aqi": round(
                float(
                    forecast[
                        "predicted_aqi"
                    ].mean()
                ),
                3,
            ),
            "maximum_aqi": round(
                float(
                    forecast[
                        "predicted_aqi"
                    ].max()
                ),
                3,
            ),
            "minimum_aqi": round(
                float(
                    forecast[
                        "predicted_aqi"
                    ].min()
                ),
                3,
            ),
            "dominant_category":
                build_forecast_summary(
                    forecast
                )["dominant_category"],
            "pollutants_available":
                list(
                    pollution_metadata.get(
                        "pollutants",
                        {},
                    ).keys()
                ),
            "location":
                location,
            "cache_status":
                cache_status,
            "assets":
                assets,
            "validation":
                validation,
            "model_selection":
                False,
            "hyperparameter_tuning":
                False,
            "model_retraining":
                False,
            "validation_used":
                False,
            "test_used":
                False,
            "future_target_leakage":
                "NONE",
        }

        with DASHBOARD_REPORT.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )

        info(
            "Step 20 report",
            DASHBOARD_REPORT,
        )

        # ---------------------------------------------------------------
        # Final
        # ---------------------------------------------------------------
        print()
        line()
        print(
            "STEP 20 COMPLETE"
            if all_passed
            else "STEP 20 FAILED"
        )
        line()

        info(
            "Dashboard status",
            "PASS" if all_passed else "FAIL",
        )

        info(
            "Dashboard HTML",
            DASHBOARD_HTML,
        )

        info(
            "Pollution ingredients",
            (
                f"{len(pollution_metadata.get('pollutants', {}))}"
                " available"
            ),
        )

        info(
            "Location metadata",
            (
                "AVAILABLE"
                if location
                else "NOT AVAILABLE"
            ),
        )

        info(
            "Cache status",
            cache_status,
        )

        info(
            "Execution time",
            f"{report['execution_seconds']:.3f}s",
        )

        return 0 if all_passed else 1

    except Exception as exc:

        print()
        line()
        print("STEP 20 FAILED")
        line()
        print(f"{type(exc).__name__}: {exc}")

        return 1


if __name__ == "__main__":
    sys.exit(main())