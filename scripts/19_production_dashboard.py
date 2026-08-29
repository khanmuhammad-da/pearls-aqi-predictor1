"""
PEARLS AQI PREDICTOR
STEP 19 — PRODUCTION DASHBOARD PACKAGE

Purpose:
    Build the final production dashboard package from Step 18 outputs.

Important:
    Step 19 does NOT perform:
        - model selection
        - model training
        - hyperparameter tuning
        - validation
        - test-set selection
        - prediction generation

    It packages and validates the already-generated Step 18
    production forecast and visualization artifacts.

Current production deployment:
    XGBoost tuned

Primary input:
    reports/dashboard_ready_forecast.csv

Secondary input:
    reports/visualizations/dashboard_data.json

Outputs:
    reports/production_dashboard/
        production_dashboard_forecast.csv
        production_dashboard_hourly.csv
        production_dashboard_summary.csv
        production_dashboard.html
        production_dashboard_package.json

    reports/production_dashboard_results.json
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATION_DIR = REPORTS_DIR / "visualizations"
DASHBOARD_DIR = REPORTS_DIR / "production_dashboard"

DASHBOARD_FORECAST = REPORTS_DIR / "dashboard_ready_forecast.csv"
DASHBOARD_DATA = VISUALIZATION_DIR / "dashboard_data.json"
VISUALIZATION_REPORT = REPORTS_DIR / "production_visualization_results.json"

FORECAST_PNG = VISUALIZATION_DIR / "production_72h_forecast.png"
CATEGORY_PNG = VISUALIZATION_DIR / "production_72h_categories.png"
RISK_PNG = VISUALIZATION_DIR / "production_72h_risk.png"
BLOCKS_PNG = VISUALIZATION_DIR / "production_24h_blocks.png"

OUTPUT_FORECAST = DASHBOARD_DIR / "production_dashboard_forecast.csv"
OUTPUT_HOURLY = DASHBOARD_DIR / "production_dashboard_hourly.csv"
OUTPUT_SUMMARY = DASHBOARD_DIR / "production_dashboard_summary.csv"
OUTPUT_HTML = DASHBOARD_DIR / "production_dashboard.html"
OUTPUT_PACKAGE = DASHBOARD_DIR / "production_dashboard_package.json"

STEP19_REPORT = REPORTS_DIR / "production_dashboard_results.json"

TARGET = "us_aqi"
EXPECTED_HORIZON = 72
EXPECTED_FEATURES = 101
MODEL = "XGBoost tuned"


# ============================================================================
# CONSOLE HELPERS
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def status_line(label: str, value) -> None:
    print(f"{label:<32}: {value}")


def fail(message: str) -> None:
    banner("STEP 19 FAILED")
    print(message)
    sys.exit(1)


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def json_safe(value):
    """
    Convert pandas / numpy / datetime values into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return str(value)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# AQI CLASSIFICATION
# ============================================================================

def classify_aqi(aqi: float) -> str:
    """
    US AQI category boundaries.
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


def risk_level(aqi: float) -> str:
    if aqi >= 301:
        return "Hazardous"

    if aqi >= 201:
        return "Very High"

    if aqi >= 151:
        return "High"

    if aqi >= 101:
        return "Elevated"

    if aqi >= 51:
        return "Moderate"

    return "Low"


# ============================================================================
# COLUMN RESOLUTION
# ============================================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================================
# LOAD STEP 18
# ============================================================================

def load_step18_inputs():
    banner("LOADING STEP 18 DASHBOARD OUTPUTS")

    required = {
        "dashboard_ready_forecast": DASHBOARD_FORECAST,
        "dashboard_data": DASHBOARD_DATA,
        "visualization_report": VISUALIZATION_REPORT,
    }

    for name, path in required.items():
        if path.exists():
            status_line(name, f"FOUND -> {path}")
        else:
            fail(
                f"Required Step 18 output not found:\n"
                f"{path}\n\n"
                f"Run Step 18 first."
            )

    return


# ============================================================================
# LOAD FORECAST
# ============================================================================

def load_forecast() -> pd.DataFrame:
    banner("LOADING DASHBOARD-READY FORECAST")

    try:
        df = pd.read_csv(DASHBOARD_FORECAST)
    except Exception as exc:
        fail(f"Could not read dashboard forecast CSV:\n{exc}")

    status_line("Forecast rows", len(df))
    status_line("Forecast columns", len(df.columns))

    if df.empty:
        fail("Dashboard forecast CSV is empty.")

    return df


# ============================================================================
# NORMALIZE FORECAST
# ============================================================================

def normalize_forecast(df: pd.DataFrame) -> pd.DataFrame:
    banner("NORMALIZING FORECAST STRUCTURE")

    timestamp_col = find_column(
        df,
        [
            "timestamp",
            "forecast_timestamp",
            "datetime",
            "date",
            "time",
        ],
    )

    horizon_col = find_column(
        df,
        [
            "horizon",
            "forecast_horizon",
            "horizon_hours",
        ],
    )

    prediction_col = find_column(
        df,
        [
            "predicted_aqi",
            "prediction",
            "predicted_us_aqi",
            "us_aqi_prediction",
            "forecast_aqi",
            "us_aqi",
        ],
    )

    if timestamp_col is None:
        fail(
            "Could not locate forecast timestamp column.\n"
            f"Available columns: {list(df.columns)}"
        )

    if prediction_col is None:
        fail(
            "Could not locate predicted AQI column.\n"
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result[timestamp_col],
        errors="coerce",
        utc=True,
    )

    result["predicted_aqi"] = pd.to_numeric(
        result[prediction_col],
        errors="coerce",
    )

    if horizon_col is not None:
        result["horizon"] = pd.to_numeric(
            result[horizon_col],
            errors="coerce",
        )
    else:
        result = result.sort_values("timestamp").reset_index(drop=True)
        result["horizon"] = range(1, len(result) + 1)

    result = result.sort_values("horizon").reset_index(drop=True)

    # AQI classification is regenerated here so the dashboard has
    # a deterministic classification regardless of Step 18 column names.
    result["aqi_category"] = result["predicted_aqi"].apply(
        lambda x: classify_aqi(float(x))
        if pd.notna(x)
        else None
    )

    result["risk_level"] = result["predicted_aqi"].apply(
        lambda x: risk_level(float(x))
        if pd.notna(x)
        else None
    )

    return result


# ============================================================================
# VALIDATE FORECAST
# ============================================================================

def validate_forecast(df: pd.DataFrame) -> None:
    banner("VALIDATING PRODUCTION FORECAST")

    row_count = len(df)

    status_line("Forecast rows", row_count)
    status_line("Expected rows", EXPECTED_HORIZON)

    if row_count != EXPECTED_HORIZON:
        fail(
            f"Expected exactly {EXPECTED_HORIZON} forecast rows, "
            f"found {row_count}."
        )

    horizons = df["horizon"].tolist()
    expected_horizons = list(range(1, EXPECTED_HORIZON + 1))

    horizon_pass = horizons == expected_horizons

    status_line(
        "Horizon sequence",
        "PASS" if horizon_pass else "FAIL",
    )

    if not horizon_pass:
        fail(
            "Forecast horizon sequence is invalid.\n"
            f"Expected: 1 -> {EXPECTED_HORIZON}\n"
            f"Found: {horizons}"
        )

    duplicate_timestamps = int(df["timestamp"].duplicated().sum())

    status_line(
        "Duplicate timestamps",
        duplicate_timestamps,
    )

    if duplicate_timestamps:
        fail("Duplicate forecast timestamps detected.")

    timestamp_deltas = df["timestamp"].diff().dropna()

    continuity_pass = (
        len(timestamp_deltas) == EXPECTED_HORIZON - 1
        and all(timestamp_deltas == pd.Timedelta(hours=1))
    )

    status_line(
        "Timestamp continuity",
        "PASS" if continuity_pass else "FAIL",
    )

    if not continuity_pass:
        fail("Forecast timestamps are not continuous hourly timestamps.")

    finite_predictions = bool(
        df["predicted_aqi"].notna().all()
        and df["predicted_aqi"].map(math.isfinite).all()
    )

    status_line(
        "Prediction values finite",
        "PASS" if finite_predictions else "FAIL",
    )

    if not finite_predictions:
        fail("Forecast contains NaN or infinite AQI values.")

    negative_predictions = int(
        (df["predicted_aqi"] < 0).sum()
    )

    status_line(
        "Negative AQI values",
        negative_predictions,
    )

    if negative_predictions:
        fail("Negative AQI predictions detected.")

    status_line("Forecast validation", "PASS")


# ============================================================================
# BUILD SUMMARY
# ============================================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    banner("BUILDING DASHBOARD SUMMARY")

    counts = (
        df["aqi_category"]
        .value_counts()
        .reindex(
            [
                "Good",
                "Moderate",
                "Unhealthy for Sensitive Groups",
                "Unhealthy",
                "Very Unhealthy",
                "Hazardous",
            ],
            fill_value=0,
        )
    )

    summary_rows = []

    minimum = float(df["predicted_aqi"].min())
    maximum = float(df["predicted_aqi"].max())
    mean = float(df["predicted_aqi"].mean())
    median = float(df["predicted_aqi"].median())
    std = float(df["predicted_aqi"].std(ddof=0))

    max_row = df.loc[df["predicted_aqi"].idxmax()]
    min_row = df.loc[df["predicted_aqi"].idxmin()]

    summary_rows.extend(
        [
            {
                "metric": "model",
                "value": MODEL,
            },
            {
                "metric": "target",
                "value": TARGET,
            },
            {
                "metric": "forecast_horizon_hours",
                "value": EXPECTED_HORIZON,
            },
            {
                "metric": "forecast_start",
                "value": df["timestamp"].min().isoformat(),
            },
            {
                "metric": "forecast_end",
                "value": df["timestamp"].max().isoformat(),
            },
            {
                "metric": "minimum_predicted_aqi",
                "value": minimum,
            },
            {
                "metric": "maximum_predicted_aqi",
                "value": maximum,
            },
            {
                "metric": "mean_predicted_aqi",
                "value": mean,
            },
            {
                "metric": "median_predicted_aqi",
                "value": median,
            },
            {
                "metric": "aqi_standard_deviation",
                "value": std,
            },
            {
                "metric": "highest_horizon",
                "value": int(max_row["horizon"]),
            },
            {
                "metric": "highest_category",
                "value": max_row["aqi_category"],
            },
            {
                "metric": "lowest_horizon",
                "value": int(min_row["horizon"]),
            },
            {
                "metric": "lowest_category",
                "value": min_row["aqi_category"],
            },
            {
                "metric": "hours_aqi_ge_101",
                "value": int((df["predicted_aqi"] >= 101).sum()),
            },
            {
                "metric": "hours_aqi_ge_151",
                "value": int((df["predicted_aqi"] >= 151).sum()),
            },
            {
                "metric": "hours_aqi_ge_201",
                "value": int((df["predicted_aqi"] >= 201).sum()),
            },
            {
                "metric": "hours_aqi_ge_301",
                "value": int((df["predicted_aqi"] >= 301).sum()),
            },
        ]
    )

    for category, count in counts.items():
        summary_rows.append(
            {
                "metric": f"hours_{category.lower().replace(' ', '_')}",
                "value": int(count),
            }
        )

    return pd.DataFrame(summary_rows)


# ============================================================================
# CREATE HTML
# ============================================================================

def create_html(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    banner("CREATING PRODUCTION DASHBOARD HTML")

    forecast_start = df["timestamp"].min()
    forecast_end = df["timestamp"].max()

    minimum = float(df["predicted_aqi"].min())
    maximum = float(df["predicted_aqi"].max())
    mean = float(df["predicted_aqi"].mean())
    median = float(df["predicted_aqi"].median())

    dominant_category = (
        df["aqi_category"]
        .value_counts()
        .idxmax()
    )

    rows_html = ""

    for _, row in df.iterrows():
        rows_html += f"""
        <tr>
            <td>{int(row['horizon'])}</td>
            <td>{row['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}</td>
            <td>{float(row['predicted_aqi']):.3f}</td>
            <td>{row['aqi_category']}</td>
            <td>{row['risk_level']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pearls AQI Predictor — Production Dashboard</title>

<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 0;
    padding: 0;
    background: #f4f6f8;
    color: #202124;
}}

header {{
    background: #202124;
    color: white;
    padding: 24px 32px;
}}

header h1 {{
    margin: 0 0 8px 0;
}}

header p {{
    margin: 4px 0;
}}

.container {{
    max-width: 1400px;
    margin: 24px auto;
    padding: 0 20px;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}

.card {{
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.card .label {{
    font-size: 13px;
    color: #666;
    margin-bottom: 8px;
}}

.card .value {{
    font-size: 28px;
    font-weight: bold;
}}

.panel {{
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.panel h2 {{
    margin-top: 0;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 9px 10px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}}

th {{
    background: #f0f2f4;
}}

.badge {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 5px;
    font-size: 12px;
}}

footer {{
    color: #666;
    font-size: 12px;
    padding: 20px 0 40px;
}}
</style>
</head>

<body>

<header>
    <h1>PEARLS AQI PREDICTOR</h1>
    <p>Production AQI Forecast Dashboard</p>
    <p>Model: <strong>{MODEL}</strong></p>
    <p>Target: <strong>{TARGET}</strong></p>
</header>

<div class="container">

<div class="cards">

    <div class="card">
        <div class="label">Forecast Horizon</div>
        <div class="value">72 h</div>
    </div>

    <div class="card">
        <div class="label">Minimum AQI</div>
        <div class="value">{minimum:.1f}</div>
    </div>

    <div class="card">
        <div class="label">Maximum AQI</div>
        <div class="value">{maximum:.1f}</div>
    </div>

    <div class="card">
        <div class="label">Mean AQI</div>
        <div class="value">{mean:.1f}</div>
    </div>

    <div class="card">
        <div class="label">Median AQI</div>
        <div class="value">{median:.1f}</div>
    </div>

    <div class="card">
        <div class="label">Dominant Category</div>
        <div class="value">{dominant_category}</div>
    </div>

</div>

<div class="panel">
    <h2>Forecast Window</h2>
    <p>
        <strong>Start:</strong>
        {forecast_start.strftime('%Y-%m-%d %H:%M UTC')}
    </p>
    <p>
        <strong>End:</strong>
        {forecast_end.strftime('%Y-%m-%d %H:%M UTC')}
    </p>
</div>

<div class="panel">
    <h2>72-Hour AQI Forecast</h2>
    <img src="../visualizations/production_72h_forecast.png"
         alt="72-hour AQI forecast">
</div>

<div class="panel">
    <h2>AQI Category Distribution</h2>
    <img src="../visualizations/production_72h_categories.png"
         alt="AQI category distribution">
</div>

<div class="panel">
    <h2>AQI Risk</h2>
    <img src="../visualizations/production_72h_risk.png"
         alt="AQI risk visualization">
</div>

<div class="panel">
    <h2>24-Hour Blocks</h2>
    <img src="../visualizations/production_24h_blocks.png"
         alt="24-hour AQI blocks">
</div>

<div class="panel">
    <h2>Hourly Forecast</h2>

    <table>
        <thead>
            <tr>
                <th>Horizon</th>
                <th>Timestamp</th>
                <th>Predicted AQI</th>
                <th>Category</th>
                <th>Risk</th>
            </tr>
        </thead>

        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>

<footer>
    PEARLS AQI PREDICTOR — Step 19 Production Dashboard Package<br>
    Model selection: NOT performed in Step 19<br>
    Validation: NOT performed in Step 19<br>
    Test-set selection: NOT performed in Step 19
</footer>

</div>

</body>
</html>
"""

    OUTPUT_HTML.write_text(
        html,
        encoding="utf-8",
    )

    status_line(
        "Dashboard HTML",
        f"SAVED -> {OUTPUT_HTML}",
    )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict:
    banner("SAVING PRODUCTION DASHBOARD PACKAGE")

    DASHBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Full dashboard forecast
    # ------------------------------------------------------------------------

    dashboard_columns = [
        "horizon",
        "timestamp",
        "predicted_aqi",
        "aqi_category",
        "risk_level",
    ]

    dashboard_df = df[dashboard_columns].copy()

    dashboard_df["timestamp"] = dashboard_df[
        "timestamp"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    dashboard_df.to_csv(
        OUTPUT_FORECAST,
        index=False,
    )

    status_line(
        "Dashboard forecast CSV",
        f"SAVED -> {OUTPUT_FORECAST}",
    )

    # ------------------------------------------------------------------------
    # Hourly output
    # ------------------------------------------------------------------------

    hourly = dashboard_df.copy()

    hourly.to_csv(
        OUTPUT_HOURLY,
        index=False,
    )

    status_line(
        "Hourly dashboard CSV",
        f"SAVED -> {OUTPUT_HOURLY}",
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    status_line(
        "Dashboard summary CSV",
        f"SAVED -> {OUTPUT_SUMMARY}",
    )

    # ------------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------------

    create_html(
        df,
        summary,
    )

    # ------------------------------------------------------------------------
    # Package manifest
    # ------------------------------------------------------------------------

    package = {
        "project": "PEARLS AQI PREDICTOR",
        "step": 19,
        "step_name": "PRODUCTION DASHBOARD PACKAGE",
        "status": "SUCCESS",
        "model": MODEL,
        "target": TARGET,
        "forecast_horizon": EXPECTED_HORIZON,
        "features_used": EXPECTED_FEATURES,
        "model_selection": False,
        "hyperparameter_tuning": False,
        "model_retraining": False,
        "validation": False,
        "test_set_usage": False,
        "future_target_leakage": "NONE",
        "inputs": {
            "dashboard_ready_forecast": str(
                DASHBOARD_FORECAST
            ),
            "dashboard_data": str(
                DASHBOARD_DATA
            ),
            "visualization_report": str(
                VISUALIZATION_REPORT
            ),
        },
        "visualizations": {
            "forecast": str(FORECAST_PNG),
            "categories": str(CATEGORY_PNG),
            "risk": str(RISK_PNG),
            "24_hour_blocks": str(BLOCKS_PNG),
        },
        "outputs": {
            "forecast": str(OUTPUT_FORECAST),
            "hourly": str(OUTPUT_HOURLY),
            "summary": str(OUTPUT_SUMMARY),
            "html": str(OUTPUT_HTML),
            "package": str(OUTPUT_PACKAGE),
        },
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_json(
        OUTPUT_PACKAGE,
        package,
    )

    status_line(
        "Dashboard package JSON",
        f"SAVED -> {OUTPUT_PACKAGE}",
    )

    return package


# ============================================================================
# VALIDATE PACKAGE
# ============================================================================

def validate_package() -> None:
    banner("VALIDATING DASHBOARD PACKAGE")

    required_outputs = {
        "production_dashboard_forecast": OUTPUT_FORECAST,
        "production_dashboard_hourly": OUTPUT_HOURLY,
        "production_dashboard_summary": OUTPUT_SUMMARY,
        "production_dashboard_html": OUTPUT_HTML,
        "production_dashboard_package": OUTPUT_PACKAGE,
    }

    for name, path in required_outputs.items():
        exists = path.exists()

        status_line(
            name,
            "PASS" if exists else "FAIL",
        )

        if not exists:
            fail(
                f"Required dashboard output missing:\n{path}"
            )

        if path.is_file() and path.stat().st_size == 0:
            fail(
                f"Dashboard output is empty:\n{path}"
            )

    visualization_files = {
        "forecast_png": FORECAST_PNG,
        "categories_png": CATEGORY_PNG,
        "risk_png": RISK_PNG,
        "blocks_png": BLOCKS_PNG,
    }

    for name, path in visualization_files.items():
        exists = path.exists()

        status_line(
            name,
            "PASS" if exists else "FAIL",
        )

        if not exists:
            fail(
                f"Required Step 18 visualization missing:\n{path}"
            )

    # Validate generated forecast again.
    generated = pd.read_csv(
        OUTPUT_FORECAST
    )

    if len(generated) != EXPECTED_HORIZON:
        fail(
            "Generated dashboard forecast does not contain "
            f"{EXPECTED_HORIZON} rows."
        )

    required_columns = {
        "horizon",
        "timestamp",
        "predicted_aqi",
        "aqi_category",
        "risk_level",
    }

    missing_columns = required_columns.difference(
        generated.columns
    )

    if missing_columns:
        fail(
            "Generated dashboard forecast is missing columns: "
            f"{sorted(missing_columns)}"
        )

    status_line(
        "Dashboard package validation",
        "PASS",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    start_time = datetime.now(
        timezone.utc
    )

    banner("PEARLS AQI PREDICTOR")
    print("STEP 19 — PRODUCTION DASHBOARD PACKAGE")
    print("=" * 70)

    status_line(
        "Base directory",
        BASE_DIR,
    )

    status_line(
        "Target",
        TARGET,
    )

    status_line(
        "Forecast horizon",
        EXPECTED_HORIZON,
    )

    status_line(
        "Production model",
        MODEL,
    )

    status_line(
        "Model selection",
        "NOT performed",
    )

    status_line(
        "Validation/test use",
        "NOT performed",
    )

    # ------------------------------------------------------------------------
    # Step 18 is the source of truth for the dashboard package.
    # ------------------------------------------------------------------------

    load_step18_inputs()

    raw_df = load_forecast()

    df = normalize_forecast(
        raw_df
    )

    validate_forecast(
        df
    )

    summary = build_summary(
        df
    )

    package = save_outputs(
        df,
        summary,
    )

    validate_package()

    end_time = datetime.now(
        timezone.utc
    )

    execution_seconds = (
        end_time - start_time
    ).total_seconds()

    # ------------------------------------------------------------------------
    # Final Step 19 report
    # ------------------------------------------------------------------------

    banner("SAVING STEP 19 REPORT")

    dominant_category = (
        df["aqi_category"]
        .value_counts()
        .idxmax()
    )

    result = {
        "project": "PEARLS AQI PREDICTOR",
        "step": 19,
        "step_name": "PRODUCTION DASHBOARD PACKAGE",
        "status": "SUCCESS",
        "model": MODEL,
        "target": TARGET,
        "forecast_horizon": EXPECTED_HORIZON,
        "forecast_start": df["timestamp"].min(),
        "forecast_end": df["timestamp"].max(),
        "minimum_predicted_aqi": float(
            df["predicted_aqi"].min()
        ),
        "maximum_predicted_aqi": float(
            df["predicted_aqi"].max()
        ),
        "mean_predicted_aqi": float(
            df["predicted_aqi"].mean()
        ),
        "median_predicted_aqi": float(
            df["predicted_aqi"].median()
        ),
        "dominant_aqi_category": dominant_category,
        "model_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "model_retraining_performed": False,
        "validation_used": False,
        "test_used": False,
        "future_target_leakage": "NONE",
        "dashboard_package_validation": "PASS",
        "dashboard_html": str(
            OUTPUT_HTML
        ),
        "dashboard_forecast": str(
            OUTPUT_FORECAST
        ),
        "dashboard_hourly": str(
            OUTPUT_HOURLY
        ),
        "dashboard_summary": str(
            OUTPUT_SUMMARY
        ),
        "dashboard_package": str(
            OUTPUT_PACKAGE
        ),
        "execution_time_seconds": execution_seconds,
        "created_at": end_time,
    }

    save_json(
        STEP19_REPORT,
        result,
    )

    status_line(
        "Step 19 report",
        f"SAVED -> {STEP19_REPORT}",
    )

    # ------------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------------

    banner("STEP 19 COMPLETE")

    print(f"Production model                : {MODEL}")
    print(f"Forecast horizons               : {EXPECTED_HORIZON}")
    print(
        f"Forecast start                  : "
        f"{df['timestamp'].min().isoformat()}"
    )
    print(
        f"Forecast end                    : "
        f"{df['timestamp'].max().isoformat()}"
    )
    print(
        f"Minimum predicted AQI           : "
        f"{df['predicted_aqi'].min():.3f}"
    )
    print(
        f"Maximum predicted AQI           : "
        f"{df['predicted_aqi'].max():.3f}"
    )
    print(
        f"Mean predicted AQI              : "
        f"{df['predicted_aqi'].mean():.3f}"
    )
    print(
        f"Median predicted AQI            : "
        f"{df['predicted_aqi'].median():.3f}"
    )
    print(
        f"Dominant AQI category           : "
        f"{dominant_category}"
    )

    print()
    print("Model selection performed       : FALSE")
    print("Hyperparameter tuning performed : FALSE")
    print("Model retraining performed      : FALSE")
    print("Validation used                 : FALSE")
    print("Test used                       : FALSE")
    print("Future target leakage           : NONE")
    print("Dashboard package validation    : PASS")

    print()
    print("Dashboard HTML:")
    print(OUTPUT_HTML)

    print()
    print("Dashboard forecast:")
    print(OUTPUT_FORECAST)

    print()
    print("Dashboard hourly:")
    print(OUTPUT_HOURLY)

    print()
    print("Dashboard summary:")
    print(OUTPUT_SUMMARY)

    print()
    print("Dashboard package:")
    print(OUTPUT_PACKAGE)

    print()
    print("Step 19 report:")
    print(STEP19_REPORT)

    print()
    print(f"Execution time                  : {execution_seconds:.4f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Execution interrupted by user.")
    except SystemExit:
        raise
    except Exception as exc:
        fail(
            f"{type(exc).__name__}: {exc}"
        )