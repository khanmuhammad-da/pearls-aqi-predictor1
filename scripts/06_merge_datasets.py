from pathlib import Path
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports"

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

AQI_FILE = (
    RAW_DIR /
    "lahore_historical_air_quality_extended.csv"
)

WEATHER_FILE = (
    RAW_DIR /
    "lahore_historical_weather_extended.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_FILE = (
    INTERIM_DIR /
    "lahore_merged_hourly.csv"
)

REPORT_FILE = (
    REPORT_DIR /
    "merge_report.json"
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

AQI_COLUMNS = [
    "time",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "european_aqi",
    "us_aqi",
]

WEATHER_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def check_columns(
    dataframe,
    expected,
    name,
):

    missing = [
        column
        for column in expected
        if column not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} is missing columns: "
            + ", ".join(missing)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 6 — MERGE AQI + WEATHER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    print_section(
        "INPUT FILE CHECK"
    )

    check_file(AQI_FILE)
    check_file(WEATHER_FILE)

    print(
        f"AQI file     : FOUND"
    )

    print(
        f"Weather file : FOUND"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print_section(
        "LOADING DATA"
    )

    aqi = pd.read_csv(
        AQI_FILE,
        parse_dates=["time"],
    )

    weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["time"],
    )

    print(
        f"AQI rows     : {len(aqi):,}"
    )

    print(
        f"Weather rows : {len(weather):,}"
    )

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    print_section(
        "COLUMN VALIDATION"
    )

    check_columns(
        aqi,
        AQI_COLUMNS,
        "Air quality data",
    )

    check_columns(
        weather,
        WEATHER_COLUMNS,
        "Weather data",
    )

    print(
        "AQI columns     : PASS"
    )

    print(
        "Weather columns : PASS"
    )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    print_section(
        "TIMESTAMP VALIDATION"
    )

    if aqi["time"].duplicated().any():

        raise RuntimeError(
            "Duplicate timestamps found "
            "in AQI dataset."
        )

    if weather["time"].duplicated().any():

        raise RuntimeError(
            "Duplicate timestamps found "
            "in weather dataset."
        )

    print(
        "AQI timestamps     : unique"
    )

    print(
        "Weather timestamps : unique"
    )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    aqi = aqi[AQI_COLUMNS].copy()

    weather = weather[WEATHER_COLUMNS].copy()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print_section(
        "MERGING DATASETS"
    )

    merged = pd.merge(
        aqi,
        weather,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged rows    : "
        f"{len(merged):,}"
    )

    print(
        f"Merged columns : "
        f"{len(merged.columns)}"
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    merged = (
        merged
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check expected row count
    # --------------------------------------------------------

    print_section(
        "MERGE VALIDATION"
    )

    expected_rows = min(
        len(aqi),
        len(weather),
    )

    print(
        f"Expected rows : "
        f"{expected_rows:,}"
    )

    print(
        f"Actual rows   : "
        f"{len(merged):,}"
    )

    if len(merged) != expected_rows:

        raise RuntimeError(
            "Rows were lost during the merge."
        )

    print(
        "Row count: PASS"
    )

    # --------------------------------------------------------
    # Check timestamps
    # --------------------------------------------------------

    aqi_times = set(aqi["time"])
    weather_times = set(weather["time"])
    merged_times = set(merged["time"])

    only_aqi = (
        aqi_times - weather_times
    )

    only_weather = (
        weather_times - aqi_times
    )

    missing_after_merge = (
        (aqi_times & weather_times)
        - merged_times
    )

    print()

    print(
        f"Only AQI timestamps       : "
        f"{len(only_aqi)}"
    )

    print(
        f"Only weather timestamps   : "
        f"{len(only_weather)}"
    )

    print(
        f"Missing after merge       : "
        f"{len(missing_after_merge)}"
    )

    if (
        only_aqi
        or only_weather
        or missing_after_merge
    ):

        raise RuntimeError(
            "Timestamp mismatch detected."
        )

    print(
        "Timestamp alignment: PASS"
    )

    # --------------------------------------------------------
    # Missing-value check
    # --------------------------------------------------------

    print_section(
        "MISSING VALUE CHECK"
    )

    missing = merged.isna().sum()

    total_missing = (
        int(missing.sum())
    )

    if total_missing == 0:

        print(
            "Missing values: NONE"
        )

    else:

        print(
            f"Total missing values: "
            f"{total_missing}"
        )

        for column, count in missing.items():

            if count > 0:

                print(
                    f"  {column}: {count}"
                )

        raise RuntimeError(
            "Merged dataset contains "
            "missing values."
        )

    # --------------------------------------------------------
    # Time continuity
    # --------------------------------------------------------

    print_section(
        "TIME CONTINUITY CHECK"
    )

    time_diff = (
        merged["time"]
        .diff()
        .dropna()
    )

    irregular = (
        time_diff
        != pd.Timedelta(hours=1)
    )

    irregular_count = int(
        irregular.sum()
    )

    print(
        f"Irregular gaps: "
        f"{irregular_count}"
    )

    if irregular_count > 0:

        print(
            "Time continuity: FAIL"
        )

        raise RuntimeError(
            "Hourly time continuity failed."
        )

    print(
        "Time continuity: PASS"
    )

    # --------------------------------------------------------
    # Target check
    # --------------------------------------------------------

    print_section(
        "TARGET CHECK"
    )

    if "us_aqi" not in merged.columns:

        raise RuntimeError(
            "us_aqi target is missing."
        )

    target_missing = int(
        merged["us_aqi"].isna().sum()
    )

    target_min = float(
        merged["us_aqi"].min()
    )

    target_max = float(
        merged["us_aqi"].max()
    )

    print(
        f"Target column : us_aqi"
    )

    print(
        f"Missing       : {target_missing}"
    )

    print(
        f"Minimum       : {target_min}"
    )

    print(
        f"Maximum       : {target_max}"
    )

    if target_missing > 0:

        raise RuntimeError(
            "AQI target contains missing values."
        )

    print(
        "Target check: PASS"
    )

    # --------------------------------------------------------
    # Save merged dataset
    # --------------------------------------------------------

    print_section(
        "SAVING DATASET"
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {

        "step": 6,

        "description":
            "Merged historical AQI and weather data",

        "source": "Open-Meteo",

        "rows": int(
            len(merged)
        ),

        "columns": int(
            len(merged.columns)
        ),

        "start":
            str(merged["time"].min()),

        "end":
            str(merged["time"].max()),

        "missing_values":
            total_missing,

        "irregular_gaps":
            irregular_count,

        "target": {

            "column": "us_aqi",

            "missing":
                target_missing,

            "minimum":
                target_min,

            "maximum":
                target_max,
        },

        "validation": {

            "row_count": True,

            "timestamp_alignment": True,

            "missing_values": True,

            "time_continuity": True,

            "target": True,
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print_section(
        "FIRST 5 ROWS"
    )

    print(
        merged.head(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 6 COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Rows       : {len(merged):,}"
    )

    print(
        f"Columns    : {len(merged.columns)}"
    )

    print(
        f"Start      : {merged['time'].min()}"
    )

    print(
        f"End        : {merged['time'].max()}"
    )

    print(
        f"Target     : us_aqi"
    )

    print()

    print(
        "No cleaning performed."
    )

    print(
        "No missing values filled."
    )

    print(
        "No outliers removed."
    )

    print(
        "No features created."
    )


if __name__ == "__main__":
    main()