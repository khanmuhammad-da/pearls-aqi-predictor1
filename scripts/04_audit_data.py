from pathlib import Path
import json

import pandas as pd
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "config.yaml"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "open_meteo"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FILES
# ============================================================

AQI_FILE = (
    RAW_DIR
    / "lahore_historical_air_quality.csv"
)

WEATHER_FILE = (
    RAW_DIR
    / "lahore_historical_weather.csv"
)

METADATA_FILE = (
    RAW_DIR
    / "lahore_data_metadata.json"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8",
) as file:

    config = yaml.safe_load(file)


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
# EXPECTED PERIOD
# ============================================================

EXPECTED_START = pd.Timestamp(
    "2024-01-01 00:00:00"
)

EXPECTED_END = pd.Timestamp(
    "2025-12-31 23:00:00"
)


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_columns(df, expected, name):

    print()
    print(f"{name} columns:")

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    extra = [
        column
        for column in df.columns
        if column not in expected
    ]

    if missing:

        print("  MISSING:")
        for column in missing:
            print(f"    - {column}")

    else:

        print("  Missing expected columns: NONE")

    if extra:

        print("  Additional columns:")
        for column in extra:
            print(f"    - {column}")

    else:

        print("  Additional columns: NONE")

    return len(missing) == 0


def audit_missing_values(df, name):

    print()
    print(f"{name} missing values:")

    missing = df.isna().sum()

    missing_percent = (
        missing / len(df) * 100
    )

    for column in df.columns:

        print(
            f"  {column:25s} "
            f": {missing[column]:6,} "
            f"({missing_percent[column]:6.2f}%)"
        )

    return missing


def audit_duplicates(df, name):

    duplicate_rows = df.duplicated().sum()
    duplicate_times = df["time"].duplicated().sum()

    print()
    print(f"{name} duplicates:")
    print(
        f"  Duplicate rows       : "
        f"{duplicate_rows:,}"
    )
    print(
        f"  Duplicate timestamps : "
        f"{duplicate_times:,}"
    )

    return duplicate_rows, duplicate_times


def audit_time_continuity(df, name):

    times = (
        df["time"]
        .sort_values()
        .reset_index(drop=True)
    )

    differences = times.diff()

    expected_frequency = pd.Timedelta(
        hours=1
    )

    irregular = (
        differences.notna()
        & (differences != expected_frequency)
    )

    irregular_count = irregular.sum()

    print()
    print(f"{name} time continuity:")

    print(
        f"  First timestamp : "
        f"{times.iloc[0]}"
    )

    print(
        f"  Last timestamp  : "
        f"{times.iloc[-1]}"
    )

    print(
        f"  Irregular gaps  : "
        f"{irregular_count:,}"
    )

    if irregular_count > 0:

        print()
        print("  First irregular gaps:")

        indices = (
            irregular[irregular]
            .index[:10]
        )

        for index in indices:

            previous = times.iloc[index - 1]
            current = times.iloc[index]

            print(
                f"    {previous} → {current} "
                f"({current - previous})"
            )

    return irregular_count


def audit_range(
    df,
    name,
    expected_start,
    expected_end,
):

    actual_start = df["time"].min()
    actual_end = df["time"].max()

    print()
    print(f"{name} date range:")

    print(
        f"  Expected start : "
        f"{expected_start}"
    )

    print(
        f"  Actual start   : "
        f"{actual_start}"
    )

    print(
        f"  Expected end   : "
        f"{expected_end}"
    )

    print(
        f"  Actual end     : "
        f"{actual_end}"
    )

    start_ok = (
        actual_start == expected_start
    )

    end_ok = (
        actual_end == expected_end
    )

    print()
    print(
        f"  Start correct: "
        f"{'YES' if start_ok else 'NO'}"
    )

    print(
        f"  End correct  : "
        f"{'YES' if end_ok else 'NO'}"
    )

    return start_ok and end_ok


def audit_numeric_statistics(df, columns):

    print()
    print("Basic statistics:")

    for column in columns:

        if column not in df.columns:
            continue

        series = df[column]

        print()
        print(f"  {column}")

        print(
            f"    min    : "
            f"{series.min()}"
        )

        print(
            f"    max    : "
            f"{series.max()}"
        )

        print(
            f"    mean   : "
            f"{series.mean():.2f}"
        )

        print(
            f"    median : "
            f"{series.median():.2f}"
        )


def audit_aqi(df):

    print_section(
        "AQI TARGET AUDIT"
    )

    if "us_aqi" not in df.columns:

        print(
            "ERROR: us_aqi column not found."
        )

        return {}

    aqi = df["us_aqi"]

    result = {
        "missing": int(aqi.isna().sum()),
        "zero": int((aqi == 0).sum()),
        "negative": int((aqi < 0).sum()),
        "maximum": float(aqi.max()),
    }

    print(
        f"Missing AQI : "
        f"{result['missing']:,}"
    )

    print(
        f"Zero AQI    : "
        f"{result['zero']:,}"
    )

    print(
        f"Negative AQI: "
        f"{result['negative']:,}"
    )

    print(
        f"Maximum AQI : "
        f"{result['maximum']}"
    )

    print()
    print("AQI distribution:")

    bins = [
        -float("inf"),
        50,
        100,
        150,
        200,
        300,
        500,
        float("inf"),
    ]

    labels = [
        "0-50",
        "51-100",
        "101-150",
        "151-200",
        "201-300",
        "301-500",
        "500+",
    ]

    categories = pd.cut(
        aqi,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distribution = (
        categories
        .value_counts(sort=False)
    )

    for category, count in distribution.items():

        percentage = (
            count / len(aqi) * 100
        )

        print(
            f"  {str(category):8s}: "
            f"{count:6,} "
            f"({percentage:6.2f}%)"
        )

    return result


def audit_pollutants(df):

    print_section(
        "POLLUTANT AUDIT"
    )

    pollutants = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    results = {}

    for column in pollutants:

        if column not in df.columns:
            continue

        series = df[column]

        negative = (
            series < 0
        ).sum()

        zero = (
            series == 0
        ).sum()

        results[column] = {
            "missing": int(
                series.isna().sum()
            ),
            "negative": int(
                negative
            ),
            "zero": int(
                zero
            ),
            "min": float(
                series.min()
            ),
            "max": float(
                series.max()
            ),
            "mean": float(
                series.mean()
            ),
        }

        print()
        print(f"  {column}")

        print(
            f"    Missing : "
            f"{results[column]['missing']:,}"
        )

        print(
            f"    Negative: "
            f"{results[column]['negative']:,}"
        )

        print(
            f"    Zero    : "
            f"{results[column]['zero']:,}"
        )

        print(
            f"    Min     : "
            f"{results[column]['min']:.3f}"
        )

        print(
            f"    Max     : "
            f"{results[column]['max']:.3f}"
        )

        print(
            f"    Mean    : "
            f"{results[column]['mean']:.3f}"
        )

    return results


def audit_constant_columns(df, name):

    print()
    print(
        f"{name} constant-column check:"
    )

    results = []

    for column in df.columns:

        unique = df[column].nunique(
            dropna=False
        )

        if unique <= 1:

            results.append(column)

            print(
                f"  WARNING: {column} "
                f"has only {unique} unique value."
            )

    if not results:

        print(
            "  No constant columns."
        )

    return results


def audit_alignment(
    aqi_df,
    weather_df,
):

    print_section(
        "AQI / WEATHER TIMESTAMP ALIGNMENT"
    )

    aqi_times = set(
        aqi_df["time"]
    )

    weather_times = set(
        weather_df["time"]
    )

    only_aqi = (
        aqi_times - weather_times
    )

    only_weather = (
        weather_times - aqi_times
    )

    common = (
        aqi_times & weather_times
    )

    print(
        f"AQI timestamps     : "
        f"{len(aqi_times):,}"
    )

    print(
        f"Weather timestamps : "
        f"{len(weather_times):,}"
    )

    print(
        f"Common timestamps  : "
        f"{len(common):,}"
    )

    print(
        f"Only AQI           : "
        f"{len(only_aqi):,}"
    )

    print(
        f"Only weather       : "
        f"{len(only_weather):,}"
    )

    aligned = (
        len(only_aqi) == 0
        and len(only_weather) == 0
    )

    print()

    print(
        "Alignment: "
        f"{'PASS' if aligned else 'FAIL'}"
    )

    return {
        "aqi_timestamps": len(aqi_times),
        "weather_timestamps": len(weather_times),
        "common": len(common),
        "only_aqi": len(only_aqi),
        "only_weather": len(only_weather),
        "aligned": aligned,
    }


def audit_metadata():

    print_section(
        "SOURCE METADATA"
    )

    if not METADATA_FILE.exists():

        print(
            "Metadata file not found."
        )

        return None

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    location = metadata.get(
        "location",
        {}
    )

    print(
        f"Requested location : "
        f"{location.get('requested_city')}, "
        f"{location.get('requested_country')}"
    )

    print(
        f"Resolved location  : "
        f"{location.get('resolved_city')}, "
        f"{location.get('resolved_country')}"
    )

    print(
        f"Requested latitude : "
        f"{location.get('latitude')}"
    )

    print(
        f"Requested longitude: "
        f"{location.get('longitude')}"
    )

    aq = metadata.get(
        "air_quality",
        {}
    )

    print()
    print("Air-quality API grid:")

    print(
        f"  Latitude : "
        f"{aq.get('latitude_returned')}"
    )

    print(
        f"  Longitude: "
        f"{aq.get('longitude_returned')}"
    )

    weather = metadata.get(
        "weather",
        {}
    )

    print()
    print("Weather API location:")

    print(
        f"  Latitude : "
        f"{weather.get('latitude_returned')}"
    )

    print(
        f"  Longitude: "
        f"{weather.get('longitude_returned')}"
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 4 — DATA QUALITY AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    print_section(
        "RAW FILE CHECK"
    )

    files_ok = True

    for file in [
        AQI_FILE,
        WEATHER_FILE,
    ]:

        exists = file.exists()

        print(
            f"{file.name:45s}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            files_ok = False

    if not files_ok:

        raise FileNotFoundError(
            "One or more required raw files "
            "are missing."
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print_section(
        "LOADING RAW DATA"
    )

    aqi_df = pd.read_csv(
        AQI_FILE,
        parse_dates=["time"],
    )

    weather_df = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["time"],
    )

    print(
        f"AQI rows     : "
        f"{len(aqi_df):,}"
    )

    print(
        f"Weather rows : "
        f"{len(weather_df):,}"
    )

    # --------------------------------------------------------
    # Column checks
    # --------------------------------------------------------

    print_section(
        "COLUMN CHECK"
    )

    aqi_columns_ok = check_columns(
        aqi_df,
        AQI_COLUMNS,
        "Air quality",
    )

    weather_columns_ok = check_columns(
        weather_df,
        WEATHER_COLUMNS,
        "Weather",
    )

    # --------------------------------------------------------
    # Date ranges
    # --------------------------------------------------------

    print_section(
        "DATE RANGE CHECK"
    )

    aqi_range_ok = audit_range(
        aqi_df,
        "Air quality",
        EXPECTED_START,
        EXPECTED_END,
    )

    weather_range_ok = audit_range(
        weather_df,
        "Weather",
        EXPECTED_START,
        EXPECTED_END,
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print_section(
        "MISSING VALUE AUDIT"
    )

    aqi_missing = audit_missing_values(
        aqi_df,
        "Air quality",
    )

    weather_missing = audit_missing_values(
        weather_df,
        "Weather",
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    print_section(
        "DUPLICATE AUDIT"
    )

    aqi_duplicates = audit_duplicates(
        aqi_df,
        "Air quality",
    )

    weather_duplicates = audit_duplicates(
        weather_df,
        "Weather",
    )

    # --------------------------------------------------------
    # Time continuity
    # --------------------------------------------------------

    print_section(
        "TIME CONTINUITY AUDIT"
    )

    aqi_gaps = audit_time_continuity(
        aqi_df,
        "Air quality",
    )

    weather_gaps = audit_time_continuity(
        weather_df,
        "Weather",
    )

    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    aqi_results = audit_aqi(
        aqi_df
    )

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------

    pollutant_results = audit_pollutants(
        aqi_df
    )

    # --------------------------------------------------------
    # Constant columns
    # --------------------------------------------------------

    print_section(
        "CONSTANT VALUE AUDIT"
    )

    aqi_constant = audit_constant_columns(
        aqi_df,
        "Air quality",
    )

    weather_constant = audit_constant_columns(
        weather_df,
        "Weather",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_section(
        "AQI / POLLUTANT STATISTICS"
    )

    audit_numeric_statistics(
        aqi_df,
        [
            "us_aqi",
            "european_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
    )

    print_section(
        "WEATHER STATISTICS"
    )

    audit_numeric_statistics(
        weather_df,
        WEATHER_COLUMNS[1:],
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    alignment = audit_alignment(
        aqi_df,
        weather_df,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = audit_metadata()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    critical_pass = all(
        [
            aqi_columns_ok,
            weather_columns_ok,
            aqi_range_ok,
            weather_range_ok,
            alignment["aligned"],
            aqi_duplicates[1] == 0,
            weather_duplicates[1] == 0,
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "project": config["project"]["name"],

        "dataset": {
            "aqi_rows": len(aqi_df),
            "weather_rows": len(weather_df),
            "expected_start": str(
                EXPECTED_START
            ),
            "expected_end": str(
                EXPECTED_END
            ),
        },

        "columns": {
            "aqi_ok": aqi_columns_ok,
            "weather_ok": weather_columns_ok,
        },

        "date_range": {
            "aqi_ok": aqi_range_ok,
            "weather_ok": weather_range_ok,
        },

        "missing_values": {
            "aqi": {
                column: int(value)
                for column, value
                in aqi_missing.items()
            },
            "weather": {
                column: int(value)
                for column, value
                in weather_missing.items()
            },
        },

        "duplicates": {
            "aqi_rows": aqi_duplicates[0],
            "aqi_timestamps": aqi_duplicates[1],
            "weather_rows": weather_duplicates[0],
            "weather_timestamps": weather_duplicates[1],
        },

        "time_gaps": {
            "aqi": int(aqi_gaps),
            "weather": int(weather_gaps),
        },

        "aqi": aqi_results,

        "pollutants": pollutant_results,

        "constant_columns": {
            "aqi": aqi_constant,
            "weather": weather_constant,
        },

        "alignment": alignment,

        "critical_checks_pass": critical_pass,
    }

    report_file = (
        REPORT_DIR
        / "data_quality_audit.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=lambda obj: obj.item()
            if hasattr(obj, "item")
            else str(obj),
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL AUDIT RESULT")
    print("=" * 70)

    if critical_pass:

        print()
        print(
            "PASS — Critical structural "
            "data checks passed."
        )

    else:

        print()
        print(
            "WARNING — One or more critical "
            "structural checks failed."
        )

    print()
    print(
        f"Audit report saved to:\n"
        f"{report_file}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No raw files were modified."
    )

    print(
        "No missing values were filled."
    )

    print(
        "No outliers were removed."
    )

    print(
        "No feature engineering was performed."
    )


if __name__ == "__main__":
    main()