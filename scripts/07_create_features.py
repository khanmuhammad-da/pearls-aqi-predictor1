from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lahore_merged_hourly.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "lahore_features_hourly.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "feature_engineering_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "us_aqi"

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_COLUMNS = [
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


def require_columns(df, columns):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def add_lag_features(
    df,
    column,
    lags,
):
    """
    Create historical lag features.

    Example:
        aqi_lag_1 at time t = AQI at t-1

    shift() guarantees that the current observation
    is not used as its own historical feature.
    """

    for lag in lags:
        df[f"{column}_lag_{lag}"] = (
            df[column].shift(lag)
        )

    return df


def add_rolling_features(
    df,
    column,
    windows,
):
    """
    Create leakage-safe rolling statistics.

    IMPORTANT:
    We shift by one hour BEFORE rolling.

    Therefore, at time t:

        rolling_mean_24

    uses:

        t-24 ... t-1

    and never t itself.
    """

    historical = df[column].shift(1)

    for window in windows:

        df[
            f"{column}_rolling_mean_{window}"
        ] = (
            historical
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    return df


def add_rolling_std_features(
    df,
    column,
    windows,
):
    historical = df[column].shift(1)

    for window in windows:

        df[
            f"{column}_rolling_std_{window}"
        ] = (
            historical
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
        )

    return df


def add_change_features(
    df,
    column,
    periods,
):
    """
    Change at t is calculated against an older
    observation.

    Example:

        change_24h = value(t-1) - value(t-25)

    This keeps the feature strictly historical.
    """

    previous = df[column].shift(1)

    for period in periods:

        df[
            f"{column}_change_{period}h"
        ] = (
            previous
            - df[column].shift(period + 1)
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 7 — FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load merged data
    # --------------------------------------------------------

    print_section(
        "LOADING MERGED DATA"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["time"],
    )

    print(
        f"Input rows    : {len(df):,}"
    )

    print(
        f"Input columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print_section(
        "INPUT VALIDATION"
    )

    required_columns = [
        "time",
        TARGET,
        *POLLUTANTS,
        *WEATHER_COLUMNS,
    ]

    require_columns(
        df,
        required_columns,
    )

    if df["time"].duplicated().any():

        raise RuntimeError(
            "Duplicate timestamps detected."
        )

    if not df["time"].is_monotonic_increasing:

        raise RuntimeError(
            "Data is not sorted chronologically."
        )

    time_diff = (
        df["time"]
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

    if irregular_count > 0:

        raise RuntimeError(
            f"Found {irregular_count} "
            "irregular hourly gaps."
        )

    print(
        "Required columns : PASS"
    )

    print(
        "Unique timestamps: PASS"
    )

    print(
        "Chronological     : PASS"
    )

    print(
        "Hourly continuity : PASS"
    )

    # --------------------------------------------------------
    # Preserve original row count
    # --------------------------------------------------------

    original_rows = len(df)

    # ========================================================
    # 1. TIME FEATURES
    # ========================================================

    print_section(
        "1. TIME FEATURES"
    )

    df["hour"] = (
        df["time"].dt.hour
    )

    df["day_of_week"] = (
        df["time"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["time"].dt.dayofyear
    )

    df["month"] = (
        df["time"].dt.month
    )

    df["week_of_year"] = (
        df["time"].dt.isocalendar().week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical hour
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # --------------------------------------------------------
    # Cyclical day of week
    # --------------------------------------------------------

    df["day_of_week_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    df["day_of_week_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    # --------------------------------------------------------
    # Cyclical day of year
    # --------------------------------------------------------

    df["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_year"]
        / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_year"]
        / 365.25
    )

    print(
        "Time features created."
    )

    # ========================================================
    # 2. AQI LAGS
    # ========================================================

    print_section(
        "2. AQI LAG FEATURES"
    )

    aqi_lags = [
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        72,
    ]

    df = add_lag_features(
        df,
        TARGET,
        aqi_lags,
    )

    print(
        "AQI lags:"
    )

    for lag in aqi_lags:
        print(
            f"  us_aqi_lag_{lag}"
        )

    # ========================================================
    # 3. POLLUTANT LAGS
    # ========================================================

    print_section(
        "3. POLLUTANT LAG FEATURES"
    )

    pollutant_lags = [
        1,
        3,
        6,
        12,
        24,
    ]

    for pollutant in POLLUTANTS:

        df = add_lag_features(
            df,
            pollutant,
            pollutant_lags,
        )

    print(
        f"Created {len(POLLUTANTS) * len(pollutant_lags)} "
        "pollutant lag features."
    )

    # ========================================================
    # 4. AQI ROLLING MEANS
    # ========================================================

    print_section(
        "4. AQI ROLLING FEATURES"
    )

    aqi_rolling_windows = [
        3,
        6,
        12,
        24,
        48,
        72,
    ]

    df = add_rolling_features(
        df,
        TARGET,
        aqi_rolling_windows,
    )

    df = add_rolling_std_features(
        df,
        TARGET,
        [6, 24, 72],
    )

    print(
        "Leakage-safe AQI rolling means:"
    )

    for window in aqi_rolling_windows:
        print(
            f"  {window} hours"
        )

    print(
        "AQI rolling standard deviations:"
    )

    for window in [6, 24, 72]:
        print(
            f"  {window} hours"
        )

    # ========================================================
    # 5. POLLUTANT ROLLING MEANS
    # ========================================================

    print_section(
        "5. POLLUTANT ROLLING FEATURES"
    )

    pollutant_rolling_windows = [
        3,
        6,
        12,
        24,
    ]

    # Focus rolling features on the two most
    # directly relevant particulate variables.
    for pollutant in [
        "pm2_5",
        "pm10",
    ]:

        df = add_rolling_features(
            df,
            pollutant,
            pollutant_rolling_windows,
        )

    print(
        "PM2.5 rolling means: "
        "3, 6, 12, 24 hours"
    )

    print(
        "PM10 rolling means: "
        "3, 6, 12, 24 hours"
    )

    # ========================================================
    # 6. AQI CHANGE FEATURES
    # ========================================================

    print_section(
        "6. AQI CHANGE FEATURES"
    )

    aqi_change_periods = [
        1,
        3,
        6,
        12,
        24,
    ]

    df = add_change_features(
        df,
        TARGET,
        aqi_change_periods,
    )

    print(
        "Created AQI historical change features:"
    )

    for period in aqi_change_periods:
        print(
            f"  us_aqi_change_{period}h"
        )

    # ========================================================
    # 7. POLLUTANT CHANGE FEATURES
    # ========================================================

    print_section(
        "7. POLLUTANT CHANGE FEATURES"
    )

    for pollutant in [
        "pm2_5",
        "pm10",
    ]:

        df = add_change_features(
            df,
            pollutant,
            [3, 6, 24],
        )

    print(
        "Created PM2.5 and PM10 "
        "historical change features."
    )

    # ========================================================
    # 8. WEATHER DERIVED FEATURES
    # ========================================================

    print_section(
        "8. WEATHER-DERIVED FEATURES"
    )

    # --------------------------------------------------------
    # Wind vector components
    #
    # Wind direction is circular. Treating 359° and 1°
    # as numerically far apart is undesirable.
    #
    # u = east/west component
    # v = north/south component
    # --------------------------------------------------------

    radians = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["wind_u"] = (
        df["wind_speed_10m"]
        * np.sin(radians)
    )

    df["wind_v"] = (
        df["wind_speed_10m"]
        * np.cos(radians)
    )

    # --------------------------------------------------------
    # Weather changes
    # --------------------------------------------------------

    for column in [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
    ]:

        df = add_change_features(
            df,
            column,
            [3, 24],
        )

    print(
        "Created wind vector features."
    )

    print(
        "Created historical temperature, "
        "humidity and pressure changes."
    )

    # ========================================================
    # 9. FEATURE SAFETY CHECK
    # ========================================================

    print_section(
        "9. FEATURE SAFETY CHECK"
    )

    # --------------------------------------------------------
    # The target itself must remain unchanged.
    # --------------------------------------------------------

    target_missing = int(
        df[TARGET].isna().sum()
    )

    print(
        f"Current target missing values: "
        f"{target_missing}"
    )

    if target_missing != 0:

        raise RuntimeError(
            "Target contains missing values."
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "time",
            TARGET,
        ]
    ]

    print(
        f"Feature columns created: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Verify that all historical features use lagged data.
    #
    # The current raw pollutant/weather values are intentionally
    # retained because they represent information available at
    # prediction time.
    # --------------------------------------------------------

    print()
    print(
        "Current-observation variables retained:"
    )

    current_variables = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
    ]

    for column in current_variables:

        print(
            f"  {column}"
        )

    # ========================================================
    # 10. INITIAL ROW LOSS
    # ========================================================

    print_section(
        "10. INITIAL FEATURE ROW LOSS"
    )

    # The longest historical dependency is 72 hours.
    #
    # The first 72 rows cannot have complete lag/rolling
    # information. We remove those rows only after feature
    # creation.
    #
    # We do NOT fill these values with zeros.
    # ========================================================

    max_history = 72

    before_drop = len(df)

    df = df.iloc[
        max_history:
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    after_drop = len(df)

    print(
        f"Rows before : {before_drop:,}"
    )

    print(
        f"Rows after  : {after_drop:,}"
    )

    print(
        f"Rows removed: "
        f"{before_drop - after_drop:,}"
    )

    # ========================================================
    # 11. FINAL MISSING-VALUE CHECK
    # ========================================================

    print_section(
        "11. FINAL FEATURE VALIDATION"
    )

    missing_counts = (
        df.isna().sum()
    )

    total_missing = int(
        missing_counts.sum()
    )

    if total_missing > 0:

        print(
            "Missing values found:"
        )

        for column, count in (
            missing_counts.items()
        ):

            if count > 0:

                print(
                    f"  {column}: {count}"
                )

        raise RuntimeError(
            "Feature dataset contains "
            "missing values after warm-up removal."
        )

    print(
        "Missing values: NONE"
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = int(
        df["time"].duplicated().sum()
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise RuntimeError(
            "Duplicate timestamps found."
        )

    print(
        "Duplicate check: PASS"
    )

    # --------------------------------------------------------
    # Time continuity check
    # --------------------------------------------------------

    time_diff = (
        df["time"]
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
        f"Irregular time gaps: "
        f"{irregular_count}"
    )

    if irregular_count > 0:

        raise RuntimeError(
            "Feature dataset is not "
            "hourly continuous."
        )

    print(
        "Time continuity: PASS"
    )

    # ========================================================
    # 12. SAVE FEATURE DATASET
    # ========================================================

    print_section(
        "12. SAVING FEATURE DATASET"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    # ========================================================
    # 13. FEATURE LIST
    # ========================================================

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "time",
            TARGET,
        ]
    ]

    print_section(
        "FEATURE SUMMARY"
    )

    print(
        f"Final rows    : {len(df):,}"
    )

    print(
        f"Final columns : {len(df.columns)}"
    )

    print(
        f"Features      : {len(feature_columns)}"
    )

    print()
    print(
        "Feature groups:"
    )

    print(
        "  Time features"
    )

    print(
        "  AQI lag features"
    )

    print(
        "  Pollutant lag features"
    )

    print(
        "  AQI rolling features"
    )

    print(
        "  Pollutant rolling features"
    )

    print(
        "  AQI change features"
    )

    print(
        "  Pollutant change features"
    )

    print(
        "  Wind vector features"
    )

    print(
        "  Weather change features"
    )

    # ========================================================
    # 14. SAVE REPORT
    # ========================================================

    report = {

        "step": 7,

        "description":
            "Leakage-safe historical feature engineering",

        "input_file":
            str(INPUT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "target":
            TARGET,

        "input_rows":
            int(original_rows),

        "output_rows":
            int(len(df)),

        "rows_removed_for_warmup":
            int(original_rows - len(df)),

        "input_start":
            str(
                pd.to_datetime(
                    pd.read_csv(
                        INPUT_FILE,
                        usecols=["time"],
                    )["time"]
                ).min()
            ),

        "output_start":
            str(df["time"].min()),

        "output_end":
            str(df["time"].max()),

        "feature_count":
            int(len(feature_columns)),

        "feature_columns":
            feature_columns,

        "leakage_protection": {

            "lag_features_use_shift":
                True,

            "rolling_features_shifted_before_rolling":
                True,

            "change_features_use_historical_values":
                True,

            "future_target_columns_created":
                False,
        },

        "validation": {

            "missing_values":
                total_missing == 0,

            "duplicate_timestamps":
                duplicates == 0,

            "hourly_continuity":
                irregular_count == 0,
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

    # ========================================================
    # 15. SAMPLE
    # ========================================================

    print_section(
        "FIRST 3 FEATURE ROWS"
    )

    print(
        df.head(3).to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7 COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Input rows       : {original_rows:,}"
    )

    print(
        f"Output rows      : {len(df):,}"
    )

    print(
        f"Output columns   : {len(df.columns)}"
    )

    print(
        f"Feature count    : {len(feature_columns)}"
    )

    print(
        f"Target           : {TARGET}"
    )

    print(
        f"Output file      : {OUTPUT_FILE}"
    )

    print(
        f"Report           : {REPORT_FILE}"
    )

    print()

    print(
        "Future target columns were NOT created."
    )

    print(
        "Feature engineering completed "
        "without filling missing values."
    )

    print(
        "Rolling features were shifted to "
        "prevent target leakage."
    )


if __name__ == "__main__":
    main()