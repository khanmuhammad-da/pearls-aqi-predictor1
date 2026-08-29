from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import requests
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXISTING FILES
# ============================================================

AQI_FILE = (
    RAW_DIR /
    "lahore_historical_air_quality.csv"
)

WEATHER_FILE = (
    RAW_DIR /
    "lahore_historical_weather.csv"
)

METADATA_FILE = (
    RAW_DIR /
    "lahore_data_metadata.json"
)


# ============================================================
# EXTENDED FILES
# ============================================================

EXTENDED_AQI_FILE = (
    RAW_DIR /
    "lahore_historical_air_quality_extended.csv"
)

EXTENDED_AQI_JSON = (
    RAW_DIR /
    "lahore_historical_air_quality_extended.json"
)

EXTENDED_WEATHER_FILE = (
    RAW_DIR /
    "lahore_historical_weather_extended.csv"
)

EXTENDED_WEATHER_JSON = (
    RAW_DIR /
    "lahore_historical_weather_extended.json"
)

EXTENDED_METADATA_FILE = (
    RAW_DIR /
    "lahore_extended_data_metadata.json"
)


# ============================================================
# API ENDPOINTS
# ============================================================

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ============================================================
# VARIABLES
# ============================================================

AIR_QUALITY_VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "european_aqi",
    "us_aqi",
]

WEATHER_VARIABLES = [
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


def download_json(url, params):

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    print()
    print("URL:")
    print(response.url)

    print()
    print(
        f"HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data and data["error"]:

        reason = data.get(
            "reason",
            "Unknown API error",
        )

        raise RuntimeError(
            f"Open-Meteo API error: {reason}"
        )

    return data


def json_to_dataframe(
    data,
    variables,
):

    hourly = data.get("hourly")

    if hourly is None:

        raise RuntimeError(
            "API response does not contain "
            "'hourly' data."
        )

    if "time" not in hourly:

        raise RuntimeError(
            "API response does not contain "
            "hourly timestamps."
        )

    dataframe = pd.DataFrame(
        hourly
    )

    expected = [
        "time",
        *variables,
    ]

    missing = [
        column
        for column in expected
        if column not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing expected columns: "
            + ", ".join(missing)
        )

    dataframe = dataframe[
        expected
    ].copy()

    dataframe["time"] = pd.to_datetime(
        dataframe["time"]
    )

    return dataframe


def save_dataset(
    dataframe,
    csv_path,
    json_path,
    raw_response,
):

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            raw_response,
            file,
            indent=2,
        )


def verify_dataset(
    dataframe,
    name,
):

    print()
    print(
        f"{name}:"
    )

    print(
        f"  Rows    : "
        f"{len(dataframe):,}"
    )

    print(
        f"  Columns : "
        f"{len(dataframe.columns)}"
    )

    print(
        f"  First   : "
        f"{dataframe['time'].min()}"
    )

    print(
        f"  Last    : "
        f"{dataframe['time'].max()}"
    )

    missing = dataframe.isna().sum().sum()

    duplicate_times = (
        dataframe["time"]
        .duplicated()
        .sum()
    )

    print(
        f"  Missing values      : "
        f"{missing:,}"
    )

    print(
        f"  Duplicate timestamps: "
        f"{duplicate_times:,}"
    )

    if missing > 0:

        print(
            "  WARNING: Missing values "
            "were found."
        )

    if duplicate_times > 0:

        print(
            "  WARNING: Duplicate timestamps "
            "were found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 5 — EXTEND HISTORICAL DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Check existing files
    # --------------------------------------------------------

    print_section(
        "CHECKING EXISTING DATA"
    )

    for file in [
        AQI_FILE,
        WEATHER_FILE,
        METADATA_FILE,
    ]:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

        print(
            f"FOUND: {file.name}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    location = metadata.get(
        "location",
        {},
    )

    latitude = location.get(
        "latitude"
    )

    longitude = location.get(
        "longitude"
    )

    city = location.get(
        "resolved_city",
        "Lahore",
    )

    country = location.get(
        "resolved_country",
        "Pakistan",
    )

    timezone = location.get(
        "timezone",
        "Asia/Karachi",
    )

    if latitude is None or longitude is None:

        raise RuntimeError(
            "Latitude/longitude not found "
            "in metadata."
        )

    # --------------------------------------------------------
    # Read current raw data
    # --------------------------------------------------------

    print_section(
        "CURRENT DATASET"
    )

    existing_aqi = pd.read_csv(
        AQI_FILE,
        parse_dates=["time"],
    )

    existing_weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["time"],
    )

    existing_start = min(
        existing_aqi["time"].min(),
        existing_weather["time"].min(),
    )

    existing_end = max(
        existing_aqi["time"].max(),
        existing_weather["time"].max(),
    )

    print(
        f"City      : {city}, {country}"
    )

    print(
        f"Latitude  : {latitude}"
    )

    print(
        f"Longitude : {longitude}"
    )

    print(
        f"Timezone  : {timezone}"
    )

    print()
    print(
        f"Existing start: {existing_start}"
    )

    print(
        f"Existing end  : {existing_end}"
    )

    # --------------------------------------------------------
    # Determine latest completed historical day
    # --------------------------------------------------------

    today = datetime.now().date()

    latest_completed_day = (
        today - timedelta(days=1)
    )

    target_end = pd.Timestamp(
        latest_completed_day
    )

    print()
    print(
        f"Latest completed day: "
        f"{target_end.date()}"
    )

    if target_end <= existing_end:

        print()
        print(
            "Existing dataset is already "
            "up to date."
        )

        print()
        print(
            "No download is necessary."
        )

        return

    # --------------------------------------------------------
    # API date range
    # --------------------------------------------------------

    start_date = (
        existing_start
        .strftime("%Y-%m-%d")
    )

    end_date = (
        target_end
        .strftime("%Y-%m-%d")
    )

    print_section(
        "EXTENSION RANGE"
    )

    print(
        f"Start: {start_date}"
    )

    print(
        f"End  : {end_date}"
    )

    print()
    print(
        "The original raw files will NOT "
        "be overwritten."
    )

    # --------------------------------------------------------
    # Historical AQI
    # --------------------------------------------------------

    print_section(
        "DOWNLOADING EXTENDED AIR QUALITY"
    )

    aqi_params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": ",".join(
            AIR_QUALITY_VARIABLES
        ),

        "start_date": start_date,

        "end_date": end_date,

        "timezone": timezone,
    }

    aqi_response = download_json(
        AIR_QUALITY_URL,
        aqi_params,
    )

    extended_aqi = json_to_dataframe(
        aqi_response,
        AIR_QUALITY_VARIABLES,
    )

    verify_dataset(
        extended_aqi,
        "Extended air quality",
    )

    # --------------------------------------------------------
    # Historical weather
    # --------------------------------------------------------

    print_section(
        "DOWNLOADING EXTENDED WEATHER"
    )

    weather_params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": ",".join(
            WEATHER_VARIABLES
        ),

        "start_date": start_date,

        "end_date": end_date,

        "timezone": timezone,
    }

    weather_response = download_json(
        WEATHER_URL,
        weather_params,
    )

    extended_weather = json_to_dataframe(
        weather_response,
        WEATHER_VARIABLES,
    )

    verify_dataset(
        extended_weather,
        "Extended weather",
    )

    # --------------------------------------------------------
    # Check alignment
    # --------------------------------------------------------

    print_section(
        "TIMESTAMP ALIGNMENT"
    )

    aqi_times = set(
        extended_aqi["time"]
    )

    weather_times = set(
        extended_weather["time"]
    )

    common = (
        aqi_times & weather_times
    )

    only_aqi = (
        aqi_times - weather_times
    )

    only_weather = (
        weather_times - aqi_times
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

    if only_aqi or only_weather:

        raise RuntimeError(
            "AQI and weather timestamps "
            "are not aligned."
        )

    print()
    print(
        "Alignment: PASS"
    )

    # --------------------------------------------------------
    # Save extended data
    # --------------------------------------------------------

    print_section(
        "SAVING EXTENDED DATA"
    )

    save_dataset(
        extended_aqi,
        EXTENDED_AQI_FILE,
        EXTENDED_AQI_JSON,
        aqi_response,
    )

    save_dataset(
        extended_weather,
        EXTENDED_WEATHER_FILE,
        EXTENDED_WEATHER_JSON,
        weather_response,
    )

    extended_metadata = {

        "project": "Pearls AQI Predictor",

        "source": "Open-Meteo",

        "location": {
            "city": city,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
        },

        "period": {
            "start": str(
                extended_aqi["time"].min()
            ),
            "end": str(
                extended_aqi["time"].max()
            ),
        },

        "rows": {
            "air_quality": len(
                extended_aqi
            ),
            "weather": len(
                extended_weather
            ),
        },

        "variables": {
            "air_quality":
                AIR_QUALITY_VARIABLES,
            "weather":
                WEATHER_VARIABLES,
        },

        "created_at": datetime.now().isoformat(),

        "note": (
            "Extended historical dataset. "
            "Original Step 3 raw files "
            "were not modified."
        ),
    }

    with open(
        EXTENDED_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            extended_metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Extended AQI rows     : "
        f"{len(extended_aqi):,}"
    )

    print(
        f"Extended weather rows : "
        f"{len(extended_weather):,}"
    )

    print(
        f"Dataset start         : "
        f"{extended_aqi['time'].min()}"
    )

    print(
        f"Dataset end           : "
        f"{extended_aqi['time'].max()}"
    )

    print()

    print(
        "Files created:"
    )

    print(
        f"  {EXTENDED_AQI_FILE}"
    )

    print(
        f"  {EXTENDED_AQI_JSON}"
    )

    print(
        f"  {EXTENDED_WEATHER_FILE}"
    )

    print(
        f"  {EXTENDED_WEATHER_JSON}"
    )

    print(
        f"  {EXTENDED_METADATA_FILE}"
    )

    print()

    print(
        "Original raw files were preserved."
    )

    print(
        "No cleaning or feature engineering "
        "was performed."
    )


if __name__ == "__main__":
    main()