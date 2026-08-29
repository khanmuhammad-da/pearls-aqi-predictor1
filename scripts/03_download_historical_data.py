from pathlib import Path
from datetime import date, timedelta
import json

import pandas as pd
import requests
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

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


OPEN_METEO = config["api"]["open_meteo"]

GEOCODING_URL = OPEN_METEO["geocoding_url"]
AIR_QUALITY_URL = OPEN_METEO["air_quality_url"]
HISTORICAL_WEATHER_URL = (
    OPEN_METEO["historical_weather_url"]
)

CITY = config["location"]["default_city"]
COUNTRY = config["location"]["default_country"]

AIR_QUALITY_VARIABLES = (
    OPEN_METEO["air_quality_variables"]
)

WEATHER_VARIABLES = (
    OPEN_METEO["weather_variables"]
)


# ============================================================
# DATA PERIOD
# ============================================================

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)

TIMEZONE = "Asia/Karachi"


# ============================================================
# HTTP
# ============================================================

def get_json(url, params, label):

    print()
    print(f"Downloading {label}...")
    print(f"URL: {url}")

    try:

        response = requests.get(
            url,
            params=params,
            timeout=180,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"{label} request failed: {exc}"
        ) from exc

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"{label} failed.\n"
            f"Response:\n"
            f"{response.text[:3000]}"
        )

    try:

        return response.json()

    except ValueError as exc:

        raise RuntimeError(
            f"{label} returned invalid JSON."
        ) from exc


# ============================================================
# GEOCODING
# ============================================================

def geocode_location():

    print()
    print("=" * 70)
    print("LOCATION")
    print("=" * 70)

    print(
        f"Searching for: "
        f"{CITY}, {COUNTRY}"
    )

    params = {
        "name": CITY,
        "count": 5,
        "language": "en",
        "format": "json",
    }

    data = get_json(
        GEOCODING_URL,
        params,
        "geocoding",
    )

    results = data.get(
        "results",
        [],
    )

    if not results:

        raise RuntimeError(
            "No geocoding results found."
        )

    selected = None

    for result in results:

        if (
            result.get("name", "").lower()
            == CITY.lower()
            and
            result.get("country", "").lower()
            == COUNTRY.lower()
        ):

            selected = result
            break

    if selected is None:
        selected = results[0]

    location = {
        "requested_city": CITY,
        "requested_country": COUNTRY,
        "resolved_city": selected.get("name"),
        "resolved_country": selected.get("country"),
        "latitude": selected["latitude"],
        "longitude": selected["longitude"],
        "timezone": selected.get("timezone"),
        "elevation": selected.get("elevation"),
    }

    print()
    print("Resolved location:")
    print(
        f"City      : "
        f"{location['resolved_city']}"
    )
    print(
        f"Country   : "
        f"{location['resolved_country']}"
    )
    print(
        f"Latitude  : "
        f"{location['latitude']}"
    )
    print(
        f"Longitude : "
        f"{location['longitude']}"
    )
    print(
        f"Timezone  : "
        f"{location['timezone']}"
    )

    return location


# ============================================================
# AIR QUALITY
# ============================================================

def download_air_quality(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(
            AIR_QUALITY_VARIABLES
        ),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "timezone": TIMEZONE,
    }

    data = get_json(
        AIR_QUALITY_URL,
        params,
        "historical air quality",
    )

    hourly = data.get(
        "hourly",
        {},
    )

    if not hourly:

        raise RuntimeError(
            "Historical air-quality response "
            "contains no hourly data."
        )

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:

        raise RuntimeError(
            "Air-quality response contains "
            "no time column."
        )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    output_csv = (
        RAW_DIR
        / "lahore_historical_air_quality.csv"
    )

    output_json = (
        RAW_DIR
        / "lahore_historical_air_quality.json"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    print()
    print("Historical air quality:")
    print(
        f"Rows    : {len(df):,}"
    )
    print(
        f"Columns : {len(df.columns)}"
    )
    print(
        f"First   : {df['time'].min()}"
    )
    print(
        f"Last    : {df['time'].max()}"
    )

    print()
    print(
        f"CSV saved:\n"
        f"{output_csv}"
    )

    print(
        f"Raw JSON saved:\n"
        f"{output_json}"
    )

    return df, data


# ============================================================
# HISTORICAL WEATHER
# ============================================================

def download_historical_weather(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(
            WEATHER_VARIABLES
        ),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "timezone": TIMEZONE,
    }

    data = get_json(
        HISTORICAL_WEATHER_URL,
        params,
        "historical weather",
    )

    hourly = data.get(
        "hourly",
        {},
    )

    if not hourly:

        raise RuntimeError(
            "Historical weather response "
            "contains no hourly data."
        )

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:

        raise RuntimeError(
            "Weather response contains "
            "no time column."
        )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    output_csv = (
        RAW_DIR
        / "lahore_historical_weather.csv"
    )

    output_json = (
        RAW_DIR
        / "lahore_historical_weather.json"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    print()
    print("Historical weather:")
    print(
        f"Rows    : {len(df):,}"
    )
    print(
        f"Columns : {len(df.columns)}"
    )
    print(
        f"First   : {df['time'].min()}"
    )
    print(
        f"Last    : {df['time'].max()}"
    )

    print()
    print(
        f"CSV saved:\n"
        f"{output_csv}"
    )

    print(
        f"Raw JSON saved:\n"
        f"{output_json}"
    )

    return df, data


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    location,
    air_quality_data,
    weather_data,
):

    metadata = {

        "project": config["project"]["name"],

        "location": location,

        "requested_period": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
        },

        "timezone": TIMEZONE,

        "air_quality": {
            "source": "Open-Meteo",
            "url": AIR_QUALITY_URL,
            "variables": AIR_QUALITY_VARIABLES,
            "latitude_returned": air_quality_data.get(
                "latitude"
            ),
            "longitude_returned": air_quality_data.get(
                "longitude"
            ),
            "timezone_returned": air_quality_data.get(
                "timezone"
            ),
        },

        "weather": {
            "source": "Open-Meteo",
            "url": HISTORICAL_WEATHER_URL,
            "variables": WEATHER_VARIABLES,
            "latitude_returned": weather_data.get(
                "latitude"
            ),
            "longitude_returned": weather_data.get(
                "longitude"
            ),
            "timezone_returned": weather_data.get(
                "timezone"
            ),
        },
    }

    output_file = (
        RAW_DIR
        / "lahore_data_metadata.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        f"Metadata saved:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 3 — HISTORICAL DATA DOWNLOAD")
    print("=" * 70)

    print()
    print(
        f"Location : "
        f"{CITY}, {COUNTRY}"
    )

    print(
        f"Start    : "
        f"{START_DATE}"
    )

    print(
        f"End      : "
        f"{END_DATE}"
    )

    print(
        f"Timezone : "
        f"{TIMEZONE}"
    )

    # --------------------------------------------------------
    # Resolve location
    # --------------------------------------------------------

    location = geocode_location()

    latitude = location["latitude"]
    longitude = location["longitude"]

    # --------------------------------------------------------
    # Download AQ
    # --------------------------------------------------------

    air_quality_df, air_quality_raw = (
        download_air_quality(
            latitude,
            longitude,
        )
    )

    # --------------------------------------------------------
    # Download weather
    # --------------------------------------------------------

    weather_df, weather_raw = (
        download_historical_weather(
            latitude,
            longitude,
        )
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        location,
        air_quality_raw,
        weather_raw,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Air-quality rows : "
        f"{len(air_quality_df):,}"
    )

    print(
        f"Weather rows     : "
        f"{len(weather_df):,}"
    )

    print()
    print("Raw data directory:")
    print(RAW_DIR)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The data has NOT been cleaned, "
        "merged, or feature-engineered."
    )

    print(
        "That will happen in later steps."
    )


if __name__ == "__main__":
    main()