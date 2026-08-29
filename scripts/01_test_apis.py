from pathlib import Path
import json
import sys

import requests
import yaml


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# ============================================================
# CONFIG VALUES
# ============================================================

CITY = config["location"]["default_city"]
COUNTRY = config["location"]["default_country"]

OPEN_METEO = config["api"]["open_meteo"]

GEOCODING_URL = OPEN_METEO["geocoding_url"]
AIR_QUALITY_URL = OPEN_METEO["air_quality_url"]
WEATHER_URL = OPEN_METEO["weather_url"]
HISTORICAL_WEATHER_URL = OPEN_METEO["historical_weather_url"]

AIR_QUALITY_VARIABLES = OPEN_METEO["air_quality_variables"]
WEATHER_VARIABLES = OPEN_METEO["weather_variables"]

TIMEZONE = "Asia/Karachi"

# Small test period only.
# This is NOT our final training period.
TEST_START = "2025-01-01"
TEST_END = "2025-01-07"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "open_meteo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HTTP HELPER
# ============================================================

def get_json(url, params):
    """Send GET request and return JSON response."""

    try:
        response = requests.get(
            url,
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        print(f"REQUEST ERROR: {exc}")
        return None

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("Response:")
        print(response.text[:1000])
        return None

    try:
        return response.json()
    except ValueError:
        print("ERROR: Response was not valid JSON.")
        print(response.text[:1000])
        return None


# ============================================================
# 1. GEOCODING
# ============================================================

def test_geocoding():

    print()
    print("=" * 70)
    print("1. GEOCODING")
    print("=" * 70)

    print(f"City    : {CITY}")
    print(f"Country : {COUNTRY}")
    print()

    params = {
        "name": CITY,
        "count": 5,
        "language": "en",
        "format": "json",
    }

    data = get_json(
        GEOCODING_URL,
        params,
    )

    if not data:
        return None

    results = data.get("results", [])

    if not results:
        print("ERROR: No geocoding results found.")
        return None

    print(f"Results found: {len(results)}")
    print()

    selected = None

    for result in results:

        name = result.get("name")
        country = result.get("country")
        latitude = result.get("latitude")
        longitude = result.get("longitude")

        print(
            f"{name}, {country} "
            f"→ ({latitude}, {longitude})"
        )

        if (
            name.lower() == CITY.lower()
            and country.lower() == COUNTRY.lower()
        ):
            selected = result

    if selected is None:
        selected = results[0]

    latitude = selected["latitude"]
    longitude = selected["longitude"]

    print()
    print("SELECTED LOCATION")
    print(f"City      : {selected.get('name')}")
    print(f"Country   : {selected.get('country')}")
    print(f"Latitude  : {latitude}")
    print(f"Longitude : {longitude}")
    print(f"Timezone  : {selected.get('timezone')}")

    return {
        "city": selected.get("name"),
        "country": selected.get("country"),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": selected.get("timezone"),
    }


# ============================================================
# 2. HISTORICAL AIR QUALITY
# ============================================================

def test_historical_air_quality(latitude, longitude):

    print()
    print("=" * 70)
    print("2. HISTORICAL AIR QUALITY")
    print("=" * 70)

    print(f"Start : {TEST_START}")
    print(f"End   : {TEST_END}")
    print()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(AIR_QUALITY_VARIABLES),
        "start_date": TEST_START,
        "end_date": TEST_END,
        "timezone": TIMEZONE,
    }

    data = get_json(
        AIR_QUALITY_URL,
        params,
    )

    if not data:
        return False

    hourly = data.get("hourly")

    if not hourly:
        print("ERROR: No hourly air-quality data.")
        return False

    print()
    print("Hourly variables:")

    for variable, values in hourly.items():

        if isinstance(values, list):
            print(
                f"  {variable:25s}: "
                f"{len(values)} observations"
            )

    times = hourly.get("time", [])
    us_aqi = hourly.get("us_aqi", [])
    pm25 = hourly.get("pm2_5", [])

    print()
    print("First 5 observations:")

    for i in range(min(5, len(times))):

        aqi = us_aqi[i] if i < len(us_aqi) else None
        pm = pm25[i] if i < len(pm25) else None

        print(
            f"  {times[i]} | "
            f"US AQI={aqi} | "
            f"PM2.5={pm}"
        )

    return True


# ============================================================
# 3. HISTORICAL WEATHER
# ============================================================

def test_historical_weather(latitude, longitude):

    print()
    print("=" * 70)
    print("3. HISTORICAL WEATHER")
    print("=" * 70)

    print(f"Start : {TEST_START}")
    print(f"End   : {TEST_END}")
    print()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": TEST_START,
        "end_date": TEST_END,
        "timezone": TIMEZONE,
    }

    data = get_json(
        HISTORICAL_WEATHER_URL,
        params,
    )

    if not data:
        return False

    hourly = data.get("hourly")

    if not hourly:
        print("ERROR: No hourly weather data.")
        return False

    print()
    print("Hourly variables:")

    for variable, values in hourly.items():

        if isinstance(values, list):
            print(
                f"  {variable:25s}: "
                f"{len(values)} observations"
            )

    return True


# ============================================================
# 4. CURRENT WEATHER FORECAST
# ============================================================

def test_current_weather(latitude, longitude):

    print()
    print("=" * 70)
    print("4. CURRENT WEATHER FORECAST")
    print("=" * 70)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_days": 3,
        "timezone": "auto",
    }

    data = get_json(
        WEATHER_URL,
        params,
    )

    if not data:
        return False

    hourly = data.get("hourly")

    if not hourly:
        print("ERROR: No forecast weather data.")
        return False

    times = hourly.get("time", [])

    print()
    print(
        f"Forecast hourly observations: "
        f"{len(times)}"
    )

    if times:
        print(f"Forecast begins: {times[0]}")
        print(f"Forecast ends  : {times[-1]}")

    return True


# ============================================================
# 5. CURRENT AIR-QUALITY FORECAST
# ============================================================

def test_current_air_quality(latitude, longitude):

    print()
    print("=" * 70)
    print("5. CURRENT AIR-QUALITY FORECAST")
    print("=" * 70)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(AIR_QUALITY_VARIABLES),
        "forecast_days": 3,
        "timezone": "auto",
    }

    data = get_json(
        AIR_QUALITY_URL,
        params,
    )

    if not data:
        return False

    hourly = data.get("hourly")

    if not hourly:
        print("ERROR: No forecast air-quality data.")
        return False

    times = hourly.get("time", [])

    print()
    print(
        f"Forecast hourly observations: "
        f"{len(times)}"
    )

    if times:
        print(f"Forecast begins: {times[0]}")
        print(f"Forecast ends  : {times[-1]}")

    return True


# ============================================================
# SAVE TEST SUMMARY
# ============================================================

def save_summary(location, results):

    output_file = (
        OUTPUT_DIR
        / "api_test_summary.json"
    )

    summary = {
        "location": location,
        "test_period": {
            "start": TEST_START,
            "end": TEST_END,
        },
        "tests": results,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Test summary saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 1 — API CONNECTIVITY TEST")
    print("=" * 70)

    print()
    print(f"Default location: {CITY}, {COUNTRY}")

    # --------------------------------------------------------
    # Geocoding
    # --------------------------------------------------------

    location = test_geocoding()

    if location is None:

        print()
        print("STOP: Geocoding failed.")
        sys.exit(1)

    latitude = location["latitude"]
    longitude = location["longitude"]

    # --------------------------------------------------------
    # API tests
    # --------------------------------------------------------

    results = {}

    results["historical_air_quality"] = (
        test_historical_air_quality(
            latitude,
            longitude,
        )
    )

    results["historical_weather"] = (
        test_historical_weather(
            latitude,
            longitude,
        )
    )

    results["current_weather_forecast"] = (
        test_current_weather(
            latitude,
            longitude,
        )
    )

    results["current_air_quality_forecast"] = (
        test_current_air_quality(
            latitude,
            longitude,
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    save_summary(
        location,
        results,
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    for name, success in results.items():

        status = "PASS" if success else "FAIL"

        print(
            f"{name:35s}: {status}"
        )

    print()

    if all(results.values()):

        print(
            "ALL API TESTS PASSED."
        )

        print()
        print(
            "Step 1 is complete."
        )

    else:

        print(
            "ONE OR MORE API TESTS FAILED."
        )

        print(
            "Do not continue to the downloader yet."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()