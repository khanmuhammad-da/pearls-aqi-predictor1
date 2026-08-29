from pathlib import Path
import requests
import yaml


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "config.yaml"
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

AIR_QUALITY_URL = (
    OPEN_METEO["air_quality_url"]
)


# ============================================================
# TEST LOCATION
# ============================================================

LATITUDE = 31.558
LONGITUDE = 74.35071

TIMEZONE = "Asia/Karachi"


# ============================================================
# TEST PERIOD
# ============================================================

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "european_aqi",
    "us_aqi",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 2 — HISTORICAL RANGE TEST")
    print("=" * 70)

    print()
    print(f"Location : Lahore")
    print(f"Latitude : {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")

    print()
    print(f"Start date: {START_DATE}")
    print(f"End date  : {END_DATE}")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(VARIABLES),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }

    print()
    print("Requesting historical air quality...")
    print(AIR_QUALITY_URL)

    try:

        response = requests.get(
            AIR_QUALITY_URL,
            params=params,
            timeout=120,
        )

    except requests.RequestException as exc:

        print()
        print("REQUEST FAILED")
        print(exc)

        return

    print()
    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:

        print()
        print("Response:")
        print(response.text[:3000])

        return

    try:

        data = response.json()

    except ValueError:

        print()
        print("ERROR: Response is not JSON.")

        return

    print()
    print("Response received successfully.")

    print()
    print("Metadata:")

    print(
        f"Latitude : "
        f"{data.get('latitude')}"
    )

    print(
        f"Longitude: "
        f"{data.get('longitude')}"
    )

    print(
        f"Timezone : "
        f"{data.get('timezone')}"
    )

    hourly = data.get(
        "hourly",
        {}
    )

    times = hourly.get(
        "time",
        []
    )

    print()
    print("=" * 70)
    print("DATA SIZE")
    print("=" * 70)

    print(
        f"Hourly observations: "
        f"{len(times):,}"
    )

    expected = 24 * 366 * 2

    print(
        f"Approximate expected: "
        f"{expected:,}"
    )

    print()
    print("Variables:")

    for variable in VARIABLES:

        values = hourly.get(
            variable,
            []
        )

        print(
            f"  {variable:25s}"
            f": {len(values):,}"
        )

    print()
    print("=" * 70)
    print("TIME RANGE RETURNED")
    print("=" * 70)

    if times:

        print(
            f"First timestamp: "
            f"{times[0]}"
        )

        print(
            f"Last timestamp : "
            f"{times[-1]}"
        )

    print()
    print("=" * 70)
    print("SAMPLE")
    print("=" * 70)

    us_aqi = hourly.get(
        "us_aqi",
        []
    )

    pm25 = hourly.get(
        "pm2_5",
        []
    )

    for i in range(
        min(5, len(times))
    ):

        print(
            f"{times[i]} | "
            f"US AQI={us_aqi[i]} | "
            f"PM2.5={pm25[i]}"
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    if len(times) > 15_000:

        print(
            "PASS — Two years of hourly "
            "historical data are available "
            "for this request."
        )

        print()
        print(
            "We can proceed with a "
            "two-year dataset."
        )

    elif len(times) > 7_000:

        print(
            "PASS — At least one year of "
            "hourly historical data is available."
        )

        print()
        print(
            "Two years were not fully returned."
        )

    else:

        print(
            "WARNING — Less than one year "
            "of data was returned."
        )

        print()
        print(
            "We need to investigate the "
            "historical data availability."
        )


if __name__ == "__main__":
    main()