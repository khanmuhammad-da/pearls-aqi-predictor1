from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# ============================================================
# PEARLS AQI PREDICTOR
# CLOUD LIVE AQI REFRESH
# ============================================================

LATITUDE = 31.5204
LONGITUDE = 74.4036

CITY = "Lahore"
AREA = "Lahore Cantonment"
COUNTRY = "Pakistan"
STATION = "Lahore Cantonment"
TIMEZONE = "Asia/Karachi"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "data"

LIVE_FILE = DATA_DIR / "pearl_live_aqi_intelligence.json"
DASHBOARD_DATA_FILE = DATA_DIR / "pearl_intelligence_dashboard_data.json"


OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


HOURLY_VARIABLES = ",".join(
    [
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]
)


# ============================================================
# AQI CLASSIFICATION
# ============================================================

def aqi_category(aqi: float) -> str:

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


def aqi_risk(aqi: float) -> str:

    if aqi <= 50:
        return "LOW"

    if aqi <= 100:
        return "MODERATE"

    if aqi <= 150:
        return "ELEVATED"

    if aqi <= 200:
        return "HIGH"

    return "SEVERE"


# ============================================================
# FETCH
# ============================================================

def fetch_open_meteo() -> dict:

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": HOURLY_VARIABLES,
        "timezone": TIMEZONE,
        "forecast_days": 5,
    }

    url = OPEN_METEO_URL + "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "PEARLS-AQI-Predictor/1.0",
        },
    )

    with urlopen(request, timeout=30) as response:

        if response.status != 200:
            raise RuntimeError(
                f"Open-Meteo HTTP {response.status}"
            )

        return json.loads(
            response.read().decode("utf-8")
        )


# ============================================================
# HELPERS
# ============================================================

def value_at(hourly: dict, name: str, index: int):

    values = hourly.get(name, [])

    if index >= len(values):
        return None

    return values[index]


def finite_values(values):

    return [
        float(v)
        for v in values
        if v is not None
    ]


# ============================================================
# BUILD LIVE INTELLIGENCE
# ============================================================

def build_live_intelligence(payload: dict) -> dict:

    hourly = payload["hourly"]

    times = hourly["time"]

    if not times:
        raise RuntimeError("Open-Meteo returned no hourly data.")

    aqi_values = hourly.get("us_aqi", [])

    valid_aqi = finite_values(aqi_values)

    if not valid_aqi:
        raise RuntimeError(
            "Open-Meteo returned no valid US AQI values."
        )

    # Current/latest available observation.
    current_index = 0

    current_aqi = float(
        aqi_values[current_index]
    )

    current_time = times[current_index]

    category = aqi_category(current_aqi)

    risk = aqi_risk(current_aqi)

    # --------------------------------------------------------
    # 72-hour forecast
    # --------------------------------------------------------

    forecast_rows = []

    for i, timestamp in enumerate(times[:72]):

        aqi = value_at(
            hourly,
            "us_aqi",
            i
        )

        if aqi is None:
            continue

        forecast_rows.append(
            {
                "timestamp": timestamp,
                "aqi": round(float(aqi), 3),
                "category": aqi_category(float(aqi)),
            }
        )

    forecast_aqi = [
        row["aqi"]
        for row in forecast_rows
    ]

    if not forecast_aqi:
        raise RuntimeError(
            "No valid forecast AQI rows were produced."
        )

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------

    pollutant_map = {
        "PM2.5": "pm2_5",
        "PM10": "pm10",
        "O3": "ozone",
        "NO2": "nitrogen_dioxide",
        "SO2": "sulphur_dioxide",
        "CO": "carbon_monoxide",
    }

    pollutants = []

    for label, source in pollutant_map.items():

        value = value_at(
            hourly,
            source,
            current_index
        )

        pollutants.append(
            {
                "pollutant": label,
                "source_column": source,
                "value": (
                    None
                    if value is None
                    else round(float(value), 3)
                ),
                "available": value is not None,
                "unit": "dataset units",
            }
        )

    available_pollutants = [
        p for p in pollutants
        if p["available"]
    ]

    dominant_pollutant = None

    if available_pollutants:

        dominant_pollutant = max(
            available_pollutants,
            key=lambda p: p["value"]
        )["pollutant"]

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if len(forecast_aqi) >= 12:

        first = statistics.mean(
            forecast_aqi[:6]
        )

        last = statistics.mean(
            forecast_aqi[-6:]
        )

        if last > first + 3:
            trend = "rising"

        elif last < first - 3:
            trend = "falling"

        else:
            trend = "stable"

    else:

        trend = "stable"

    generated_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    # --------------------------------------------------------
    # Live intelligence object
    # --------------------------------------------------------

    intelligence = {

        "project": "PEARLS AQI PREDICTOR",

        "step": 25,

        "dashboard": {
            "name": "Pearl Intelligence Dashboard",
            "type": "Live Production Intelligence",
            "generated_at": generated_at,
        },

        "location": {
            "city": CITY,
            "country": COUNTRY,
            "area": AREA,
            "station": STATION,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
            "timezone_label": "Pakistan Standard Time (PKT)",
            "location_source": "configured_location_context",
        },

        "data_freshness": {
            "status": "FRESH",
            "checked_at": generated_at,
            "observed_at": current_time,
            "source": "Open-Meteo Air Quality API",
        },

        "current": {

            "timestamp": current_time,

            "aqi": round(
                current_aqi,
                3
            ),

            "category": category,

            "category_short": category,

            "risk": risk,

            "message": (
                "Air quality is currently "
                f"{category.lower()}."
            ),

            "dominant_pollutant":
                dominant_pollutant,

            "meter": {

                "value": round(
                    current_aqi,
                    3
                ),

                "position_percent": round(
                    min(
                        max(
                            current_aqi / 300 * 100,
                            0
                        ),
                        100
                    ),
                    3,
                ),

                "scale_min": 0,

                "scale_max": 300,

                "markers": [
                    {
                        "label": "Good",
                        "value": 50,
                    },
                    {
                        "label": "Moderate",
                        "value": 100,
                    },
                    {
                        "label": "USG",
                        "value": 150,
                    },
                    {
                        "label": "Unhealthy",
                        "value": 200,
                    },
                    {
                        "label": "Very Unhealthy",
                        "value": 300,
                    },
                ],
            },

            "pollutants": pollutants,

            "aqi_source":
                "Open-Meteo Air Quality API",
        },

        "forecast": {

            "available": True,

            "rows": len(
                forecast_rows
            ),

            "minimum": min(
                forecast_aqi
            ),

            "maximum": max(
                forecast_aqi
            ),

            "mean": statistics.mean(
                forecast_aqi
            ),

            "median": statistics.median(
                forecast_aqi
            ),

            "category":
                aqi_category(
                    statistics.mean(
                        forecast_aqi
                    )
                ),

            "trend": trend,

            "start":
                forecast_rows[0]["timestamp"],

            "end":
                forecast_rows[-1]["timestamp"],

            "series":
                forecast_rows,
        },

        "source": {
            "provider":
                "Open-Meteo",

            "endpoint":
                OPEN_METEO_URL,

            "latitude":
                LATITUDE,

            "longitude":
                LONGITUDE,
        },
    }

    return intelligence


# ============================================================
# WRITE LIVE FILE
# ============================================================

def write_live_file(data: dict):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LIVE_FILE.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# UPDATE STEP 23 DATA
# ============================================================

def update_dashboard_data(live: dict):

    if DASHBOARD_DATA_FILE.exists():

        existing = json.loads(
            DASHBOARD_DATA_FILE.read_text(
                encoding="utf-8"
            )
        )

    else:

        existing = {}

    current = live["current"]

    forecast = live["forecast"]

    existing.setdefault(
        "meta",
        {}
    )

    existing["meta"].update(
        {
            "product": "PEARLS AQI Predictor",
            "dashboard": "Pearl Intelligence Dashboard",
            "step": 23,
            "target": "us_aqi",
            "forecast_horizon": 72,
            "generated_at":
                live["dashboard"]["generated_at"],
            "data_source":
                "Open-Meteo Air Quality API",
        }
    )

    existing["location"] = live["location"]

    existing["current"] = {

        "aqi": current["aqi"],

        "category":
            current["category"],

        "short_category":
            current["category_short"],

        "gauge_position":
            current["meter"]["position_percent"] / 100,

        "source":
            "live production observation",

        "timestamp":
            current["timestamp"],
    }

    existing["forecast"] = {

        "min":
            forecast["minimum"],

        "max":
            forecast["maximum"],

        "mean":
            forecast["mean"],

        "median":
            forecast["median"],

        "trend":
            forecast["trend"],

        "start":
            forecast["start"],

        "end":
            forecast["end"],

        "records":
            forecast["series"],
    }

    existing["live"] = live

    DASHBOARD_DATA_FILE.write_text(
        json.dumps(
            existing,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("CLOUD LIVE AQI REFRESH")
    print("=" * 70)

    print(
        f"Location : {CITY} / {AREA}"
    )

    print(
        f"Coordinates : "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        "Source : Open-Meteo Air Quality API"
    )

    payload = fetch_open_meteo()

    live = build_live_intelligence(
        payload
    )

    write_live_file(
        live
    )

    update_dashboard_data(
        live
    )

    print()
    print(
        f"Current AQI : "
        f"{live['current']['aqi']}"
    )

    print(
        f"Category : "
        f"{live['current']['category']}"
    )

    print(
        f"Forecast rows : "
        f"{live['forecast']['rows']}"
    )

    print(
        f"Forecast min : "
        f"{live['forecast']['minimum']:.2f}"
    )

    print(
        f"Forecast max : "
        f"{live['forecast']['maximum']:.2f}"
    )

    print(
        f"Trend : "
        f"{live['forecast']['trend']}"
    )

    print()
    print(
        f"Updated : {LIVE_FILE}"
    )

    print(
        f"Dashboard data : "
        f"{DASHBOARD_DATA_FILE}"
    )

    print()
    print("REFRESH SUCCESS")


if __name__ == "__main__":
    main()