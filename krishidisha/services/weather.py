"""Weather forecasts and farm advisories via the free Open-Meteo API (no key needed)."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Heavy showers", 82: "Violent showers", 95: "Thunderstorm", 96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}

_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: int, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def geocode(place: str) -> dict[str, Any] | None:
    def _do():
        r = requests.get(GEOCODE_URL, params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=8)
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        g = results[0]
        return {"name": g["name"], "latitude": g["latitude"], "longitude": g["longitude"],
                "admin1": g.get("admin1"), "country": g.get("country")}
    try:
        return _cached(f"geo:{place.lower()}", 86400, _do)
    except Exception as exc:
        log.warning("Geocoding failed for %s: %s", place, exc)
        return None


def forecast(latitude: float, longitude: float, days: int = 7, ttl: int = 900) -> dict[str, Any]:
    days = max(1, min(days, 14))

    def _do():
        params = {
            "latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": days,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,"
                     "wind_speed_10m_max,et0_fao_evapotranspiration",
            "hourly": "soil_moisture_0_to_1cm,soil_temperature_0cm",
        }
        r = requests.get(FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    raw = _cached(f"fc:{round(latitude, 2)},{round(longitude, 2)},{days}", ttl, _do)
    cur = raw.get("current", {})
    daily = raw.get("daily", {})
    out_days = []
    for i, date in enumerate(daily.get("time", [])):
        out_days.append({
            "date": date,
            "condition": WMO_CODES.get(daily["weather_code"][i], "Unknown"),
            "t_max": daily["temperature_2m_max"][i],
            "t_min": daily["temperature_2m_min"][i],
            "rain_mm": daily["precipitation_sum"][i],
            "rain_prob": daily.get("precipitation_probability_max", [None] * 99)[i],
            "wind_max": daily["wind_speed_10m_max"][i],
            "et0": daily.get("et0_fao_evapotranspiration", [None] * 99)[i],
        })
    hourly = raw.get("hourly", {})
    soil_moisture = None
    soil_temp = None
    if hourly.get("soil_moisture_0_to_1cm"):
        vals = [v for v in hourly["soil_moisture_0_to_1cm"][:24] if v is not None]
        soil_moisture = round(sum(vals) / len(vals), 3) if vals else None
        tv = [v for v in hourly.get("soil_temperature_0cm", [])[:24] if v is not None]
        soil_temp = round(sum(tv) / len(tv), 1) if tv else None
    result = {
        "latitude": latitude, "longitude": longitude, "timezone": raw.get("timezone"),
        "current": {
            "temperature": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"), "wind": cur.get("wind_speed_10m"),
            "condition": WMO_CODES.get(cur.get("weather_code"), "Unknown"),
        },
        "soil_moisture_top": soil_moisture,
        "soil_temperature": soil_temp,
        "days": out_days,
    }
    result["advisories"] = advisories(result)
    return result


def forecast_for_place(place: str, days: int = 7) -> dict[str, Any]:
    geo = geocode(place)
    if not geo:
        return {"error": f"Could not find location '{place}'. Try a district or city name."}
    fc = forecast(geo["latitude"], geo["longitude"], days)
    fc["place"] = geo
    return fc


def advisories(fc: dict[str, Any]) -> list[dict[str, str]]:
    """Rule-based agro-advisories derived from the forecast."""
    tips: list[dict[str, str]] = []
    days = fc.get("days", [])
    if not days:
        return tips
    next3 = days[:3]
    rain3 = sum(d["rain_mm"] or 0 for d in next3)
    rain7 = sum(d["rain_mm"] or 0 for d in days)
    max_t = max(d["t_max"] for d in days if d["t_max"] is not None)
    min_t = min(d["t_min"] for d in days if d["t_min"] is not None)
    windy = any((d["wind_max"] or 0) > 35 for d in next3)
    storm = any("hunder" in d["condition"] for d in next3)

    if rain3 >= 20:
        tips.append({"level": "warning", "title": "Heavy rain in next 3 days",
                     "text": f"About {rain3:.0f} mm expected. Postpone fertilizer top-dressing and pesticide sprays; "
                             "ensure field drainage to avoid waterlogging."})
    elif rain3 >= 5:
        tips.append({"level": "info", "title": "Light showers expected",
                     "text": "Good window for transplanting and basal fertilizer application. Avoid spraying just before rain."})
    else:
        tips.append({"level": "info", "title": "Dry spell ahead",
                     "text": "No significant rain in the next 3 days. Schedule irrigation for crops at critical growth stages "
                             "and use mulch to conserve soil moisture."})
    if rain7 < 5:
        tips.append({"level": "warning", "title": "Dry week",
                     "text": "Less than 5 mm rain expected this week. Prioritise irrigation for vegetables and flowering crops."})
    if max_t >= 40:
        tips.append({"level": "danger", "title": "Heat stress alert",
                     "text": f"Temperatures up to {max_t:.0f}°C. Irrigate in the evening, apply light irrigation to standing crops "
                             "and provide shade nets for nurseries."})
    if min_t <= 5:
        tips.append({"level": "danger", "title": "Cold / frost risk",
                     "text": f"Night temperature may drop to {min_t:.0f}°C. Light irrigation in the evening and smoke can protect "
                             "against frost in wheat, potato and mustard."})
    if windy or storm:
        tips.append({"level": "warning", "title": "Strong wind / thunderstorm",
                     "text": "Stake tall crops (banana, papaya, maize), delay spraying, and harvest mature produce early."})
    hum = fc.get("current", {}).get("humidity")
    if hum and hum > 85 and rain3 > 0:
        tips.append({"level": "warning", "title": "High disease pressure",
                     "text": "Warm, humid and wet conditions favour fungal diseases (blight, rust, mildew). Scout fields daily "
                             "and keep protective fungicide ready."})
    return tips
