"""Mandi (wholesale market) prices.

Primary source: data.gov.in "Current Daily Price of Various Commodities from
Various Markets (Mandi)" - Agmarknet feed, resource 9ef84268-d588-465a-a308-a864a43d0070.
Falls back to bundled MSP / reference prices when the API is unreachable.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE}"

_cache: dict[str, tuple[float, Any]] = {}
_FALLBACK: dict[str, Any] | None = None


def _load_fallback(data_dir: Path) -> dict[str, Any]:
    global _FALLBACK
    if _FALLBACK is None:
        path = Path(data_dir) / "knowledge" / "msp_2025_26.json"
        try:
            _FALLBACK = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            log.warning("MSP fallback missing: %s", exc)
            _FALLBACK = {"season": "", "prices": []}
    return _FALLBACK


def mandi_prices(api_key: str, data_dir: Path, commodity: str | None = None, state: str | None = None,
                 district: str | None = None, limit: int = 20) -> dict[str, Any]:
    key = f"{commodity}|{state}|{district}|{limit}".lower()
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < 1800:
        return hit[1]

    params: dict[str, Any] = {"api-key": api_key, "format": "json", "limit": limit}
    if commodity:
        params["filters[commodity]"] = commodity.strip().title()
    if state:
        params["filters[state]"] = state.strip().title()
    if district:
        params["filters[district]"] = district.strip().title()
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        records = payload.get("records", [])
        rows = [{
            "commodity": rec.get("commodity"), "variety": rec.get("variety"), "state": rec.get("state"),
            "district": rec.get("district"), "market": rec.get("market"), "date": rec.get("arrival_date"),
            "min_price": _num(rec.get("min_price")), "max_price": _num(rec.get("max_price")),
            "modal_price": _num(rec.get("modal_price")), "unit": "INR per quintal",
        } for rec in records]
        result = {"source": "data.gov.in Agmarknet (live)", "count": len(rows), "records": rows}
        if not rows:
            result["note"] = "No live records matched; showing MSP reference prices."
            result["msp"] = msp_reference(data_dir, commodity)
    except Exception as exc:
        log.warning("Mandi API failed: %s", exc)
        result = {"source": "offline MSP reference (live API unavailable)", "count": 0, "records": [],
                  "msp": msp_reference(data_dir, commodity), "error": str(exc)}
        # Cache a failure only briefly (2 min) so the live feed is retried soon; successes keep 30 min.
        _cache[key] = (now - 1800 + 120, result)
        return result
    _cache[key] = (now, result)
    return result


def msp_reference(data_dir: Path, commodity: str | None = None) -> list[dict[str, Any]]:
    fb = _load_fallback(data_dir)
    prices = fb.get("prices", [])
    if commodity:
        c = commodity.lower()
        prices = [p for p in prices if c in p["commodity"].lower()] or prices
    return prices


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
