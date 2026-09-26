"""Tool registry shared by every LLM provider (and the offline fallback bot).

Each tool has a JSON schema (Anthropic ``input_schema`` shape, also converted
to OpenAI function-calling format) and a Python callable returning a JSON-
serialisable dict.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .knowledge import FERTILIZER_REQUIREMENTS, KnowledgeBase, fertilizer_calculator
from .ml import FERT_CROP_TYPES, SOIL_TYPES, MLService
from . import market, weather

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., dict]

    def anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.input_schema}}


def _num(desc: str) -> dict:
    return {"type": "number", "description": desc}


def build_tools(ml: MLService, kb: KnowledgeBase, data_dir, data_gov_key: str,
                search_products: Callable[..., list[dict]] | None = None) -> list[Tool]:
    tools: list[Tool] = []

    # ------------------------------------------------------------ ML models
    tools.append(Tool(
        "recommend_crop",
        "Recommend the best crop to grow from soil test values and climate. Use when the farmer gives N, P, K, "
        "temperature, humidity, pH and rainfall numbers (ask for any that are missing).",
        {"type": "object", "properties": {
            "N": _num("Soil nitrogen (kg/ha), 0-140"), "P": _num("Soil phosphorus (kg/ha), 5-145"),
            "K": _num("Soil potassium (kg/ha), 5-205"), "temperature": _num("Average temperature in Celsius"),
            "humidity": _num("Relative humidity %"), "ph": _num("Soil pH, 3.5-10"),
            "rainfall": _num("Annual/season rainfall in mm")},
         "required": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]},
        lambda **kw: ml.recommend_crop(**kw),
    ))
    tools.append(Tool(
        "recommend_fertilizer",
        "Recommend a commercial fertilizer (Urea, DAP, 10-26-26, 14-35-14, 17-17-17, 20-20, 28-28) by working out "
        "the crop's nutrient requirement, subtracting what the soil test already supplies, and matching the "
        "remaining N-P2O5-K2O ratio against each product. Returns the pick, a one-line reason, the ranked top 3 "
        "and the kg/ha deficit; pass area to also get the dose in kg and 50-kg bags. "
        f"soil_type must be one of {SOIL_TYPES}; crop_type one of {FERT_CROP_TYPES}. "
        "N, K and P are available soil nutrients in kg/ha (Soil Health Card values).",
        {"type": "object", "properties": {
            "temperature": _num("Temperature in Celsius"), "humidity": _num("Humidity %"),
            "moisture": _num("Soil moisture %"),
            "soil_type": {"type": "string", "enum": SOIL_TYPES},
            "crop_type": {"type": "string", "enum": FERT_CROP_TYPES},
            "N": _num("Available soil nitrogen, kg/ha"), "K": _num("Available soil potassium, kg/ha"),
            "P": _num("Available soil phosphorus, kg/ha"),
            "area": _num("Field area (optional) - adds a dose calculation"),
            "unit": {"type": "string", "enum": ["acre", "hectare"], "description": "Area unit (default acre)"}},
         "required": ["temperature", "humidity", "moisture", "soil_type", "crop_type", "N", "K", "P"]},
        lambda **kw: ml.recommend_fertilizer(**kw),
    ))
    tools.append(Tool(
        "predict_yield",
        "Estimate crop yield (tonnes/hectare) for an Indian state and season from a model trained on 1997-2020 "
        "state-level records. Returns the point estimate, a 10-90% expected_range and baseline_yield (the 5-year "
        "median for that crop and state) so the farmer can see whether the model is saying anything new. "
        "Never invent fertilizer or pesticide figures: leave them out and regional medians are used, and the "
        "result reports which values it fell back on. Do NOT pass production - yield is derived from production "
        "in the source data, so supplying it would just echo the answer back. "
        "Call get_reference_lists first if unsure about valid crop/state/season names.",
        {"type": "object", "properties": {
            "crop": {"type": "string"}, "crop_year": {"type": "integer"},
            "season": {"type": "string", "description": "Kharif, Rabi, Whole Year, Summer, Autumn or Winter"},
            "state": {"type": "string"}, "area": _num("Cultivated area in hectares"),
            "annual_rainfall": _num("Annual rainfall mm"),
            "fertilizer": _num("Total fertilizer used in kg (optional; omit if unknown)"),
            "pesticide": _num("Total pesticide used in kg (optional; omit if unknown)")},
         "required": ["crop", "crop_year", "season", "state", "area", "annual_rainfall"]},
        lambda **kw: ml.predict_yield(**kw),
    ))
    tools.append(Tool(
        "get_reference_lists",
        "Return valid crop, state and season names accepted by predict_yield, plus supported crops for the "
        "fertilizer calculator.",
        {"type": "object", "properties": {}},
        lambda: {**ml.yield_meta, "fertilizer_calculator_crops": sorted(FERTILIZER_REQUIREMENTS)},
    ))
    tools.append(Tool(
        "fertilizer_calculator",
        "Compute how many kg / 50-kg bags of Urea, DAP and MOP a farmer needs for a crop and field area, with a "
        "split-application schedule. Optionally adjust for soil-test values.",
        {"type": "object", "properties": {
            "crop": {"type": "string"}, "area": _num("Field area"),
            "unit": {"type": "string", "enum": ["acre", "hectare"], "description": "Area unit (default acre)"},
            "soil_n": _num("Available soil N kg/ha from soil health card (optional)"),
            "soil_p": _num("Available soil P kg/ha (optional)"), "soil_k": _num("Available soil K kg/ha (optional)")},
         "required": ["crop", "area"]},
        lambda crop, area, unit="acre", soil_n=None, soil_p=None, soil_k=None: fertilizer_calculator(
            crop, float(area), unit, soil_n, soil_p, soil_k),
    ))

    # ----------------------------------------------------------- external
    tools.append(Tool(
        "get_weather",
        "7-day weather forecast and agro-advisories (rain, heat, frost, spraying windows) for any Indian town, "
        "district or village name.",
        {"type": "object", "properties": {"place": {"type": "string", "description": "Town/district, e.g. 'Indore'"},
                                          "days": {"type": "integer", "description": "1-14, default 7"}},
         "required": ["place"]},
        lambda place, days=7: weather.forecast_for_place(place, int(days)),
    ))
    tools.append(Tool(
        "get_mandi_prices",
        "Latest wholesale mandi (APMC) prices in INR per quintal from Agmarknet for a commodity, optionally "
        "filtered by state/district. Falls back to official MSP when live data is unavailable.",
        {"type": "object", "properties": {"commodity": {"type": "string", "description": "e.g. Wheat, Tomato, Soyabean"},
                                          "state": {"type": "string"}, "district": {"type": "string"}},
         "required": ["commodity"]},
        lambda commodity, state=None, district=None: market.mandi_prices(data_gov_key, data_dir, commodity, state,
                                                                         district),
    ))

    # ------------------------------------------------------------ knowledge
    tools.append(Tool(
        "search_knowledge",
        "Search the KrishiDisha agricultural knowledge base (disease guides, crop cultivation guides, pest "
        "management, fertilizer schedules, crop calendar, government schemes). Use for any factual farming question.",
        {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
         "required": ["query"]},
        lambda query, k=4: {"results": kb.search(query, k=int(k))},
    ))
    tools.append(Tool(
        "get_disease_info",
        "Detailed description, cause and treatment steps for a plant disease (PlantVillage catalogue), plus the "
        "recommended supplement product.",
        {"type": "object", "properties": {"disease": {"type": "string", "description": "e.g. 'Tomato late blight'"}},
         "required": ["disease"]},
        lambda disease: kb.disease_by_label(disease) or {"results": kb.search(disease, k=2, kinds=("disease",))},
    ))
    tools.append(Tool(
        "crop_guide",
        "Full cultivation guide for a crop: season, soil, sowing time, seed rate, spacing, irrigation, fertilizer "
        "schedule, pests, harvest, yield, varieties.",
        {"type": "object", "properties": {"crop": {"type": "string"}}, "required": ["crop"]},
        lambda crop: kb.crop_guide(crop) or {"error": f"No guide for {crop}", "available": sorted(kb.crop_guides)},
    ))
    tools.append(Tool(
        "crop_calendar",
        "Sowing and harvesting windows by season and region. Filter by crop, season (Kharif/Rabi/Zaid) or month.",
        {"type": "object", "properties": {"crop": {"type": "string"}, "season": {"type": "string"},
                                          "month": {"type": "string"}}},
        lambda crop=None, season=None, month=None: {"rows": kb.calendar_for(crop, season, month)},
    ))
    tools.append(Tool(
        "government_schemes",
        "Look up Indian central government schemes for farmers (PM-KISAN, PMFBY, KCC, PM-KUSUM, etc.) with "
        "benefits, eligibility and how to apply.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "keyword, or empty for all"}}},
        lambda query=None: {"schemes": kb.scheme_lookup(query)},
    ))

    # ---------------------------------------------------------- marketplace
    if search_products is not None:
        tools.append(Tool(
            "search_products",
            "Search the KrishiDisha marketplace for fertilizers, fungicides, insecticides, seeds, organic inputs and "
            "tools. Returns name, price, unit and a link to buy. Use after diagnosing a disease or recommending a "
            "fertilizer so the farmer can purchase it.",
            {"type": "object", "properties": {
                "query": {"type": "string"}, "category": {"type": "string"},
                "disease": {"type": "string", "description": "PlantVillage label e.g. Tomato___Late_blight"},
                "crop": {"type": "string"}}},
            lambda query=None, category=None, disease=None, crop=None: {
                "products": search_products(query=query, category=category, disease=disease, crop=crop)},
        ))
    return tools


def run_tool(tools: list[Tool], name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool by name. Returns (json_string, is_error)."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        result = tool.fn(**(args or {}))
        return json.dumps(result, default=str)[:12000], False
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"}), True
    except Exception as exc:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return json.dumps({"error": str(exc)}), True
