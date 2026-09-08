"""Unit tests for the framework-agnostic service layer."""
from __future__ import annotations

import pytest

from krishidisha.services.knowledge import FERTILIZER_REQUIREMENTS, fertilizer_calculator
from krishidisha.services.market import msp_reference
from krishidisha.services.weather import advisories


# ------------------------------------------------------------ fertilizer calc
def test_fertilizer_calculator_wheat_two_acres():
    res = fertilizer_calculator("wheat", 2, "acre")
    assert res["crop"] == "wheat"
    assert res["area_ha"] == pytest.approx(0.809, abs=0.001)
    # 120 kg N/ha * 0.809 ha ~ 97 kg N; DAP supplies part of it
    assert res["nutrient_requirement_kg"]["N"] == pytest.approx(97.1, abs=0.2)
    assert res["fertilizers_kg"]["DAP"] > 0 and res["fertilizers_kg"]["Urea"] > 0 and res["fertilizers_kg"]["MOP"] > 0
    assert res["bags_50kg"]["Urea"] == pytest.approx(res["fertilizers_kg"]["Urea"] / 50, abs=0.1)
    assert len(res["schedule"]) == 3


def test_fertilizer_calculator_alias_and_soil_adjustment():
    base = fertilizer_calculator("paddy", 1, "hectare")
    low_n = fertilizer_calculator("rice", 1, "hectare", soil_n=100)
    high_n = fertilizer_calculator("rice", 1, "hectare", soil_n=700)
    assert base["crop"] == "rice"
    assert low_n["nutrient_requirement_kg"]["N"] == pytest.approx(base["nutrient_requirement_kg"]["N"] * 1.25)
    assert high_n["nutrient_requirement_kg"]["N"] == pytest.approx(base["nutrient_requirement_kg"]["N"] * 0.75)
    assert low_n["soil_status"]["N"].startswith("low")


def test_fertilizer_calculator_unknown_crop():
    res = fertilizer_calculator("dragonfruit", 1)
    assert "error" in res
    assert "rice" in res["error"]


def test_all_requirement_crops_have_positive_values():
    for crop, req in FERTILIZER_REQUIREMENTS.items():
        assert req["N"] >= 0 and req["P2O5"] > 0 and req["K2O"] > 0, crop


# -------------------------------------------------------------- knowledge base
def test_knowledge_base_loads_all_sources(app):
    kb = app.kb
    kinds = {d.meta.get("kind") for d in kb.docs}
    assert {"disease", "scheme", "crop", "calendar", "pest", "fertilizer"} <= kinds
    assert len(kb.crop_guides) >= 30
    assert len(kb.crop_calendar) >= 40
    assert len(kb.schemes) >= 10


def test_knowledge_search_finds_late_blight(app):
    hits = app.kb.search("tomato late blight treatment", k=3)
    assert hits
    assert any("blight" in h["title"].lower() for h in hits)


def test_disease_by_label_matches_csv_row(app):
    info = app.kb.disease_by_label("Tomato___Late_blight")
    assert info is not None
    assert "blight" in info["name"].lower()
    assert info["prevention"]
    healthy = app.kb.disease_by_label("Apple___healthy")
    assert healthy is not None and "healthy" in healthy["name"].lower()


def test_scheme_lookup_and_calendar(app):
    kb = app.kb
    assert any("PM-KISAN" in s["name"] for s in kb.scheme_lookup("kisan"))
    rows = kb.calendar_for("wheat")
    assert rows and rows[0]["season"] == "Rabi"
    assert kb.calendar_for(season="Zaid")
    guide = kb.crop_guide("Rice")
    assert guide and guide["improved_varieties"]


# ------------------------------------------------------------------ ML models
def test_crop_recommendation_model(app):
    res = app.ml.recommend_crop(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)
    assert res["recommended_crop"] == "rice"
    assert 0 < res["confidence"] <= 1
    assert len(res["alternatives"]) == 2
    assert res["profit_per_acre"] == res["revenue_per_acre"] - res["cost_per_acre"]


def test_fertilizer_model(app):
    res = app.ml.recommend_fertilizer(temperature=26, humidity=52, moisture=38, soil_type="Sandy", crop_type="Maize",
                                      N=37, K=0, P=0)
    assert res["recommended_fertilizer"] == "Urea"
    assert res["npk"] == "46-0-0"


def test_yield_model_and_meta(app):
    meta = app.ml.yield_meta
    assert "Wheat" in meta["crops"] and "Punjab" in meta["states"] and "Rabi" in meta["seasons"]
    res = app.ml.predict_yield(crop="Wheat", crop_year=2020, season="Rabi", state="Punjab", area=1000,
                               production=4000, annual_rainfall=600, fertilizer=150000, pesticide=300)
    assert res["predicted_yield"] > 0
    assert res["estimated_production"] == pytest.approx(res["predicted_yield"] * 1000, rel=0.01)


# ------------------------------------------------------------------ rules bot
def test_rules_bot_fertilizer_dose(app):
    out = app.assistant.chat("How much urea and DAP for 2 acres of wheat?")
    assert out["provider"] == "rules"
    assert "fertilizer_calculator" in out["tools_used"]
    assert "Urea" in out["reply"] and "DAP" in out["reply"]


def test_rules_bot_crop_from_numbers(app):
    out = app.assistant.chat("recommend crop N=90 P=42 K=43 temp=21 humidity=82 ph=6.5 rain=200")
    assert "recommend_crop" in out["tools_used"]
    assert "Rice" in out["reply"]


def test_rules_bot_hindi_greeting_and_schemes(app):
    out = app.assistant.chat("नमस्ते")
    assert "कृषिदिशा" in out["reply"]
    out = app.assistant.chat("tell me about PM-KISAN scheme")
    assert "government_schemes" in out["tools_used"]
    assert "PM-KISAN" in out["reply"]


def test_rules_bot_disease_question_links_products(app):
    out = app.assistant.chat("how to control late blight in tomato")
    assert "blight" in out["reply"].lower()
    assert "/marketplace/product/" in out["reply"]


# -------------------------------------------------------------- disease labels
def test_split_label_handles_every_label_style():
    from krishidisha.services.disease import split_label

    assert split_label("Tomato___Late_blight") == ("Tomato", "Late blight")
    assert split_label("Pepper,_bell___healthy") == ("Pepper bell", "healthy")
    assert split_label("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust")
    assert split_label("Apple : Apple scab") == ("Apple", "Apple scab")
    assert split_label("Bell Pepper with Bacterial Spot") == ("Bell Pepper", "Bacterial Spot")
    assert split_label("Healthy Corn (Maize) Plant") == ("Corn (Maize)", "healthy")
    assert split_label("Healthy Apple") == ("Apple", "healthy")
    assert split_label("Cedar Apple Rust") == ("Apple", "Cedar Apple Rust")
    assert split_label("Tomato Yellow Leaf Curl Virus")[0] == "Tomato"


# --------------------------------------------------------------------- market
def test_msp_reference_filters(app):
    prices = msp_reference(app.config["DATA_DIR"], "wheat")
    assert prices and all("wheat" in p["commodity"].lower() for p in prices)
    assert prices[0]["msp"] == 2585


# -------------------------------------------------------------------- weather
def _day(t_max, t_min, rain, wind=10, cond="Clear sky"):
    return {"t_max": t_max, "t_min": t_min, "rain_mm": rain, "wind_max": wind, "condition": cond}


def test_weather_advisories_rules():
    heavy = advisories({"days": [_day(30, 20, 15), _day(31, 21, 12), _day(29, 20, 0)] + [_day(30, 20, 0)] * 4,
                        "current": {"humidity": 90}})
    titles = [a["title"] for a in heavy]
    assert "Heavy rain in next 3 days" in titles
    assert "High disease pressure" in titles

    frost_heat = advisories({"days": [_day(42, 4, 0)] * 7, "current": {"humidity": 30}})
    titles = [a["title"] for a in frost_heat]
    assert "Heat stress alert" in titles and "Cold / frost risk" in titles and "Dry week" in titles
