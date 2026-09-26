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


def test_fertilizer_rule_beats_the_lookup_table(app):
    """Sandy / Maize / N=37, P=0, K=0 - the row the 99-row CSV labels 'Urea'.

    The rule disagrees, and it is right to. Maize wants 120-60-40 kg/ha of
    N-P2O5-K2O; all three soil values test low (N=37 << 280, P=0 < 10,
    K=0 < 120), so the STCR adjustment raises the dose 25 % across the board to
    150-75-50. Straight Urea (46-0-0) would pour nitrogen onto a field with
    *zero* available phosphorus and potassium. Cosine similarity against that
    deficit ranks the balanced N+P complexes first (28-28 and 20-20 both score
    0.909, ahead of 17-17-17 at 0.907 and Urea at 0.857); the tie breaks
    towards 28-28 for its higher nutrient concentration. The remaining
    potassium is then supplied by MOP in the dose calculator.
    """
    res = app.ml.recommend_fertilizer(temperature=26, humidity=52, moisture=38, soil_type="Sandy", crop_type="Maize",
                                      N=37, K=0, P=0)
    assert res["recommended_fertilizer"] == "28-28"
    assert res["npk"] == "28-28-0"
    assert res["deficit_kg_per_ha"] == {"N": 150.0, "P2O5": 75.0, "K2O": 50.0}
    assert res["confidence"] == pytest.approx(0.909, abs=0.002)
    assert [r["fertilizer"] for r in res["ranked"]] == ["28-28", "20-20", "17-17-17"]
    assert "28-28" in res["why"]
    # the classifier is still reported - clearly labelled as a hint, never obeyed
    assert res["model_hint"]["fertilizer"] and res["model_hint"]["note"]


def test_fertilizer_phosphorus_deficit_picks_a_phosphorus_product(app):
    """A genuinely phosphorus-led gap picks a phosphorus-led product.

    Pulses map to chickpea, whose base requirement is 20-50-20 (legumes fix
    their own nitrogen). With soil N and K testing high (-25 %) and P testing
    low (+25 %) the gap becomes 15-62.5-15, and the ranking returns 14-35-14
    ahead of DAP.
    """
    res = app.ml.recommend_fertilizer(temperature=29, humidity=52, moisture=45, soil_type="Loamy",
                                      crop_type="Pulses", N=700, K=400, P=0)
    assert res["deficit_kg_per_ha"]["P2O5"] > res["deficit_kg_per_ha"]["N"]
    assert res["recommended_fertilizer"] == "14-35-14"
    assert [r["fertilizer"] for r in res["ranked"][:2]] == ["14-35-14", "DAP"]


def test_soil_test_cannot_reorder_an_n_led_requirement(app):
    """The STCR adjustment scales each nutrient by +-25 %; it cannot flip the ratio.

    Sugarcane needs 250-100-120. Even with soil N testing high and soil P
    testing low, the remaining gap is 187.5-125-90 - still nitrogen-led - so the
    balanced 17-17-17 wins rather than a phosphorus product. This is a real
    limitation of the rule, documented rather than hidden: the crop's base
    requirement dominates the product choice and the soil test only nudges it.
    """
    res = app.ml.recommend_fertilizer(temperature=29, humidity=52, moisture=45, soil_type="Loamy",
                                      crop_type="Sugarcane", N=700, K=400, P=0)
    assert res["deficit_kg_per_ha"] == {"N": 187.5, "P2O5": 125.0, "K2O": 90.0}
    assert res["recommended_fertilizer"] == "17-17-17"


def test_fertilizer_with_area_includes_dose(app):
    res = app.ml.recommend_fertilizer(temperature=26, humidity=52, moisture=38, soil_type="Sandy", crop_type="Maize",
                                      N=37, K=0, P=0, area=2, unit="acre")
    assert res["calculator"]["crop"] == "maize"
    assert res["calculator"]["fertilizers_kg"]["Urea"] > 0


def test_yield_inputs_have_no_production_column():
    from krishidisha.services.ml import YIELD_NUMERIC

    assert "Production" not in YIELD_NUMERIC


def test_yield_model_and_meta(app):
    meta = app.ml.yield_meta
    assert "Wheat" in meta["crops"] and "Punjab" in meta["states"] and "Rabi" in meta["seasons"]
    assert "Coconut" not in meta["crops"]  # nuts/ha, dropped as a non-tonne unit
    assert meta["defaults"]["global"]["fert_per_ha"] > 0
    res = app.ml.predict_yield(crop="Wheat", crop_year=2020, season="Rabi", state="Punjab", area=1000,
                               annual_rainfall=600, fertilizer=150000, pesticide=300)
    assert res["predicted_yield"] > 0
    assert res["estimated_production"] == pytest.approx(res["predicted_yield"] * 1000, rel=0.01)
    lo, hi = res["expected_range"]
    assert 0 <= lo <= res["predicted_yield"] <= hi
    assert res["baseline_yield"] > 0
    assert res["inputs_used"]["source"] == {"fertilizer": "provided", "pesticide": "provided"}


def test_yield_fills_missing_inputs_and_ignores_production(app):
    """Omitted inputs come from regional medians; a stray `production` is ignored."""
    res = app.ml.predict_yield(crop="Wheat", crop_year=2020, season="Rabi", state="Punjab", area=1000,
                               annual_rainfall=600)
    assert res["inputs_used"]["fertilizer"] > 0
    assert res["inputs_used"]["source"]["fertilizer"].endswith("median")
    # `production` is swallowed by **_deprecated - it used to leak the target
    leaky = app.ml.predict_yield(crop="Wheat", crop_year=2020, season="Rabi", state="Punjab", area=1000,
                                 annual_rainfall=600, production=999999)
    assert leaky["predicted_yield"] == res["predicted_yield"]


def test_crop_recommendation_flags_out_of_range_inputs(app):
    normal = app.ml.recommend_crop(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)
    assert normal["warnings"] == []
    weird = app.ml.recommend_crop(N=900, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)
    assert any("N=900" in w for w in weird["warnings"])


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
