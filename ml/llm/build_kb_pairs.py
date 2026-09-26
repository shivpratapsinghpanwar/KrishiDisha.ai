"""Deterministic instruction pairs and tool trajectories from KrishiDisha's own knowledge base.

    python -m ml.llm.build_kb_pairs --out data/llm/kb_pairs.jsonl [--seed 42] [--max-per-topic 3]

Sources: crop guides (32 crops x 12 fields), pest management (41), disease catalogue (39), government
schemes (16), crop calendar (52), fertilizer requirement table; plus *real* tool trajectories where the
assistant calls ``fertilizer_calculator``, ``crop_calendar``, ``government_schemes``, ``crop_guide``,
``get_disease_info``, ``get_reference_lists``, ``recommend_crop`` and ``predict_yield`` with valid inputs,
the tool actually runs inside the app, and the answer is rendered from the tool's output. This is the
cheapest way to teach a small model the exact tool-call format and to ground it in our data.

Languages: English, Hindi (Devanagari) and Hinglish templates written by hand below.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

from .common import (DATA_DIR, app_context, assistant_tool_call, make_example, openai_tool_schemas, run_tool_json,
                     tool_result_turn, write_jsonl)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- templates
# {crop} = crop name as the farmer would write it; {hcrop} = Hindi name when known.
HINDI_CROPS = {"rice": "धान", "wheat": "गेहूं", "maize": "मक्का", "cotton": "कपास", "sugarcane": "गन्ना", "soybean": "सोयाबीन",
               "groundnut": "मूंगफली", "mustard": "सरसों", "chickpea": "चना", "pigeonpeas": "अरहर", "lentil": "मसूर",
               "potato": "आलू", "tomato": "टमाटर", "onion": "प्याज", "banana": "केला", "mango": "आम", "grapes": "अंगूर",
               "apple": "सेब", "coffee": "कॉफी", "jute": "जूट", "millets": "बाजरा", "barley": "जौ", "watermelon": "तरबूज",
               "muskmelon": "खरबूजा", "papaya": "पपीता", "coconut": "नारियल", "orange": "संतरा", "pomegranate": "अनार",
               "blackgram": "उड़द", "mungbean": "मूंग", "mothbeans": "मोठ", "kidneybeans": "राजमा"}
HINGLISH_CROPS = {"rice": "dhaan", "wheat": "gehu", "maize": "makka", "cotton": "kapas", "sugarcane": "ganna",
                  "groundnut": "moongphali", "mustard": "sarson", "chickpea": "chana", "pigeonpeas": "arhar",
                  "potato": "aloo", "tomato": "tamatar", "onion": "pyaz", "banana": "kela", "mango": "aam",
                  "blackgram": "urad", "mungbean": "moong", "millets": "bajra"}

GUIDE_FIELDS = {
    "season": {"en": ["When is {crop} grown in India?", "Which season is {crop} sown in?"],
               "hi": ["{hcrop} किस मौसम में बोया जाता है?", "भारत में {hcrop} की खेती कब होती है?"],
               "hinglish": ["{hcrop} kis season me bote hain?", "{hcrop} ki kheti kab hoti hai?"]},
    "climate": {"en": ["What climate does {crop} need?", "Ideal temperature and rainfall for {crop}?"],
                "hi": ["{hcrop} के लिए कैसा मौसम चाहिए?"], "hinglish": ["{hcrop} ke liye kaisa climate chahiye?"]},
    "soil": {"en": ["Which soil is best for {crop}?", "What soil type and pH suits {crop}?"],
             "hi": ["{hcrop} के लिए कौन सी मिट्टी अच्छी है?"], "hinglish": ["{hcrop} ke liye kaunsi mitti best hai?"]},
    "sowing_time": {"en": ["When should I sow {crop}?", "Best sowing time for {crop}?"],
                    "hi": ["{hcrop} की बुवाई कब करें?"], "hinglish": ["{hcrop} kab bona chahiye?"]},
    "seed_rate": {"en": ["What is the seed rate for {crop}?", "How much seed per hectare for {crop}?"],
                  "hi": ["{hcrop} का बीज दर कितना है?"], "hinglish": ["{hcrop} me per hectare kitna beej lagega?"]},
    "spacing": {"en": ["What spacing should I keep for {crop}?"], "hi": ["{hcrop} में पौधों की दूरी कितनी रखें?"],
                "hinglish": ["{hcrop} me spacing kitni rakhein?"]},
    "irrigation": {"en": ["How many irrigations does {crop} need and when?", "Irrigation schedule for {crop}?"],
                   "hi": ["{hcrop} में कितनी सिंचाई करनी चाहिए?"], "hinglish": ["{hcrop} me kitni sinchai lagti hai?"]},
    "fertilizer_schedule": {"en": ["What fertilizer schedule should I follow for {crop}?", "NPK dose for {crop}?"],
                            "hi": ["{hcrop} में खाद कब और कितनी डालें?"], "hinglish": ["{hcrop} me khad kab aur kitni daalein?"]},
    "major_pests_diseases": {"en": ["What are the major pests and diseases of {crop}?"],
                             "hi": ["{hcrop} में मुख्य कीट और रोग कौन से हैं?"], "hinglish": ["{hcrop} me kaun se rog aur keede lagte hain?"]},
    "harvest": {"en": ["When and how should I harvest {crop}?"], "hi": ["{hcrop} की कटाई कब करें?"],
                "hinglish": ["{hcrop} ki katai kab karein?"]},
    "expected_yield": {"en": ["What yield can I expect from {crop}?"], "hi": ["{hcrop} की पैदावार कितनी होती है?"],
                       "hinglish": ["{hcrop} ki paidawar kitni hoti hai?"]},
    "improved_varieties": {"en": ["Which improved varieties of {crop} should I grow?", "Recommend {crop} varieties."],
                           "hi": ["{hcrop} की उन्नत किस्में कौन सी हैं?"], "hinglish": ["{hcrop} ki acchi variety batao."]},
}
FIELD_LABEL = {"en": {k: k.replace("_", " ") for k in GUIDE_FIELDS},
               "hi": {"season": "मौसम", "climate": "जलवायु", "soil": "मिट्टी", "sowing_time": "बुवाई का समय", "seed_rate": "बीज दर",
                      "spacing": "दूरी", "irrigation": "सिंचाई", "fertilizer_schedule": "खाद", "major_pests_diseases": "कीट व रोग",
                      "harvest": "कटाई", "expected_yield": "पैदावार", "improved_varieties": "उन्नत किस्में"}}
FIELD_LABEL["hinglish"] = FIELD_LABEL["en"]

ANSWER_PREFIX = {"en": ["For {crop}, {label}: ", "{Crop} ({label}): "],
                 "hi": ["{hcrop} ({label}): "], "hinglish": ["{hcrop} ke liye {label}: "]}
CONFIRM_KVK = {"en": " Confirm doses with your nearest KVK or a soil test.",
               "hi": " सही मात्रा के लिए नज़दीकी कृषि विज्ञान केंद्र या मिट्टी जांच की सलाह लें।",
               "hinglish": " Sahi dose ke liye apne KVK ya soil test se confirm karein."}

PEST_Q = {"en": ["How do I control {name} in {crops}?", "My {crops} has {name}. What should I do?", "Symptoms and control of {name}?"],
          "hi": ["{crops} में {name} का नियंत्रण कैसे करें?", "{name} के लक्षण और उपाय बताइए।"],
          "hinglish": ["{crops} me {name} ka control kaise karein?", "{name} ke lakshan aur ilaj batao."]}
DISEASE_Q = {"en": ["What is {name} and how do I treat it?", "How to prevent {name}?"],
             "hi": ["{name} क्या है और इसका इलाज कैसे करें?"], "hinglish": ["{name} kya hai aur iska ilaj kya hai?"]}
SCHEME_Q = {"en": ["Tell me about {name}.", "Am I eligible for {name} and how do I apply?", "What is the benefit of {name}?"],
            "hi": ["{name} के बारे में बताइए।", "{name} में कौन पात्र है और आवेदन कैसे करें?"],
            "hinglish": ["{name} kya hai?", "{name} ke liye apply kaise karein?"]}

FERT_CALC_Q = {"en": ["How much urea and DAP do I need for {area} {unit} of {crop}?", "Fertilizer dose for {area} {unit} {crop}?",
                      "I have {area} {unit} of {crop}. How many bags of fertilizer?"],
               "hi": ["{area} {hunit} {hcrop} के लिए कितना यूरिया और डीएपी चाहिए?", "{area} {hunit} {hcrop} में कितनी खाद डालें?"],
               "hinglish": ["{area} {unit} {hcrop} ke liye kitna urea aur DAP lagega?", "{area} {unit} {hcrop} me kitne bag khad chahiye?"]}
CAL_Q = {"en": ["When should I sow and harvest {crop}?", "What is the sowing window for {crop} in {season}?"],
         "hi": ["{hcrop} कब बोएं और कब काटें?"], "hinglish": ["{hcrop} kab bona aur kab kaatna hai?"]}
GUIDE_TOOL_Q = {"en": ["Give me a complete cultivation guide for {crop}.", "How do I grow {crop} from sowing to harvest?"],
                "hi": ["{hcrop} की पूरी खेती की जानकारी दीजिए।"], "hinglish": ["{hcrop} ki kheti kaise karein, pura bata do."]}
REC_Q = {"en": ["My soil test shows N={N}, P={P}, K={K}, pH {ph}. Temperature is {t} C, humidity {h}% and rainfall {r} mm. Which crop should I grow?",
                "Recommend a crop: N {N}, P {P}, K {K}, temp {t}, humidity {h}, ph {ph}, rain {r}."],
         "hi": ["मिट्टी जांच में N={N}, P={P}, K={K}, pH {ph} आया है। तापमान {t} डिग्री, नमी {h}% और वर्षा {r} मिमी है। कौन सी फसल लगाऊं?"],
         "hinglish": ["Soil test me N={N}, P={P}, K={K}, pH {ph} hai; temp {t}, humidity {h}, rain {r} mm. Kaunsi fasal lagau?"]}


def _fmt(template: str, **kw) -> str:
    kw.setdefault("Crop", kw.get("crop", "").title())
    return template.format(**kw)


def _crop_names(crop: str, lang: str) -> dict[str, str]:
    return {"crop": crop, "hcrop": HINDI_CROPS.get(crop, crop) if lang == "hi" else
            HINGLISH_CROPS.get(crop, crop) if lang == "hinglish" else crop}


# --------------------------------------------------------------------------- generators
def kb_pairs(app, rng: random.Random, max_per_topic: int) -> list[dict]:
    kb = app.kb
    out: list[dict] = []

    # crop guides: one Q/A per (crop, field, language)
    for crop, guide in kb.crop_guides.items():
        for field, qs in GUIDE_FIELDS.items():
            value = guide.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            for lang in ("en", "hi", "hinglish"):
                names = _crop_names(crop, lang)
                for q in rng.sample(qs[lang], min(len(qs[lang]), max_per_topic)):
                    label = FIELD_LABEL[lang][field]
                    ans = _fmt(rng.choice(ANSWER_PREFIX[lang]), label=label, **names) + value.strip()
                    if field in ("fertilizer_schedule", "seed_rate"):
                        ans += CONFIRM_KVK[lang]
                    out.append(make_example("kb_guide", lang, [{"role": "user", "content": _fmt(q, **names)},
                                                                {"role": "assistant", "content": ans}],
                                            meta={"crop": crop, "field": field}))

    # pest management
    for p in kb.pests:
        crops = p.get("crops", "")
        for lang in ("en", "hi", "hinglish"):
            for q in rng.sample(PEST_Q[lang], min(len(PEST_Q[lang]), max_per_topic)):
                ans = (f"**{p['name']}** ({crops})\n\n**Symptoms:** {p['symptoms']}\n\n**Management:** {p['management']}"
                       + CONFIRM_KVK[lang])
                out.append(make_example("kb_pest", lang, [{"role": "user", "content": _fmt(q, name=p["name"], crops=crops.split(",")[0].strip())},
                                                          {"role": "assistant", "content": ans}], meta={"pest": p["id"]}))

    # disease catalogue (PlantVillage names, English answers; Hindi question with English body is realistic)
    for d in kb.all_diseases():
        name = str(d["name"]).replace(" : ", " ").replace("_", " ")
        if "healthy" in name.lower() or "background" in name.lower():
            continue
        for lang in ("en", "hi", "hinglish"):
            q = rng.choice(DISEASE_Q[lang])
            ans = f"**{name}**\n\n{d['description']}\n\n**Prevention and treatment:** {d['prevention']}" + CONFIRM_KVK[lang]
            out.append(make_example("kb_disease", lang, [{"role": "user", "content": _fmt(q, name=name)},
                                                         {"role": "assistant", "content": ans}], meta={"disease": d["index"]}))

    # schemes
    for s in kb.schemes:
        for lang in ("en", "hi", "hinglish"):
            for q in rng.sample(SCHEME_Q[lang], min(len(SCHEME_Q[lang]), max_per_topic)):
                ans = (f"**{s['name']}** ({s.get('ministry', '')})\n\n{s['summary']}\n\n**Benefit:** {s.get('benefit', '')}\n\n"
                       f"**Eligibility:** {s.get('eligibility', '')}\n\n**How to apply:** {s.get('how_to_apply', '')}"
                       + (f"\n\nOfficial site: {s['url']}" if s.get("url") else ""))
                out.append(make_example("kb_scheme", lang, [{"role": "user", "content": _fmt(q, name=s["name"])},
                                                            {"role": "assistant", "content": ans}], meta={"scheme": s["id"]}))
    return out


def tool_trajectories(app, rng: random.Random, n_calc: int = 300, n_rec: int = 150, n_yield: int = 100) -> list[dict]:
    """Real tool calls executed in the app; answers rendered from the tool output."""
    tools = openai_tool_schemas(app)
    out: list[dict] = []
    from krishidisha.services.knowledge import FERTILIZER_REQUIREMENTS

    skipped = {"n": 0}

    def add(source, lang, user, call_name, args, render, meta=None):
        result, is_err = run_tool_json(app, call_name, args)
        # a str result means the JSON was truncated by run_tool's size cap -> not a clean training example
        if is_err or not isinstance(result, (dict, list)) or (isinstance(result, dict) and "error" in result):
            skipped["n"] += 1
            return
        try:
            answer = render(result)
        except Exception as exc:  # noqa: BLE001 - unexpected tool output shape
            log.warning("render failed for %s %s: %s", call_name, args, exc)
            skipped["n"] += 1
            return
        call = assistant_tool_call(call_name, args)
        msgs = [{"role": "user", "content": user}, call, tool_result_turn(call, result),
                {"role": "assistant", "content": answer}]
        out.append(make_example(source, lang, msgs, tools=tools, meta=meta))

    # fertilizer calculator
    crops = sorted(FERTILIZER_REQUIREMENTS)
    for _ in range(n_calc):
        crop = rng.choice(crops)
        area = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 8, 10])
        unit = rng.choice(["acre", "acre", "hectare"])
        lang = rng.choice(["en", "hi", "hinglish"])
        names = _crop_names(crop, lang)
        hunit = "एकड़" if unit == "acre" else "हेक्टेयर"
        q = _fmt(rng.choice(FERT_CALC_Q[lang]), area=f"{area:g}", unit=unit, hunit=hunit, **names)

        def render(r, lang=lang, crop=crop, area=area, unit=unit):
            kg, bags = r["fertilizers_kg"], r["bags_50kg"]
            head = {"en": f"For {area:g} {unit} of {crop} (about {r['area_ha']} ha):",
                    "hi": f"{area:g} {hunit} {HINDI_CROPS.get(crop, crop)} (लगभग {r['area_ha']} हेक्टेयर) के लिए:",
                    "hinglish": f"{area:g} {unit} {HINGLISH_CROPS.get(crop, crop)} (approx {r['area_ha']} ha) ke liye:"}[lang]
            sched = {"en": "Schedule", "hi": "समय", "hinglish": "Schedule"}[lang]
            return (f"{head}\n- Urea: {kg['Urea']} kg (~{bags['Urea']} bags)\n- DAP: {kg['DAP']} kg (~{bags['DAP']} bags)\n"
                    f"- MOP: {kg['MOP']} kg (~{bags['MOP']} bags)\n\n**{sched}:**\n" + "\n".join(f"- {s}" for s in r["schedule"])
                    + "\n\n" + r["note"] + CONFIRM_KVK[lang])
        add("tool_fert_calc", lang, q, "fertilizer_calculator", {"crop": crop, "area": area, "unit": unit}, render,
            meta={"crop": crop})

    # crop calendar
    for c in app.kb.crop_calendar:
        crop = c["crop"].lower()
        for lang in ("en", "hi", "hinglish"):
            names = _crop_names(crop, lang)
            q = _fmt(rng.choice(CAL_Q[lang]), season=c["season"], **names)

            def render(r, lang=lang):
                rows = r.get("rows", [])[:4]
                head = {"en": "Sowing and harvest windows:", "hi": "बुवाई और कटाई का समय:", "hinglish": "Bone aur kaatne ka time:"}[lang]
                return head + "\n" + "\n".join(f"- **{x['crop']}** ({x['season']}): sow {x['sowing']}, harvest {x['harvest']} "
                                               f"({x.get('duration_days', '?')} days; {x.get('regions', '')})" for x in rows)
            add("tool_calendar", lang, q, "crop_calendar", {"crop": c["crop"]}, render, meta={"crop": crop})

    # full crop guide via tool
    for crop in sorted(app.kb.crop_guides):
        for lang in ("en", "hi", "hinglish"):
            names = _crop_names(crop, lang)
            q = _fmt(rng.choice(GUIDE_TOOL_Q[lang]), **names)

            def render(g, crop=crop, lang=lang):
                keys = ["season", "soil", "sowing_time", "seed_rate", "spacing", "irrigation", "fertilizer_schedule",
                        "major_pests_diseases", "harvest", "expected_yield", "improved_varieties"]
                return f"**{crop.title()} cultivation guide**\n" + "\n".join(
                    f"- **{FIELD_LABEL[lang].get(k, k).title()}:** {g[k]}" for k in keys if g.get(k)) + CONFIRM_KVK[lang]
            add("tool_guide", lang, q, "crop_guide", {"crop": crop}, render, meta={"crop": crop})

    # schemes via tool
    for s in app.kb.schemes:
        lang = rng.choice(["en", "hi", "hinglish"])
        q = _fmt(rng.choice(SCHEME_Q[lang]), name=s["name"])
        key = s["name"].split("(")[-1].strip(")") if "(" in s["name"] else s["name"].split()[0]

        def render(r, s=s):
            hit = next((x for x in r.get("schemes", []) if isinstance(x, dict) and x.get("id") == s["id"]), s)
            return (f"**{hit.get('name')}**\n\n{hit.get('summary', '')}\n\n**Benefit:** {hit.get('benefit', '')}\n\n"
                    f"**Eligibility:** {hit.get('eligibility', '')}\n\n**Apply:** {hit.get('how_to_apply', '')}")
        add("tool_scheme", lang, q, "government_schemes", {"query": key}, render, meta={"scheme": s["id"]})

    # disease info via tool
    for d in app.kb.all_diseases():
        name = str(d["name"])
        if "healthy" in name.lower() or "background" in name.lower():
            continue
        lang = rng.choice(["en", "hi", "hinglish"])
        pretty = name.replace(" : ", " ").replace("_", " ")
        q = _fmt(rng.choice(DISEASE_Q[lang]), name=pretty)

        def render(r):
            if "name" in r:
                return f"**{r['name']}**\n\n{r['description']}\n\n**Prevention and treatment:** {r['prevention']}"
            hits = r.get("results", [])
            return hits[0]["text"] if hits else "I could not find that disease in the catalogue."
        add("tool_disease", lang, q, "get_disease_info", {"disease": pretty}, render, meta={"disease": d["index"]})

    # crop recommendation from numbers (sample plausible rows from the training CSV)
    import pandas as pd

    df = pd.read_csv(Path(app.config["DATA_DIR"]) / "Crop_recommendation.csv")
    for _, row in df.sample(n=min(n_rec, len(df)), random_state=rng.randint(0, 10**6)).iterrows():
        lang = rng.choice(["en", "hi", "hinglish"])
        args = {"N": float(row.N), "P": float(row.P), "K": float(row.K), "temperature": round(float(row.temperature), 1),
                "humidity": round(float(row.humidity), 1), "ph": round(float(row.ph), 2), "rainfall": round(float(row.rainfall), 1)}
        q = _fmt(rng.choice(REC_Q[lang]), N=args["N"], P=args["P"], K=args["K"], ph=args["ph"], t=args["temperature"],
                 h=args["humidity"], r=args["rainfall"])

        def render(r, lang=lang):
            alts = ", ".join(f"{a['crop']} ({a['probability'] * 100:.0f}%)" for a in r.get("alternatives", []))
            head = {"en": "Recommended crop", "hi": "अनुशंसित फसल", "hinglish": "Recommended fasal"}[lang]
            econ = ""
            if r.get("revenue_per_acre"):
                econ = f"\nIndicative economics: revenue ₹{r['revenue_per_acre']:,}/acre, cost ₹{r['cost_per_acre']:,}/acre."
            return f"**{head}: {r['recommended_crop'].title()}** ({r['confidence'] * 100:.0f}% confidence)\nAlternatives: {alts}.{econ}" + CONFIRM_KVK[lang]
        add("tool_recommend_crop", lang, q, "recommend_crop", args, render)

    # yield prediction (whatever signature the tool currently has - read it from the schema)
    yschema = next((t for t in tools if t["function"]["name"] == "predict_yield"), None)
    if yschema:
        meta = app.ml.yield_meta
        required = yschema["function"]["parameters"].get("required", [])
        for _ in range(n_yield):
            crop, state, season = rng.choice(meta["crops"]), rng.choice(meta["states"]), rng.choice(meta["seasons"])
            area = rng.choice([1, 2, 5, 10, 20, 50])
            args = {"crop": crop, "crop_year": rng.choice([2022, 2023, 2024]), "season": season, "state": state,
                    "area": area, "annual_rainfall": rng.choice([500, 700, 900, 1100, 1400])}
            if "production" in required:
                args["production"] = area * 3
            if "fertilizer" in required:
                args["fertilizer"] = area * 150
            if "pesticide" in required:
                args["pesticide"] = area * 0.5
            q = (f"I grow {crop} in {state} in the {season.strip()} season on {area} hectares with about {args['annual_rainfall']} mm "
                 f"rainfall. What yield can I expect?")

            def render(r):
                rng_txt = f" (likely range {r['expected_range'][0]}-{r['expected_range'][1]})" if r.get("expected_range") else ""
                base = f" The 5-year state average is about {r['baseline_yield']} t/ha." if r.get("baseline_yield") else ""
                tips = ("\n" + "\n".join(f"- {t}" for t in r.get("tips", []))) if r.get("tips") else ""
                return (f"Expected yield: **{r['predicted_yield']} {r['unit']}**{rng_txt}, i.e. about {r['estimated_production']} tonnes "
                        f"from your field.{base}{tips}\n\nThis is a model estimate from state-level historical data; your field's soil, "
                        f"variety and management matter more.")
            add("tool_yield", "en", q, "predict_yield", args, render)
    if skipped["n"]:
        log.warning("%d tool trajectories skipped (tool error, truncated output or render failure)", skipped["n"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DATA_DIR / "kb_pairs.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-topic", type=int, default=2)
    p.add_argument("--n-calc", type=int, default=300)
    p.add_argument("--n-rec", type=int, default=150)
    p.add_argument("--n-yield", type=int, default=100)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)
    with app_context() as app:
        rows = kb_pairs(app, rng, args.max_per_topic)
        n_kb = len(rows)
        rows += tool_trajectories(app, rng, args.n_calc, args.n_rec, args.n_yield)
    # de-duplicate by id
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    write_jsonl(args.out, uniq)
    by_src = {}
    by_lang = {}
    for r in uniq:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
    print(f"wrote {args.out}: {len(uniq)} examples ({n_kb} KB pairs, {len(uniq) - n_kb} tool trajectories)")
    print("by source:", json.dumps(by_src, indent=1))
    print("by language:", json.dumps(by_lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
