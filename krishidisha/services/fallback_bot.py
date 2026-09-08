"""Offline rule-based assistant.

Used when no LLM API key is configured or the provider fails. It is far more
capable than the original keyword bot because it drives the same tools the
LLM uses: it extracts numbers/places/crops from the message, calls the ML
models, weather and price services, and searches the knowledge base.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .tools import Tool, run_tool

GREETINGS = ("hello", "hi ", "hi", "hey", "namaste", "namaskar", "good morning", "good evening", "pranam",
             "नमस्ते", "नमस्कार", "प्रणाम", "हेलो", "हाय", "राम राम", "सत श्री अकाल", "sat sri akal", "ram ram")
THANKS = ("thank", "dhanyavad", "shukriya", "bye", "goodbye", "धन्यवाद", "शुक्रिया", "अलविदा")

KNOWN_CROPS = ["rice", "paddy", "wheat", "maize", "corn", "cotton", "sugarcane", "soybean", "groundnut", "mustard",
               "chickpea", "gram", "pigeonpea", "arhar", "tur", "lentil", "masur", "mungbean", "moong", "blackgram",
               "urad", "potato", "tomato", "onion", "banana", "mango", "grapes", "apple", "papaya", "coffee", "jute",
               "coconut", "orange", "pomegranate", "watermelon", "muskmelon", "chilli", "okra", "brinjal", "millet",
               "bajra", "jowar", "ragi", "barley", "kidneybeans", "rajma"]

HINDI_HINTS = {
    "en": {"greet": "Namaste! I am KrishiDisha Sahayak. Ask me about crops, fertilizer doses, diseases, weather, mandi "
                    "prices, government schemes or products in our marketplace.",
           "bye": "Thank you for using KrishiDisha. Jai Kisan! 🌾",
           "unknown": "I could not find an exact answer. Try asking about a crop, a disease, weather for your district, "
                      "mandi prices, or fertilizer dose for your field.",
           "need_npk": "To recommend a crop I need seven values: N, P, K (kg/ha), temperature (°C), humidity (%), soil pH "
                       "and rainfall (mm). Example: 'recommend crop N=90 P=42 K=43 temp=21 humidity=82 ph=6.5 rain=200'."},
    "hi": {"greet": "नमस्ते! मैं कृषिदिशा सहायक हूँ। फसल, खाद की मात्रा, रोग, मौसम, मंडी भाव, सरकारी योजनाओं या हमारे मार्केटप्लेस "
                    "के उत्पादों के बारे में पूछें।",
           "bye": "कृषिदिशा का उपयोग करने के लिए धन्यवाद। जय किसान! 🌾",
           "unknown": "मुझे सटीक उत्तर नहीं मिला। किसी फसल, रोग, अपने जिले के मौसम, मंडी भाव या खाद की मात्रा के बारे में पूछें।",
           "need_npk": "फसल सुझाने के लिए मुझे सात मान चाहिए: N, P, K (किग्रा/हे.), तापमान (°C), नमी (%), मिट्टी का pH और वर्षा (मिमी)।"},
}


class RulesBot:
    def __init__(self, tools: list[Tool], kb):
        self.tools = tools
        self.kb = kb

    # ---------------------------------------------------------------- utils
    def _t(self, name: str, **args) -> dict[str, Any]:
        out, _ = run_tool(self.tools, name, args)
        return json.loads(out)

    @staticmethod
    def _lang(language: str, text: str) -> str:
        if language.startswith("hi") or re.search(r"[ऀ-ॿ]", text):
            return "hi"
        return "en"

    @staticmethod
    def _find_crop(text: str) -> str | None:
        t = text.lower()
        for c in KNOWN_CROPS:
            if re.search(rf"\b{c}s?\b", t):
                return c
        return None

    @staticmethod
    def _find_numbers(text: str) -> dict[str, float]:
        """Parse 'N=90 P 42 temp:21.5 ...' style inputs."""
        keys = {"n": "N", "nitrogen": "N", "p": "P", "phosphorus": "P", "phosphorous": "P", "k": "K", "potassium": "K",
                "potash": "K", "temp": "temperature", "temperature": "temperature", "humidity": "humidity",
                "hum": "humidity", "ph": "ph", "rain": "rainfall", "rainfall": "rainfall", "moisture": "moisture"}
        found: dict[str, float] = {}
        for m in re.finditer(r"([A-Za-z]+)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text):
            k = m.group(1).lower()
            if k in keys:
                found[keys[k]] = float(m.group(2))
        return found

    @staticmethod
    def _find_place(text: str) -> str | None:
        m = re.search(r"(?:in|at|for|near|of)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", text)
        if m:
            return m.group(1)
        m = re.search(r"(?:weather|mausam|forecast|rain)\s+(?:in\s+|at\s+)?([a-zA-Z]{3,}(?:\s[a-zA-Z]{3,})?)", text, re.I)
        if m and m.group(1).lower() not in {"today", "tomorrow", "this", "next", "week", "forecast"}:
            return m.group(1)
        return None

    @staticmethod
    def _find_area(text: str) -> tuple[float, str] | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(acre|acres|hectare|hectares|ha|bigha)", text.lower())
        if not m:
            return None
        val, unit = float(m.group(1)), m.group(2)
        if unit.startswith("bigha"):
            return val * 0.62, "acre"  # approx (varies by state)
        return val, ("acre" if unit.startswith("acre") else "hectare")

    # ---------------------------------------------------------------- reply
    def reply(self, message: str, language: str = "en", detection: dict | None = None,
              farmer_context: str | None = None) -> dict[str, Any]:
        text = message.strip()
        low = text.lower()
        lang = self._lang(language, text)
        L = HINDI_HINTS[lang]
        tools_used: list[str] = []
        sources: list[str] = []

        if detection and detection.get("available"):
            top = detection["top"]
            info = self.kb.disease_by_label(top["label"]) or {}
            lines = [f"**Leaf analysis:** {top['name']} ({top['confidence'] * 100:.1f}% confidence)"]
            if top.get("is_healthy"):
                lines.append("The leaf looks healthy. Keep monitoring and follow the crop's fertilizer schedule.")
            else:
                if info.get("description"):
                    lines.append(info["description"][:700])
                if info.get("prevention"):
                    lines.append("**What to do:** " + info["prevention"][:700])
            prods = self._t("search_products", disease=top["label"]) if any(t.name == "search_products" for t in self.tools) else {}
            for p in (prods.get("products") or [])[:2]:
                lines.append(f"- Buy: [{p['name']}]({p['url']}) - ₹{p['price']} / {p['unit']}")
                tools_used.append("search_products")
            return {"reply": "\n\n".join(lines), "tools_used": tools_used, "sources": ["Disease model", "Disease guide"]}

        if not low or any(low.startswith(g) for g in GREETINGS) and len(low.split()) <= 3:
            return {"reply": L["greet"], "tools_used": [], "sources": []}
        if any(w in low for w in THANKS) and len(low.split()) <= 4:
            return {"reply": L["bye"], "tools_used": [], "sources": []}

        # ---- weather
        if any(w in low for w in ("weather", "forecast", "mausam", "barish", "rain today", "temperature today", "मौसम")):
            place = self._find_place(text) or self._profile_place(farmer_context)
            if not place:
                return {"reply": "Which town or district? e.g. 'weather in Indore'.", "tools_used": [], "sources": []}
            fc = self._t("get_weather", place=place)
            tools_used.append("get_weather")
            if "error" in fc:
                return {"reply": fc["error"], "tools_used": tools_used, "sources": []}
            cur = fc["current"]
            lines = [f"**Weather for {fc['place']['name']}, {fc['place'].get('admin1', '')}**",
                     f"Now: {cur['temperature']}°C, {cur['condition']}, humidity {cur['humidity']}%.", "",
                     "| Date | Condition | Min/Max °C | Rain mm |", "|---|---|---|---|"]
            for d in fc["days"][:5]:
                lines.append(f"| {d['date']} | {d['condition']} | {d['t_min']}/{d['t_max']} | {d['rain_mm']} |")
            lines.append("")
            for a in fc["advisories"][:3]:
                lines.append(f"- **{a['title']}:** {a['text']}")
            return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["Open-Meteo"]}

        # ---- mandi prices
        if any(w in low for w in ("price", "rate", "mandi", "bhav", "msp", "भाव", "कीमत")):
            crop = self._find_crop(text)
            commodity = {"paddy": "Paddy(Dhan)(Common)", "rice": "Rice", "corn": "Maize", "gram": "Bengal Gram(Gram)(Whole)",
                         "arhar": "Arhar (Tur/Red Gram)(Whole)", "tur": "Arhar (Tur/Red Gram)(Whole)", "moong": "Green Gram (Moong)(Whole)",
                         "urad": "Black Gram (Urd Beans)(Whole)", "soybean": "Soyabean"}.get(crop or "", (crop or "").title())
            if not commodity:
                return {"reply": "Which commodity? e.g. 'mandi price of wheat in Madhya Pradesh'.", "tools_used": [],
                        "sources": []}
            state = self._find_place(text)
            res = self._t("get_mandi_prices", commodity=commodity, state=state)
            tools_used.append("get_mandi_prices")
            lines = [f"**{commodity} prices** ({res['source']})"]
            if res.get("records"):
                lines += ["", "| Market | District | Modal ₹/qtl | Date |", "|---|---|---|---|"]
                for r in res["records"][:8]:
                    lines.append(f"| {r['market']} | {r['district']}, {r['state']} | {r['modal_price']} | {r['date']} |")
            for m in (res.get("msp") or [])[:4]:
                lines.append(f"- MSP {m['commodity']}: ₹{m['msp']}/quintal ({m.get('season', '')})")
            return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": [res["source"]]}

        # ---- fertilizer calculator
        area = self._find_area(text)
        crop = self._find_crop(text)
        if area and crop and any(w in low for w in ("fertil", "urea", "dap", "khad", "khaad", "dose", "kitna", "how much", "bag")):
            res = self._t("fertilizer_calculator", crop=crop, area=area[0], unit=area[1])
            tools_used.append("fertilizer_calculator")
            if "error" in res:
                return {"reply": res["error"], "tools_used": tools_used, "sources": []}
            b = res["bags_50kg"]
            kg = res["fertilizers_kg"]
            lines = [f"**Fertilizer for {res['crop']} on {area[0]:g} {area[1]}** (≈ {res['area_ha']} ha)",
                     f"- Urea: {kg['Urea']} kg (~{b['Urea']} bags)", f"- DAP: {kg['DAP']} kg (~{b['DAP']} bags)",
                     f"- MOP: {kg['MOP']} kg (~{b['MOP']} bags)", "", "**Schedule**"] + [f"- {s}" for s in res["schedule"]]
            lines.append("")
            lines.append(res["note"])
            return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["Fertilizer schedule"]}

        # ---- crop recommendation from numbers
        nums = self._find_numbers(text)
        need = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}
        if any(w in low for w in ("recommend", "which crop", "what crop", "suggest crop", "kaunsi fasal", "konsi fasal")) or need <= set(nums):
            if need <= set(nums):
                res = self._t("recommend_crop", **{k: nums[k] for k in need})
                tools_used.append("recommend_crop")
                alts = ", ".join(f"{a['crop']} ({a['probability'] * 100:.0f}%)" for a in res["alternatives"])
                reply = (f"**Recommended crop: {res['recommended_crop'].title()}** ({res['confidence'] * 100:.0f}% confidence)\n"
                         f"Alternatives: {alts}\n")
                if res.get("revenue_per_acre"):
                    reply += f"Indicative revenue ₹{res['revenue_per_acre']:,}/acre, cost ₹{res['cost_per_acre']:,}/acre, profit ≈ ₹{res['profit_per_acre']:,}/acre."
                return {"reply": reply, "tools_used": tools_used, "sources": ["Crop recommendation model"]}
            if "crop" in low or "fasal" in low:
                return {"reply": L["need_npk"], "tools_used": [], "sources": []}

        # ---- schemes
        if any(w in low for w in ("scheme", "yojana", "subsidy", "pm-kisan", "pm kisan", "kisan credit", "insurance", "bima", "योजना")):
            res = self._t("government_schemes", query=None if len(low.split()) < 3 else text)
            tools_used.append("government_schemes")
            schemes = res.get("schemes", [])[:4]
            lines = ["**Government schemes for farmers**"]
            for s in schemes:
                lines.append(f"- **{s.get('name', s.get('title'))}**: {s.get('summary', s.get('text', ''))[:220]} "
                             f"{('Apply: ' + s['how_to_apply']) if s.get('how_to_apply') else ''}")
            return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["Government scheme directory"]}

        # ---- calendar
        if any(w in low for w in ("when to sow", "sowing time", "kab boye", "kab bona", "calendar", "harvest time", "season for")):
            res = self._t("crop_calendar", crop=crop)
            tools_used.append("crop_calendar")
            rows = res.get("rows", [])[:6]
            if rows:
                lines = ["| Crop | Season | Sowing | Harvest | Regions |", "|---|---|---|---|---|"]
                lines += [f"| {r['crop']} | {r['season']} | {r['sowing']} | {r['harvest']} | {r.get('regions', '')} |" for r in rows]
                return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["Crop calendar"]}

        # ---- crop guide
        if crop and any(w in low for w in ("how to grow", "cultivat", "kheti", "guide", "variety", "varieties", "seed rate", "spacing", "irrigat")):
            g = self._t("crop_guide", crop={"paddy": "rice", "corn": "maize", "gram": "chickpea", "arhar": "pigeonpeas",
                                             "tur": "pigeonpeas", "moong": "mungbean", "urad": "blackgram", "masur": "lentil",
                                             "rajma": "kidneybeans"}.get(crop, crop))
            tools_used.append("crop_guide")
            if "error" not in g:
                keys = ["season", "climate", "soil", "sowing_time", "seed_rate", "spacing", "irrigation", "fertilizer_schedule",
                        "major_pests_diseases", "harvest", "expected_yield", "improved_varieties"]
                lines = [f"**{crop.title()} cultivation guide**"] + [f"- **{k.replace('_', ' ').title()}:** {g[k]}" for k in keys if g.get(k)]
                return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["KrishiDisha crop guide"]}

        # ---- products
        if any(w in low for w in ("buy", "purchase", "product", "kharid", "marketplace", "shop", "order")) and any(t.name == "search_products" for t in self.tools):
            res = self._t("search_products", query=text, crop=crop)
            tools_used.append("search_products")
            prods = res.get("products", [])[:5]
            if prods:
                lines = ["**Products you can buy in the KrishiDisha marketplace**"]
                lines += [f"- [{p['name']}]({p['url']}) - ₹{p['price']} / {p['unit']} ({p['category']})" for p in prods]
                return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": ["Marketplace"]}

        # ---- knowledge search (diseases, pests, general)
        hits = self.kb.search(text, k=3)
        if hits:
            top = hits[0]
            lines = [f"**{top['title']}** ({top['source']})", top["text"][:900]]
            if len(hits) > 1:
                lines.append("")
                lines.append("Related: " + "; ".join(h["title"] for h in hits[1:]))
            if top.get("kind") == "disease" and any(t.name == "search_products" for t in self.tools):
                prods = self._t("search_products", query=top["title"]).get("products", [])[:2]
                for p in prods:
                    lines.append(f"- Buy: [{p['name']}]({p['url']}) - ₹{p['price']}")
            return {"reply": "\n".join(lines), "tools_used": tools_used, "sources": [h["title"] for h in hits]}

        return {"reply": L["unknown"], "tools_used": [], "sources": []}

    @staticmethod
    def _profile_place(farmer_context: str | None) -> str | None:
        if not farmer_context:
            return None
        m = re.search(r"district:\s*([^,]+)", farmer_context) or re.search(r"state:\s*([^,]+)", farmer_context)
        return m.group(1).strip() if m else None
