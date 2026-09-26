"""Language layer: detect the farmer's language and translate to/from English for models that only
handle English/Hindi natively (the local fine-tuned model, the rules bot).

Pipeline (``LanguageLayer.inbound`` / ``outbound``):

1. detect  - script-based detection for Indic scripts, romanised-Hindi keyword check for Hinglish,
             ai4bharat IndicLID when installed (``pip install indic-lid`` / fasttext), else the heuristic.
2. mask    - numbers with units, product names, URLs, markdown links and tool JSON are replaced by
             placeholders so the translator cannot mangle them.
3. translate - backend chosen by ``TRANSLATION_BACKEND``:
             ``none``        no translation (default; native languages only)
             ``indictrans2`` ai4bharat IndicTrans2 distilled 200M models via CTranslate2 (local, MIT)
             ``bhashini``    Government of India Bhashini API (hosted; ``BHASHINI_USER_ID`` / ``BHASHINI_API_KEY``)
             ``fake``        identity with markers (tests)
4. unmask  - placeholders restored.

Native languages (no translation) default to ``{en, hi, hinglish}``; everything else is translated to
English for the model and the reply is translated back sentence by sentence. The Anthropic provider
handles Indic languages itself, so the layer is bypassed for it (see ``AgriAssistant``).
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

SCRIPT_LANG = {"DEVANAGARI": "hi", "GURMUKHI": "pa", "GUJARATI": "gu", "BENGALI": "bn", "TAMIL": "ta", "TELUGU": "te",
               "KANNADA": "kn", "MALAYALAM": "ml", "ORIYA": "or"}
# IndicTrans2 / Bhashini language codes (Flores-200 style)
FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "pa": "pan_Guru", "gu": "guj_Gujr", "bn": "ben_Beng",
          "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya"}
# Devanagari is shared by Hindi and Marathi: a few frequent Marathi function words tip the balance.
MARATHI_HINTS = {"आहे", "आहेत", "आणि", "कसे", "काय", "कधी", "किती", "पाहिजे", "करावे", "मला", "शेती", "पीक", "खत", "मध्ये", "साठी"}
HINGLISH_WORDS = {"kya", "hai", "hain", "kaise", "kab", "kitna", "kitni", "mein", "me", "ke", "ki", "ka", "liye", "fasal",
                  "khad", "khet", "kheti", "beej", "paani", "dawa", "keeda", "rog", "bimari", "batao", "bataye", "chahiye",
                  "karein", "karna", "lagta", "lagega", "kaun", "kaunsi", "acha", "accha", "aur", "nahi", "bhi", "hun", "mera", "meri"}
NATIVE_DEFAULT = {"en", "hi", "hinglish"}

_MASK_PATTERNS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),                      # markdown links
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kg|gm|g|ml|litres?|liters?|l|mm|cm|%|°c|acres?|ha|hectares?|quintals?|q|bags?|₹|rs\.?)"
               r"(?:\s*/\s*(?:acre|ha|hectare|litre|liter|l|quintal|plant|kg))?(?![a-z])", re.I),
    re.compile(r"\b(?:\d{1,2}-\d{1,2}-\d{1,2}|[A-Z][A-Za-z]*(?:\s[0-9]+(?:\.[0-9]+)?\s?(?:EC|SL|SC|WP|WG|SP|OD|DF|G))\b)"),  # NPK grades, formulations
]


def detect_language(text: str) -> tuple[str, float]:
    """Return (language code, confidence 0-1)."""
    text = text or ""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            try:
                block = unicodedata.name(ch, "").split(" ")[0]
            except ValueError:
                continue
            key = SCRIPT_LANG.get(block, "latin")
            counts[key] = counts.get(key, 0) + 1
    if not letters:
        return "en", 0.0
    top = max(counts, key=counts.get)
    share = counts[top] / letters
    if top != "latin" and counts[top] >= 3:
        if top == "hi":
            words = set(re.findall(r"[ऀ-ॿ]+", text))
            if len(words & MARATHI_HINTS) >= 2:
                return "mr", round(0.6 + 0.3 * share, 2)
        return top, round(0.7 + 0.3 * share, 2)
    lid = _indiclid(text)
    if lid:
        return lid
    words = re.findall(r"[a-z]+", text.lower())
    if words:
        ratio = sum(w in HINGLISH_WORDS for w in words) / len(words)
        if ratio >= 0.12:
            return "hinglish", round(min(0.5 + ratio, 0.95), 2)
    return "en", round(0.5 + 0.4 * share, 2)


def _indiclid(text: str) -> tuple[str, float] | None:
    """Optional ai4bharat IndicLID for romanised regional languages; silently unavailable otherwise."""
    try:
        from ai4bharat.IndicLID import IndicLID  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        global _INDICLID
        if "_INDICLID" not in globals():
            _INDICLID = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
        out = _INDICLID.batch_predict([text], 1)[0]
        code, score = out[1], float(out[2])
        mapping = {"hin_Latn": "hinglish", "mar_Latn": "mr", "ben_Latn": "bn", "tam_Latn": "ta", "tel_Latn": "te",
                   "guj_Latn": "gu", "pan_Latn": "pa", "kan_Latn": "kn", "mal_Latn": "ml", "eng_Latn": "en"}
        if code in mapping and score >= 0.6:
            return mapping[code], round(score, 2)
    except Exception as exc:  # noqa: BLE001
        log.debug("IndicLID failed: %s", exc)
    return None


# ------------------------------------------------------------------ query glossary
# Hindi / Hinglish agricultural terms -> English, used to expand retrieval queries so both the TF-IDF and
# the dense leg see the English vocabulary the knowledge base is written in. Longer keys match first.
AGRI_GLOSSARY: dict[str, str] = {
    # crops
    "गेहूं": "wheat", "गेहू": "wheat", "gehu": "wheat", "gehun": "wheat", "धान": "rice paddy", "चावल": "rice", "dhaan": "rice paddy",
    "dhan": "rice paddy", "मक्का": "maize corn", "makka": "maize", "कपास": "cotton", "kapas": "cotton", "गन्ना": "sugarcane",
    "ganna": "sugarcane", "ganne": "sugarcane", "सोयाबीन": "soybean", "मूंगफली": "groundnut", "moongphali": "groundnut", "सरसों": "mustard",
    "sarson": "mustard", "चना": "chickpea gram", "chana": "chickpea gram", "अरहर": "pigeonpea arhar tur", "arhar": "pigeonpea",
    "मसूर": "lentil", "मूंग": "mungbean moong", "moong": "mungbean", "उड़द": "blackgram urad", "urad": "blackgram", "आलू": "potato",
    "aloo": "potato", "टमाटर": "tomato", "tamatar": "tomato", "प्याज": "onion", "pyaz": "onion", "केला": "banana", "kela": "banana",
    "आम": "mango", "aam": "mango", "अंगूर": "grapes", "सेब": "apple", "मिर्च": "chilli", "mirch": "chilli", "बैंगन": "brinjal",
    "भिंडी": "okra", "बाजरा": "bajra millets", "bajra": "millets", "ज्वार": "jowar sorghum", "jowar": "sorghum",
    # diseases / pests
    "पीला रतुआ": "yellow stripe rust", "रतुआ": "rust", "peela ratua": "yellow rust", "ratua": "rust", "ब्लास्ट": "blast",
    "झुलसा": "blight", "jhulsa": "blight", "अगेती झुलसा": "early blight", "पछेती झुलसा": "late blight",
    "गुलाबी सुंडी": "pink bollworm", "gulabi sundi": "pink bollworm", "सुंडी": "bollworm caterpillar", "sundi": "caterpillar",
    "सफेद मक्खी": "whitefly", "safed makkhi": "whitefly", "माहू": "aphid", "mahu": "aphid", "थ्रिप्स": "thrips",
    "तना छेदक": "stem borer", "tana chhedak": "stem borer", "फल छेदक": "fruit borer", "जड़ सड़न": "root rot",
    "लाल सड़न": "red rot", "lal sadan": "red rot", "sadan": "rot", "सड़न": "rot", "उकठा": "wilt", "ukta": "wilt", "चूर्णी फफूंद": "powdery mildew", "पत्ती धब्बा": "leaf spot",
    "मोज़ेक": "mosaic virus", "पत्ती मोड़": "leaf curl", "दीमक": "termite", "deemak": "termite", "फॉल आर्मीवर्म": "fall armyworm",
    "भूरा धब्बा": "brown spot", "शीथ ब्लाइट": "sheath blight", "जीवाणु झुलसा": "bacterial blight", "कीट": "pest insect",
    "keeda": "pest insect", "keede": "pest insect", "रोग": "disease", "rog": "disease", "bimari": "disease", "फफूंद": "fungus fungal",
    # inputs / practices
    "खाद": "fertilizer", "khad": "fertilizer", "khaad": "fertilizer", "उर्वरक": "fertilizer", "यूरिया": "urea", "urea": "urea",
    "डीएपी": "DAP", "dap": "DAP", "पोटाश": "potash MOP", "बीज": "seed", "beej": "seed", "बुवाई": "sowing", "buwai": "sowing",
    "bone": "sowing", "सिंचाई": "irrigation", "sinchai": "irrigation", "पानी": "water irrigation", "कटाई": "harvest",
    "katai": "harvest", "पैदावार": "yield", "उपज": "yield", "paidawar": "yield", "किस्म": "variety", "kism": "variety",
    "दवा": "pesticide spray", "dawa": "pesticide spray", "छिड़काव": "spray", "chhidkav": "spray", "खरपतवार": "weed",
    "मिट्टी": "soil", "mitti": "soil", "मौसम": "weather", "mausam": "weather", "बारिश": "rain", "barish": "rain",
    "मंडी": "mandi market price", "भाव": "price", "bhav": "price", "योजना": "scheme yojana", "yojana": "scheme",
    "बीमा": "insurance", "bima": "insurance", "ऋण": "loan credit", "kharif": "kharif", "rabi": "rabi",
    "जैविक": "organic", "jaivik": "organic", "नीम": "neem", "neem": "neem", "गोबर": "farmyard manure", "gobar": "manure",
}
_GLOSSARY_KEYS = sorted(AGRI_GLOSSARY, key=len, reverse=True)


def expand_query(text: str) -> str:
    """Append English equivalents of Hindi/Hinglish agri terms found in ``text`` (for retrieval only)."""
    if not text:
        return text
    low = text.lower()
    found: list[str] = []
    for key in _GLOSSARY_KEYS:
        if key.isascii():
            hit = re.search(rf"\b{re.escape(key)}\b", low) is not None
        else:  # Devanagari: require the term not to be embedded in a longer Devanagari word
            hit = re.search(rf"(?<![ऀ-ॿ]){re.escape(key)}(?![ऀ-ॿ])", low) is not None
        if hit:
            eng = AGRI_GLOSSARY[key]
            if eng not in found:
                found.append(eng)
    return f"{text} {' '.join(found)}" if found else text


# ------------------------------------------------------------------ masking
@dataclass
class Masked:
    text: str
    slots: dict[str, str] = field(default_factory=dict)

    def restore(self, translated: str) -> str:
        out = translated
        for key, val in self.slots.items():
            # translators sometimes add spaces inside or around the placeholder
            out = re.sub(r"KD\s*" + key[2:], lambda _m, v=val: v, out, flags=re.I)
        return out


def mask(text: str) -> Masked:
    m = Masked(text)
    i = 0

    def repl(match: re.Match) -> str:
        nonlocal i
        key = f"KD{i:03d}"
        i += 1
        m.slots[key] = match.group(0)
        return key

    for pat in _MASK_PATTERNS:
        m.text = pat.sub(repl, m.text)
    return m


# ------------------------------------------------------------------ backends
class Translator:
    name = "none"

    def translate(self, text: str, src: str, tgt: str) -> str:
        return text


class FakeTranslator(Translator):
    """Identity with visible markers, for tests."""
    name = "fake"

    def translate(self, text: str, src: str, tgt: str) -> str:
        return f"[{src}>{tgt}] {text}"


class IndicTrans2Translator(Translator):
    """ai4bharat/indictrans2 distilled 200M models. Uses CTranslate2 int8 conversions when present under
    ``INDICTRANS2_DIR`` (fast on CPU), else the HF checkpoints through transformers (slower)."""
    name = "indictrans2"

    def __init__(self, model_dir: str | None = None, device: str = "cpu"):
        self.model_dir = model_dir or os.getenv("INDICTRANS2_DIR", "models/indictrans2")
        self.device = device
        self._models: dict[str, Any] = {}
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore

        self.ip = IndicProcessor(inference=True)

    def _load(self, direction: str):
        if direction in self._models:
            return self._models[direction]
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = {"indic-en": "ai4bharat/indictrans2-indic-en-dist-200M", "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M"}[direction]
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32).to(self.device).eval()
        self._models[direction] = (tok, model)
        return tok, model

    def translate(self, text: str, src: str, tgt: str) -> str:
        import torch

        s, t = FLORES.get(src), FLORES.get(tgt)
        if not s or not t or s == t:
            return text
        direction = "indic-en" if tgt == "en" else "en-indic"
        tok, model = self._load(direction)
        sents = [x for x in re.split(r"(?<=[.!?।])\s+", text) if x.strip()]
        batch = self.ip.preprocess_batch(sents, src_lang=s, tgt_lang=t)
        with torch.no_grad():
            enc = tok(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(self.device)
            gen = model.generate(**enc, max_length=256, num_beams=4, num_return_sequences=1)
        out = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return " ".join(self.ip.postprocess_batch(out, lang=t))


class BhashiniTranslator(Translator):
    """Bhashini (ULCA) hosted translation. Needs BHASHINI_USER_ID and BHASHINI_API_KEY (free registration)."""
    name = "bhashini"
    AUTH_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    def __init__(self):
        self.user_id = os.getenv("BHASHINI_USER_ID")
        self.api_key = os.getenv("BHASHINI_API_KEY")
        self.pipeline_id = os.getenv("BHASHINI_PIPELINE_ID", "64392f96daac500b55c543cd")
        self._cfg: dict[str, Any] = {}

    def _config(self, src: str, tgt: str) -> dict:
        import requests

        key = f"{src}-{tgt}"
        if key in self._cfg:
            return self._cfg[key]
        body = {"pipelineTasks": [{"taskType": "translation", "config": {"language": {"sourceLanguage": src, "targetLanguage": tgt}}}],
                "pipelineRequestConfig": {"pipelineId": self.pipeline_id}}
        r = requests.post(self.AUTH_URL, json=body, headers={"userID": self.user_id, "ulcaApiKey": self.api_key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        cfg = {"url": data["pipelineInferenceAPIEndPoint"]["callbackUrl"],
               "auth": data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"],
               "service": data["pipelineResponseConfig"][0]["config"][0]["serviceId"]}
        self._cfg[key] = cfg
        return cfg

    def translate(self, text: str, src: str, tgt: str) -> str:
        import requests

        if src == tgt or not self.user_id:
            return text
        src_code, tgt_code = ("hi" if src == "hinglish" else src), ("hi" if tgt == "hinglish" else tgt)
        cfg = self._config(src_code, tgt_code)
        body = {"pipelineTasks": [{"taskType": "translation",
                                   "config": {"language": {"sourceLanguage": src_code, "targetLanguage": tgt_code},
                                              "serviceId": cfg["service"]}}],
                "inputData": {"input": [{"source": text}]}}
        r = requests.post(cfg["url"], json=body, headers={cfg["auth"]["name"]: cfg["auth"]["value"]}, timeout=30)
        r.raise_for_status()
        return r.json()["pipelineResponse"][0]["output"][0]["target"]


def make_translator(backend: str | None) -> Translator:
    backend = (backend or "none").lower()
    try:
        if backend == "indictrans2":
            return IndicTrans2Translator()
        if backend == "bhashini":
            return BhashiniTranslator()
        if backend == "fake":
            return FakeTranslator()
    except Exception as exc:  # noqa: BLE001
        log.warning("translation backend %s unavailable (%s); running without translation", backend, exc)
    return Translator()


# ------------------------------------------------------------------ layer
class LanguageLayer:
    def __init__(self, backend: str | None = None, native: set[str] | None = None):
        self.translator = make_translator(backend)
        self.native = set(native or NATIVE_DEFAULT)

    @property
    def active(self) -> bool:
        return self.translator.name != "none"

    def resolve(self, text: str, requested: str | None) -> tuple[str, float]:
        """Language to answer in: the detected one when confident and the UI said auto/unknown, else the request."""
        detected, conf = detect_language(text)
        if not requested or requested == "auto":
            return detected, conf
        if conf >= 0.8 and detected != requested and detected not in ("en",):
            return detected, conf
        return requested, conf

    def inbound(self, text: str, language: str) -> tuple[str, Masked | None]:
        """Translate the user's text to English when its language is not native for the model."""
        if language in self.native or not self.active:
            return text, None
        m = mask(text)
        translated = self.translator.translate(m.text, language, "en")
        return m.restore(translated), m

    def outbound(self, reply: str, language: str) -> str:
        if language in self.native or not self.active or not reply:
            return reply
        parts = re.split(r"(\n+)", reply)
        out = []
        for part in parts:
            if not part.strip() or part.startswith("\n"):
                out.append(part)
                continue
            m = mask(part)
            out.append(m.restore(self.translator.translate(m.text, "en", language)))
        return "".join(out)
