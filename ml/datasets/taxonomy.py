"""Canonical ``Crop___Condition`` taxonomy shared by training, the manifest and the app.

Every image source names its classes differently ("Bacterialblight", "bacterial_leaf_blight",
"Tomato leaf late blight", "Corn_(maize)___Common_rust_"). This module maps each source's raw
folder / CSV label to one canonical label so that the same disease from two datasets becomes one
class. The canonical form is the PlantVillage-style ``Crop___Condition`` string that
:func:`krishidisha.services.disease.split_label` and ``KnowledgeBase.disease_by_label`` already
understand.

Two mechanisms:

* explicit per-source dictionaries in :data:`SOURCE_MAPS` (exact, reviewed);
* a generic parser (:func:`parse_generic`) for sources whose folder names are self-explanatory.

Anything the parser cannot place returns ``None``; the manifest builder collects these into
``unmapped.json`` so they can be added here instead of silently becoming a wrong class.
"""
from __future__ import annotations

import re

TAXONOMY_VERSION = "2026.09"

NOT_A_LEAF = "Other___not_a_leaf"

# Canonical crop names (Title_Case, underscores). Aliases are lower-case tokens seen in folder names.
CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "Rice": ("rice", "paddy", "dhan"),
    "Wheat": ("wheat", "gehu"),
    "Maize": ("maize", "corn"),
    "Cotton": ("cotton", "kapas"),
    "Sugarcane": ("sugarcane", "sugar_cane", "cane"),
    "Soybean": ("soybean", "soyabean", "soya"),
    "Groundnut": ("groundnut", "peanut", "moongphali"),
    "Chilli": ("chilli", "chili", "chile", "mirchi", "capsicum_annuum"),
    "Tomato": ("tomato",),
    "Potato": ("potato",),
    "Onion": ("onion",),
    "Brinjal": ("brinjal", "eggplant", "aubergine"),
    "Okra": ("okra", "bhindi", "ladyfinger"),
    "Mango": ("mango",),
    "Banana": ("banana",),
    "Citrus": ("citrus", "orange", "lemon", "lime", "mosambi"),
    "Grape": ("grape", "grapes"),
    "Apple": ("apple",),
    "Pomegranate": ("pomegranate", "anar"),
    "Papaya": ("papaya",),
    "Guava": ("guava",),
    "Coconut": ("coconut",),
    "Cashew": ("cashew",),
    "Cassava": ("cassava",),
    "Bell_pepper": ("bell_pepper", "pepper_bell", "bellpepper", "pepper"),
    "Peach": ("peach",),
    "Cherry": ("cherry",),
    "Strawberry": ("strawberry",),
    "Blueberry": ("blueberry",),
    "Raspberry": ("raspberry",),
    "Squash": ("squash",),
    "Coffee": ("coffee",),
    "Tea": ("tea",),
    "Jute": ("jute",),
    "Mustard": ("mustard", "sarson"),
    "Chickpea": ("chickpea", "gram", "chana"),
    "Pigeonpea": ("pigeonpea", "pigeon_pea", "arhar", "tur"),
}
_ALIAS_TO_CROP = {a: c for c, aliases in CROP_ALIASES.items() for a in aliases}

HEALTHY_WORDS = {"healthy", "normal", "fresh", "good", "no_disease", "nodisease", "health"}
NOISE_WORDS = {"leaf", "leaves", "disease", "diseases", "diseased", "plant", "plants", "image", "images",
               "dataset", "of", "the", "and", "with", "infected", "affected", "class", "photo", "photos"}

# Condition synonyms -> canonical condition (Title_Case words).
CONDITION_SYNONYMS: dict[str, str] = {
    "bacterialblight": "Bacterial_blight", "bacterial_blight": "Bacterial_blight",
    "bacterial_leaf_blight": "Bacterial_leaf_blight", "blb": "Bacterial_leaf_blight",
    "bacterial_leaf_streak": "Bacterial_leaf_streak", "bacterial_panicle_blight": "Bacterial_panicle_blight",
    "brownspot": "Brown_spot", "brown_spot": "Brown_spot",
    "leaf_blast": "Blast", "blast": "Blast", "neck_blast": "Blast",
    "sheath_blight": "Sheath_blight", "leaf_scald": "Leaf_scald", "narrow_brown_spot": "Narrow_brown_spot",
    "dead_heart": "Dead_heart", "hispa": "Hispa", "tungro": "Tungro", "downy_mildew": "Downy_mildew",
    "redrot": "Red_rot", "red_rot": "Red_rot", "rust": "Rust", "mosaic": "Mosaic", "yellow": "Yellow_leaf",
    "yellow_leaf": "Yellow_leaf", "leaf_rust": "Leaf_rust", "stem_rust": "Stem_rust", "stripe_rust": "Stripe_rust",
    "yellow_rust": "Stripe_rust", "brown_rust": "Leaf_rust", "black_rust": "Stem_rust", "septoria": "Septoria",
    "powdery_mildew": "Powdery_mildew", "anthracnose": "Anthracnose", "bacterial_canker": "Bacterial_canker",
    "cutting_weevil": "Cutting_weevil", "die_back": "Die_back", "dieback": "Die_back", "gall_midge": "Gall_midge",
    "sooty_mould": "Sooty_mould", "sooty_mold": "Sooty_mould",
    "early_blight": "Early_blight", "late_blight": "Late_blight", "leaf_blight": "Leaf_blight",
    "northern_leaf_blight": "Northern_leaf_blight", "gray_leaf_spot": "Gray_leaf_spot",
    "grey_leaf_spot": "Gray_leaf_spot", "common_rust": "Common_rust", "fall_armyworm": "Fall_armyworm",
    "leaf_curl": "Leaf_curl", "leaf_curl_virus": "Leaf_curl_virus", "curl_virus": "Leaf_curl_virus",
    "yellow_leaf_curl_virus": "Yellow_leaf_curl_virus", "yellow_virus": "Yellow_leaf_curl_virus",
    "mosaic_virus": "Mosaic_virus", "leaf_mold": "Leaf_mold", "mold": "Leaf_mold", "leaf_mould": "Leaf_mold",
    "septoria_leaf_spot": "Septoria_leaf_spot", "bacterial_spot": "Bacterial_spot", "leaf_spot": "Leaf_spot",
    "target_spot": "Target_spot", "spider_mites": "Spider_mites", "two_spotted_spider_mites": "Spider_mites",
    "aphids": "Aphids", "aphid": "Aphids", "army_worm": "Army_worm", "armyworm": "Army_worm",
    "whitefly": "Whitefly", "white_fly": "Whitefly", "thrips": "Thrips", "jassid": "Jassid", "mites": "Spider_mites",
    "fusarium_wilt": "Fusarium_wilt", "wilt": "Wilt", "verticillium_wilt": "Verticillium_wilt",
    "cercospora": "Cercospora_leaf_spot", "cercospora_leaf_spot": "Cercospora_leaf_spot",
    "alternaria": "Alternaria_leaf_spot", "alternaria_leaf_spot": "Alternaria_leaf_spot",
    "black_rot": "Black_rot", "black_spot": "Black_spot", "scab": "Scab", "apple_scab": "Scab",
    "cedar_apple_rust": "Cedar_apple_rust", "apple_rust": "Cedar_apple_rust",
    "esca": "Esca", "black_measles": "Esca", "isariopsis_leaf_spot": "Leaf_blight",
    "citrus_greening": "Citrus_greening", "haunglongbing": "Citrus_greening", "huanglongbing": "Citrus_greening",
    "greening": "Citrus_greening", "canker": "Canker", "melanose": "Melanose",
    "sigatoka": "Sigatoka", "black_sigatoka": "Sigatoka", "cordana": "Cordana", "pestalotiopsis": "Pestalotiopsis",
    "panama": "Fusarium_wilt", "bunchy_top": "Bunchy_top",
    "leaf_miner": "Leaf_miner", "leaf_scorch": "Leaf_scorch", "tikka": "Tikka_leaf_spot",
    "early_leaf_spot": "Early_leaf_spot", "late_leaf_spot": "Late_leaf_spot", "nutrient_deficiency": "Nutrient_deficiency",
    "cbb": "Bacterial_blight", "cmd": "Mosaic_virus", "cbsd": "Brown_streak_disease", "cgm": "Green_mite",
    "brown_streak": "Brown_streak_disease", "green_mite": "Green_mite", "leaf_beetle": "Leaf_beetle",
    "gumosis": "Gummosis", "gummosis": "Gummosis", "red_rust": "Red_rust", "streak_virus": "Streak_virus",
    "leaf_curl_disease": "Leaf_curl", "leaf_spot_disease": "Leaf_spot", "whitefly_damage": "Whitefly",
    "yellowish": "Yellowing", "yellowing": "Yellowing", "leaf_yellowing": "Yellowing",
    "bacterial_leaf_spot": "Bacterial_spot", "frogeye": "Frogeye_leaf_spot", "frogeye_leaf_spot": "Frogeye_leaf_spot",
    "rust_disease": "Rust", "caterpillar": "Caterpillar", "mildew": "Powdery_mildew", "virus": "Viral_disease",
}


def _slug(text: str) -> str:
    text = text.strip().lower().replace("&", " and ")
    text = re.sub(r"[\(\)\[\],;:'\"/\\\.-]+", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"_+", "_", text)


def _title(cond: str) -> str:
    cond = cond.strip("_")
    if not cond:
        return ""
    return cond[0].upper() + cond[1:]


def parse_generic(raw: str, default_crop: str | None = None) -> str | None:
    """Best-effort parse of a folder name such as ``"Tomato leaf late blight"``.

    Returns a canonical label or ``None`` when neither crop nor condition can be identified.
    """
    slug = _slug(raw)
    tokens = slug.split("_")

    crop = None
    non_crop: list[str] = []          # every token except the crop name (keeps "leaf", "spot", ...)
    i = 0
    while i < len(tokens):
        # two-token aliases (bell_pepper, sugar_cane, pigeon_pea)
        pair = "_".join(tokens[i:i + 2])
        if crop is None and pair in _ALIAS_TO_CROP:
            crop = _ALIAS_TO_CROP[pair]
            i += 2
            continue
        tok = tokens[i]
        if crop is None and tok in _ALIAS_TO_CROP:
            crop = _ALIAS_TO_CROP[tok]
        elif tok:
            non_crop.append(tok)
        i += 1
    crop = crop or default_crop
    if crop is None:
        return None

    remaining = [t for t in non_crop if t not in NOISE_WORDS]
    if not remaining or any(t in HEALTHY_WORDS for t in remaining):
        return f"{crop}___healthy"

    def longest_synonym(words: list[str]) -> str | None:
        for n in range(len(words), 0, -1):
            for start in range(0, len(words) - n + 1):
                key = "_".join(words[start:start + n])
                if key in CONDITION_SYNONYMS:
                    return CONDITION_SYNONYMS[key]
        return None

    # "leaf blight" must stay Leaf_blight, so match with the noise words present first
    cond = longest_synonym(non_crop) or longest_synonym(remaining) or _title("_".join(remaining))
    return f"{crop}___{cond}"


# --------------------------------------------------------------------------- explicit maps
# Keys are lower-case slugs of the raw label (folder name or CSV value).
SOURCE_MAPS: dict[str, dict[str, str]] = {
    "paddy_doctor": {
        "bacterial_leaf_blight": "Rice___Bacterial_leaf_blight",
        "bacterial_leaf_streak": "Rice___Bacterial_leaf_streak",
        "bacterial_panicle_blight": "Rice___Bacterial_panicle_blight",
        "blast": "Rice___Blast",
        "brown_spot": "Rice___Brown_spot",
        "dead_heart": "Rice___Dead_heart",
        "downy_mildew": "Rice___Downy_mildew",
        "hispa": "Rice___Hispa",
        "normal": "Rice___healthy",
        "tungro": "Rice___Tungro",
    },
    "rice_leaf_4": {
        "bacterialblight": "Rice___Bacterial_leaf_blight",
        "bacterial_blight": "Rice___Bacterial_leaf_blight",
        "blast": "Rice___Blast",
        "brownspot": "Rice___Brown_spot",
        "brown_spot": "Rice___Brown_spot",
        "tungro": "Rice___Tungro",
        "healthy": "Rice___healthy",
    },
    "sugarcane_leaf": {
        "healthy": "Sugarcane___healthy",
        "mosaic": "Sugarcane___Mosaic",
        "redrot": "Sugarcane___Red_rot",
        "red_rot": "Sugarcane___Red_rot",
        "rust": "Sugarcane___Rust",
        "yellow": "Sugarcane___Yellow_leaf",
        "yellow_leaf": "Sugarcane___Yellow_leaf",
    },
    "mango_leaf": {
        "anthracnose": "Mango___Anthracnose",
        "bacterial_canker": "Mango___Bacterial_canker",
        "cutting_weevil": "Mango___Cutting_weevil",
        "die_back": "Mango___Die_back",
        "gall_midge": "Mango___Gall_midge",
        "healthy": "Mango___healthy",
        "powdery_mildew": "Mango___Powdery_mildew",
        "sooty_mould": "Mango___Sooty_mould",
    },
    "wheat_rust_cgiar": {
        "healthy_wheat": "Wheat___healthy",
        "leaf_rust": "Wheat___Leaf_rust",
        "stem_rust": "Wheat___Stem_rust",
    },
    "wheat_leaf": {
        "healthy": "Wheat___healthy",
        "septoria": "Wheat___Septoria",
        "stripe_rust": "Wheat___Stripe_rust",
    },
    "plantdoc": {
        "apple_scab_leaf": "Apple___Scab", "apple_leaf": "Apple___healthy", "apple_rust_leaf": "Apple___Cedar_apple_rust",
        "bell_pepper_leaf": "Bell_pepper___healthy", "bell_pepper_leaf_spot": "Bell_pepper___Bacterial_spot",
        "blueberry_leaf": "Blueberry___healthy", "cherry_leaf": "Cherry___healthy",
        "corn_gray_leaf_spot": "Maize___Gray_leaf_spot", "corn_leaf_blight": "Maize___Northern_leaf_blight",
        "corn_rust_leaf": "Maize___Common_rust", "peach_leaf": "Peach___healthy",
        "potato_leaf": "Potato___healthy", "potato_leaf_early_blight": "Potato___Early_blight",
        "potato_leaf_late_blight": "Potato___Late_blight", "raspberry_leaf": "Raspberry___healthy",
        "soyabean_leaf": "Soybean___healthy", "soybean_leaf": "Soybean___healthy",
        "squash_powdery_mildew_leaf": "Squash___Powdery_mildew", "strawberry_leaf": "Strawberry___healthy",
        "tomato_early_blight_leaf": "Tomato___Early_blight", "tomato_septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
        "tomato_leaf": "Tomato___healthy", "tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
        "tomato_leaf_late_blight": "Tomato___Late_blight", "tomato_leaf_mosaic_virus": "Tomato___Mosaic_virus",
        "tomato_leaf_yellow_virus": "Tomato___Yellow_leaf_curl_virus", "tomato_mold_leaf": "Tomato___Leaf_mold",
        "tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites",
        "grape_leaf": "Grape___healthy", "grape_leaf_black_rot": "Grape___Black_rot",
    },
    "plantvillage": {},          # already canonical after crop normalisation (see canonical_label)
    "canonical": {},             # own photos are stored under canonical folder names
    "not_leaf": {},              # every image -> NOT_A_LEAF
}

# Sources whose folder names are parsed generically, with the crop fixed by the source.
GENERIC_DEFAULT_CROP: dict[str, str | None] = {
    "cotton_leaf": "Cotton", "chilli_leaf": "Chilli", "soybean_leaf": "Soybean", "groundnut_leaf": "Groundnut",
    "banana_leaf": "Banana", "citrus_leaf": "Citrus", "ccmt": None,
}

_PV_CROP_FIX = {"corn_(maize)": "Maize", "pepper,_bell": "Bell_pepper", "cherry_(including_sour)": "Cherry",
                "orange": "Citrus", "squash": "Squash"}


def canonical_label(mapping: str, raw: str, parent: str | None = None) -> str | None:
    """Map a raw label from a source to the canonical taxonomy.

    ``parent`` is the parent folder name for nested layouts (e.g. CCMT ``Maize/leaf blight``).
    """
    if mapping == "not_leaf":
        return NOT_A_LEAF
    if mapping == "canonical":
        return raw if "___" in raw else None
    if mapping == "plantvillage":
        crop, _, cond = raw.partition("___")
        crop = _PV_CROP_FIX.get(crop.lower(), _title(_slug(crop)))
        cond = cond.strip("_")
        if cond.lower() == "healthy":
            return f"{crop}___healthy"
        return f"{crop}___{CONDITION_SYNONYMS.get(_slug(cond), _title(_slug(cond)))}"

    slug = _slug(raw)
    table = SOURCE_MAPS.get(mapping)
    if table and slug in table:
        return table[slug]
    default_crop = GENERIC_DEFAULT_CROP.get(mapping)
    if parent and default_crop is None:
        parsed_parent = parse_generic(parent)
        if parsed_parent:
            default_crop = parsed_parent.split("___", 1)[0]
    return parse_generic(raw, default_crop)


def split(label: str) -> tuple[str, str]:
    crop, _, cond = label.partition("___")
    return crop, cond


def is_healthy(label: str) -> bool:
    return label.endswith("___healthy")


def pretty(label: str) -> str:
    crop, cond = split(label)
    return f"{crop.replace('_', ' ')}: {cond.replace('_', ' ')}"
