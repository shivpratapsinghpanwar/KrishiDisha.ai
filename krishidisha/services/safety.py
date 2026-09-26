"""Post-check for pesticide / fertilizer advice, applied to every assistant reply (any provider).

Two guards:

* **Banned or restricted molecules** (CIB&RC, Government of India): if a reply recommends one, the
  sentence is flagged, a warning is appended and the molecule is listed so the UI can highlight it.
  The list covers the 2018 ban (18 pesticides), the 2020-2023 bans (e.g. dicofol, dinocap,
  methomyl, monocrotophos in vegetables), plus WHO class Ia/Ib molecules that Indian extension
  advice steers away from. It is a text guard, not a legal register - review yearly.
* **Dose sanity**: crude per-unit ceilings for common actives and fertilizers (g or ml per litre,
  kg per acre). Numbers above the ceiling get a "verify with KVK" note appended.

Also exposed as the ``safety_check`` tool so a model can self-check before answering.
"""
from __future__ import annotations

import re
from typing import Any

BANNED_MOLECULES: dict[str, str] = {
    # 2018 gazette ban (18 pesticides) and later Indian bans / phase-outs
    "benomyl": "banned in India (2018)", "carbaryl": "banned in India (2018)", "diazinon": "banned in India (2018)",
    "fenarimol": "banned in India (2018)", "fenthion": "banned in India (2018)", "linuron": "banned in India (2018)",
    "methoxy ethyl mercury chloride": "banned in India (2018)", "methyl parathion": "banned in India (2018)",
    "sodium cyanide": "banned in India (2018)", "thiometon": "banned in India (2018)", "tridemorph": "banned in India (2018)",
    "trifluralin": "banned in India (2018)", "alachlor": "banned in India (2020)", "dichlorvos": "banned in India (2020)",
    "phorate": "banned in India (2020)", "phosphamidon": "banned in India (2020)", "triazophos": "banned in India (2020)",
    "trichlorfon": "banned in India (2020)", "endosulfan": "banned in India (2011, Supreme Court)",
    "monocrotophos": "banned on vegetables in India; WHO class Ib, many farmer poisonings",
    "dicofol": "banned in India (2023)", "dinocap": "banned in India (2023)", "methomyl": "banned in India (2023)",
    "captafol": "banned in India", "aldicarb": "banned in India", "aldrin": "banned in India", "chlordane": "banned in India",
    "ddt": "banned for agriculture in India", "lindane": "banned in India", "bhc": "banned in India",
    "paraquat": "highly hazardous; restricted - many states advise against it", "carbofuran": "WHO class Ib; 3G granules restricted",
    "methamidophos": "banned in India", "parathion": "banned in India", "ethyl parathion": "banned in India",
    "chlorpyrifos": "under review; banned for several crops - prefer alternatives", "profenofos": "restricted; avoid on vegetables",
    "acephate": "restricted on vegetables/tea in India", "carbosulfan": "restricted", "oxydemeton-methyl": "restricted",
    "mancozeb": None, "imidacloprid": None, "thiamethoxam": None, "glyphosate": "restricted in India (2022): only through licensed pest control operators",
}
# entries mapped to None are common actives kept here only so the dose table below can reference them

# (regex on the active name) -> (max value, unit label, regex of the unit)
DOSE_LIMITS: list[tuple[str, float, str]] = [
    (r"mancozeb", 4.0, "g/l"), (r"copper oxychloride", 4.0, "g/l"), (r"carbendazim", 2.0, "g/l"),
    (r"propiconazole", 2.0, "ml/l"), (r"hexaconazole", 3.0, "ml/l"), (r"tebuconazole", 2.0, "ml/l"),
    (r"imidacloprid", 1.0, "ml/l"), (r"thiamethoxam", 0.6, "g/l"), (r"acetamiprid", 0.6, "g/l"),
    (r"chlorantraniliprole", 0.5, "ml/l"), (r"emamectin", 0.6, "g/l"), (r"spinosad", 0.5, "ml/l"),
    (r"lambda[- ]?cyhalothrin", 1.5, "ml/l"), (r"cypermethrin", 2.0, "ml/l"), (r"quinalphos", 3.0, "ml/l"),
    (r"neem oil", 10.0, "ml/l"), (r"azadirachtin", 5.0, "ml/l"), (r"streptocycline|streptomycin", 0.5, "g/l"),
    (r"\burea\b", 150.0, "kg/acre"), (r"\bdap\b", 100.0, "kg/acre"), (r"\bmop\b|muriate of potash", 80.0, "kg/acre"),
]

_UNIT_RE = {
    "g/l": r"(\d+(?:\.\d+)?)\s*(?:g|gm|gram|grams)\s*(?:/|per)\s*(?:l|lit|litre|liter)",
    "ml/l": r"(\d+(?:\.\d+)?)\s*(?:ml|millilit(?:re|er))\s*(?:/|per)\s*(?:l|lit|litre|liter)",
    "kg/acre": r"(\d+(?:\.\d+)?)\s*(?:kg|kilo(?:gram)?s?)\s*(?:/|per)\s*(?:acre)",
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]


def check_reply(text: str) -> dict[str, Any]:
    """Return ``{ok, flags, banned, doses, annotated}``; ``annotated`` is the text with warnings appended."""
    if not text:
        return {"ok": True, "flags": [], "banned": [], "doses": [], "annotated": text}
    low = text.lower()
    banned = []
    for name, reason in BANNED_MOLECULES.items():
        if reason and re.search(rf"\b{re.escape(name)}\b", low):
            banned.append({"molecule": name, "reason": reason})
    doses = []
    for sent in _sentences(text):
        s = sent.lower()
        for active, limit, unit in DOSE_LIMITS:
            if not re.search(active, s):
                continue
            for m in re.finditer(_UNIT_RE[unit], s):
                val = float(m.group(1))
                if val > limit:
                    doses.append({"active": active.replace("\\b", "").split("|")[0], "value": val, "unit": unit,
                                  "limit": limit, "sentence": sent[:160]})
    flags = [f"banned:{b['molecule']}" for b in banned] + [f"dose:{d['active']}" for d in doses]
    annotated = text
    if banned:
        names = ", ".join(sorted({b["molecule"] for b in banned}))
        annotated += (f"\n\n⚠️ **Safety note:** {names} is banned or restricted in India "
                      f"({'; '.join(sorted({b['reason'] for b in banned}))}). Do not use it; ask your KVK or dealer for a "
                      f"registered alternative for this crop.")
    if doses:
        dose_text = "; ".join("{} {:g} {} vs typical max {:g}".format(d["active"], d["value"], d["unit"], d["limit"])
                              for d in doses[:3])
        annotated += ("\n\n⚠️ **Check the dose:** one or more quantities above are higher than the usual label dose "
                      f"({dose_text}). Follow the product label and confirm with your KVK before spraying.")
    return {"ok": not flags, "flags": flags, "banned": banned, "doses": doses, "annotated": annotated}


def safety_check_tool(text: str) -> dict[str, Any]:
    res = check_reply(text)
    return {"ok": res["ok"], "flags": res["flags"], "banned": res["banned"], "doses": res["doses"]}
