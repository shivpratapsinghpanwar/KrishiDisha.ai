"""Agricultural knowledge base + lightweight retrieval (RAG without heavy deps).

Sources:
* data/disease_info.csv       - 38 PlantVillage diseases: description + prevention steps
* data/supplement_info.csv    - product mapped to each disease
* data/knowledge/*.json       - curated crop guides, government schemes, crop calendar,
                                fertilizer requirement table (kg/ha), pest management
All documents are indexed with TF-IDF so the assistant can ground its answers.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Doc:
    id: str
    title: str
    text: str
    source: str
    meta: dict = field(default_factory=dict)


# Nutrient uptake requirements (kg/ha of N, P2O5, K2O) - standard ICAR/state
# agricultural university recommendations for an average yield target.
FERTILIZER_REQUIREMENTS: dict[str, dict[str, float]] = {
    "rice": {"N": 120, "P2O5": 60, "K2O": 40},
    "wheat": {"N": 120, "P2O5": 60, "K2O": 40},
    "maize": {"N": 120, "P2O5": 60, "K2O": 40},
    "cotton": {"N": 100, "P2O5": 50, "K2O": 50},
    "sugarcane": {"N": 250, "P2O5": 100, "K2O": 120},
    "soybean": {"N": 20, "P2O5": 60, "K2O": 40},
    "groundnut": {"N": 20, "P2O5": 40, "K2O": 40},
    "mustard": {"N": 80, "P2O5": 40, "K2O": 20},
    "chickpea": {"N": 20, "P2O5": 50, "K2O": 20},
    "pigeonpeas": {"N": 25, "P2O5": 50, "K2O": 25},
    "lentil": {"N": 20, "P2O5": 40, "K2O": 20},
    "potato": {"N": 150, "P2O5": 80, "K2O": 100},
    "tomato": {"N": 120, "P2O5": 80, "K2O": 60},
    "onion": {"N": 100, "P2O5": 50, "K2O": 50},
    "banana": {"N": 200, "P2O5": 60, "K2O": 300},
    "mango": {"N": 100, "P2O5": 50, "K2O": 100},
    "grapes": {"N": 100, "P2O5": 60, "K2O": 120},
    "apple": {"N": 70, "P2O5": 35, "K2O": 70},
    "coffee": {"N": 120, "P2O5": 90, "K2O": 120},
    "jute": {"N": 60, "P2O5": 30, "K2O": 30},
    "millets": {"N": 60, "P2O5": 30, "K2O": 20},
    "barley": {"N": 60, "P2O5": 30, "K2O": 20},
    # FCV tobacco: moderate N (excess N spoils leaf quality), potassium-hungry.
    "tobacco": {"N": 70, "P2O5": 60, "K2O": 120},
    "watermelon": {"N": 100, "P2O5": 50, "K2O": 50},
    "muskmelon": {"N": 100, "P2O5": 50, "K2O": 50},
    "papaya": {"N": 200, "P2O5": 200, "K2O": 250},
    "coconut": {"N": 100, "P2O5": 50, "K2O": 150},
    "orange": {"N": 100, "P2O5": 50, "K2O": 100},
    "pomegranate": {"N": 120, "P2O5": 60, "K2O": 120},
}

# Straight fertilizer nutrient contents (fraction).
FERTILIZER_CONTENT = {
    "Urea": {"N": 0.46},
    "DAP": {"N": 0.18, "P2O5": 0.46},
    "MOP": {"K2O": 0.60},
    "SSP": {"P2O5": 0.16},
}

ACRE_TO_HA = 0.404686


def fertilizer_calculator(crop: str, area: float, unit: str = "acre", soil_n: float | None = None,
                          soil_p: float | None = None, soil_k: float | None = None) -> dict[str, Any]:
    """Convert crop nutrient requirement into bags of Urea / DAP / MOP.

    If soil-test values (kg/ha available N, P, K) are given the requirement is
    adjusted: high soil nutrient -> 25% less, low -> 25% more (standard STCR practice).
    """
    key = crop.strip().lower()
    req = FERTILIZER_REQUIREMENTS.get(key)
    if req is None:
        # fuzzy match e.g. "paddy" -> rice
        aliases = {"paddy": "rice", "corn": "maize", "gram": "chickpea", "arhar": "pigeonpeas", "tur": "pigeonpeas",
                   "moong": "mungbean", "urad": "blackgram", "bajra": "millets", "jowar": "millets", "ragi": "millets"}
        key = aliases.get(key, key)
        req = FERTILIZER_REQUIREMENTS.get(key)
    if req is None:
        return {"error": f"No nutrient schedule for '{crop}'. Supported: {', '.join(sorted(FERTILIZER_REQUIREMENTS))}"}

    ha = area * ACRE_TO_HA if unit.lower().startswith("acre") else area
    req = dict(req)

    def adjust(nutrient: str, value: float | None, low: float, high: float) -> str:
        if value is None:
            return "not provided"
        if value < low:
            req[nutrient] *= 1.25
            return "low (dose +25%)"
        if value > high:
            req[nutrient] *= 0.75
            return "high (dose -25%)"
        return "medium (standard dose)"

    soil_status = {
        "N": adjust("N", soil_n, 280, 560),
        "P2O5": adjust("P2O5", soil_p, 10, 25),
        "K2O": adjust("K2O", soil_k, 120, 280),
    }

    need = {k: v * ha for k, v in req.items()}
    # DAP first for P, then balance N with urea, K with MOP.
    dap = need["P2O5"] / FERTILIZER_CONTENT["DAP"]["P2O5"]
    n_from_dap = dap * FERTILIZER_CONTENT["DAP"]["N"]
    urea = max(need["N"] - n_from_dap, 0) / FERTILIZER_CONTENT["Urea"]["N"]
    mop = need["K2O"] / FERTILIZER_CONTENT["MOP"]["K2O"]
    bag = 50.0
    return {
        "crop": key,
        "area": area,
        "unit": unit,
        "area_ha": round(ha, 3),
        "nutrient_requirement_kg": {k: round(v, 1) for k, v in need.items()},
        "soil_status": soil_status,
        "fertilizers_kg": {"Urea": round(urea, 1), "DAP": round(dap, 1), "MOP": round(mop, 1)},
        "bags_50kg": {"Urea": round(urea / bag, 1), "DAP": round(dap / bag, 1), "MOP": round(mop / bag, 1)},
        "schedule": [
            "Basal (at sowing/transplanting): full DAP, full MOP, 1/3 Urea",
            "First top dressing (3-4 weeks): 1/3 Urea",
            "Second top dressing (flowering/panicle initiation): 1/3 Urea",
        ],
        "note": "Indicative dose from standard state-university recommendations. Always prefer a soil-test based schedule.",
    }


class KnowledgeBase:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.kdir = self.data_dir / "knowledge"
        self.docs: list[Doc] = []
        self.disease_info: pd.DataFrame | None = None
        self.supplement_info: pd.DataFrame | None = None
        self.schemes: list[dict] = []
        self.crop_guides: dict[str, dict] = {}
        self.crop_calendar: list[dict] = []
        self.pests: list[dict] = []
        self._vectorizer = None
        self._matrix = None
        self._load()

    # ----------------------------------------------------------------- load
    def _load(self) -> None:
        try:
            self.disease_info = pd.read_csv(self.data_dir / "disease_info.csv", encoding="cp1252")
            self.supplement_info = pd.read_csv(self.data_dir / "supplement_info.csv", encoding="cp1252")
        except Exception as exc:  # pragma: no cover
            log.warning("Could not load disease CSVs: %s", exc)
            self.disease_info = pd.DataFrame(columns=["disease_name", "description", "Possible Steps", "image_url"])
            self.supplement_info = pd.DataFrame(columns=["disease_name", "supplement name", "supplement image", "buy link"])

        for _, row in self.disease_info.iterrows():
            name = str(row["disease_name"])
            self.docs.append(Doc(
                id=f"disease:{row['index']}", title=name,
                text=f"{name}. {row['description']} Prevention and treatment: {row['Possible Steps']}",
                source="PlantVillage disease guide", meta={"kind": "disease", "index": int(row["index"])},
            ))

        self.schemes = self._load_json("government_schemes.json", [])
        for s in self.schemes:
            self.docs.append(Doc(id=f"scheme:{s['id']}", title=s["name"],
                                 text=f"{s['name']} ({s.get('ministry', '')}). {s['summary']} Benefits: {s.get('benefit', '')}. "
                                      f"Eligibility: {s.get('eligibility', '')}. Apply: {s.get('how_to_apply', '')}",
                                 source="Government scheme directory", meta={"kind": "scheme"}))

        self.crop_guides = self._load_json("crop_guides.json", {})
        for crop, g in self.crop_guides.items():
            text = " ".join(f"{k.replace('_', ' ')}: {v}" for k, v in g.items() if isinstance(v, str))
            self.docs.append(Doc(id=f"crop:{crop}", title=f"{crop.title()} cultivation guide", text=text,
                                 source="KrishiDisha crop guide", meta={"kind": "crop", "crop": crop}))

        self.crop_calendar = self._load_json("crop_calendar.json", [])
        for c in self.crop_calendar:
            self.docs.append(Doc(id=f"calendar:{c['crop']}:{c['season']}", title=f"{c['crop']} ({c['season']}) calendar",
                                 text=f"{c['crop']} is a {c['season']} crop. Sowing: {c['sowing']}. Harvest: {c['harvest']}. "
                                      f"Regions: {c.get('regions', '')}. Duration: {c.get('duration_days', '')} days.",
                                 source="Crop calendar", meta={"kind": "calendar"}))

        self.pests = self._load_json("pest_management.json", [])
        for p in self.pests:
            self.docs.append(Doc(id=f"pest:{p['id']}", title=p["name"],
                                 text=f"{p['name']} affects {p['crops']}. Symptoms: {p['symptoms']}. "
                                      f"Management: {p['management']}",
                                 source="Integrated pest management guide", meta={"kind": "pest"}))

        for crop, req in FERTILIZER_REQUIREMENTS.items():
            self.docs.append(Doc(id=f"fert:{crop}", title=f"{crop} fertilizer dose",
                                 text=f"Recommended nutrient dose for {crop}: {req['N']} kg N, {req['P2O5']} kg P2O5, "
                                      f"{req['K2O']} kg K2O per hectare.", source="Fertilizer schedule",
                                 meta={"kind": "fertilizer"}))

        # Packages-of-practices chunks produced by ml/kb/ingest_pdfs.py (retrieval only, cited by page)
        self.chunks_loaded = 0
        for path in sorted((self.kdir / "chunks").glob("*.jsonl")):
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        c = json.loads(line)
                        self.docs.append(Doc(id=c["id"], title=c["title"], text=c["text"], source=c.get("source", path.stem),
                                             meta={"kind": "pop", "url": c.get("url"), "crop": c.get("crop"),
                                                   "page": c.get("page_start")}))
                        self.chunks_loaded += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("bad chunk file %s: %s", path, exc)
        self._build_index()

    def _load_json(self, name: str, default):
        path = self.kdir / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            log.warning("Bad JSON in %s: %s", path, exc)
            return default

    def _build_index(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform([f"{d.title}. {d.text}" for d in self.docs])
        except Exception as exc:  # pragma: no cover
            log.warning("TF-IDF index unavailable: %s", exc)
            self._vectorizer = None
        # Optional dense multilingual leg (KB_EMBEDDING_MODEL); TF-IDF keeps exact matches like "PM-KISAN".
        self._dense = None
        model_name = os.getenv("KB_EMBEDDING_MODEL", "").strip()
        if model_name and self.docs:
            try:
                from .embeddings import DenseIndex

                self._dense = DenseIndex(model_name, cache_dir=self.data_dir.parent / "models")
                self._dense.build([f"{d.title}. {d.text}" for d in self.docs])
            except Exception as exc:  # noqa: BLE001 - never block startup on the optional model
                log.warning("dense KB index unavailable (%s); TF-IDF only", exc)
                self._dense = None

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 4, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        if not self.docs:
            return []
        if self._vectorizer is None:
            q = query.lower()
            hits = [d for d in self.docs if any(w in (d.title + d.text).lower() for w in q.split())]
            return [self._doc_dict(d, 0.0) for d in hits[:k]]
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1]
        dense = getattr(self, "_dense", None)
        if dense is not None:
            # hybrid: reciprocal-rank fusion of TF-IDF (exact terms) and dense multilingual similarity
            from .embeddings import fuse

            try:
                tfidf_rank = [(int(i), float(sims[i])) for i in order[: k * 5] if sims[i] > 0.02]
                dense_rank = dense.search(query, k=k * 5)
                fused = fuse([tfidf_rank, dense_rank])
                out = []
                for i, score in fused:
                    d = self.docs[i]
                    if kinds and d.meta.get("kind") not in kinds:
                        continue
                    out.append(self._doc_dict(d, float(score)))
                    if len(out) >= k:
                        break
                return out
            except Exception as exc:  # noqa: BLE001 - fall back to TF-IDF only
                log.warning("dense search failed: %s", exc)
        out = []
        for i in order:
            d = self.docs[i]
            if kinds and d.meta.get("kind") not in kinds:
                continue
            if sims[i] <= 0.02:
                break
            out.append(self._doc_dict(d, float(sims[i])))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _doc_dict(d: Doc, score: float) -> dict[str, Any]:
        return {"id": d.id, "title": d.title, "text": d.text, "source": d.source, "score": round(score, 3), **d.meta}

    # -------------------------------------------------------------- disease
    def disease_by_label(self, label: str) -> dict[str, Any] | None:
        """Match a model label such as 'Tomato___Late_blight' to the CSV row."""
        if self.disease_info is None or self.disease_info.empty:
            return None
        norm = _norm(label)
        best = None
        best_score = 0
        for _, row in self.disease_info.iterrows():
            cand = _norm(str(row["disease_name"]))
            score = _overlap(norm, cand)
            if score > best_score:
                best, best_score = row, score
        if best is None or best_score < 0.5:
            return None
        idx = int(best["index"])
        supp = None
        if self.supplement_info is not None and not self.supplement_info.empty:
            srow = self.supplement_info[self.supplement_info["index"] == idx]
            if not srow.empty:
                s = srow.iloc[0]
                supp = {"name": s["supplement name"], "image": s["supplement image"], "buy_link": s["buy link"]}
        return {
            "index": idx,
            "name": str(best["disease_name"]),
            "description": str(best["description"]),
            "prevention": str(best["Possible Steps"]),
            "image_url": str(best.get("image_url", "")),
            "supplement": supp,
        }

    def all_diseases(self) -> list[dict[str, Any]]:
        out = []
        if self.disease_info is None:
            return out
        for _, row in self.disease_info.iterrows():
            out.append({"index": int(row["index"]), "name": row["disease_name"], "description": row["description"],
                        "prevention": row["Possible Steps"], "image_url": row.get("image_url", "")})
        return out

    def scheme_lookup(self, query: str | None = None) -> list[dict]:
        if not query:
            return self.schemes
        q = query.lower()
        hits = [s for s in self.schemes if q in json.dumps(s).lower()]
        return hits or self.search(query, k=3, kinds=("scheme",))

    def crop_guide(self, crop: str) -> dict | None:
        return self.crop_guides.get(crop.strip().lower())

    def calendar_for(self, crop: str | None = None, season: str | None = None, month: str | None = None) -> list[dict]:
        rows = self.crop_calendar
        if crop:
            rows = [r for r in rows if crop.lower() in r["crop"].lower()]
        if season:
            rows = [r for r in rows if season.lower() in r["season"].lower()]
        if month:
            rows = [r for r in rows if month.lower()[:3] in (r["sowing"] + r["harvest"]).lower()]
        return rows


def _norm(s: str) -> set[str]:
    s = s.lower().replace("___", " ").replace("_", " ").replace(":", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    return {w for w in re.split(r"\s+", s) if w and w not in {"bell", "two", "spotted", "gray", "leaf", "spot"} or w in {"leaf"}}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
