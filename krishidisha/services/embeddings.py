"""Dense multilingual retrieval for the knowledge base (optional; TF-IDF stays as the exact-match leg).

``DenseIndex`` embeds every document once with ``intfloat/multilingual-e5-small`` (118M params, 384-d,
~450 MB RAM, ~40 ms per query on the laptop CPU) and caches the matrix in ``models/kb_index.npz`` keyed
by a hash of the corpus + model name, so restarts are instant. Hindi / Marathi / Tamil queries now retrieve
the English guides. ``fuse`` merges dense and TF-IDF rankings with reciprocal-rank fusion.

Enable with ``KB_EMBEDDING_MODEL=intfloat/multilingual-e5-small`` (``pip install sentence-transformers``).
Empty (the default, and always in TestConfig) keeps the app TF-IDF-only and offline.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class DenseIndex:
    def __init__(self, model_name: str, cache_dir: Path | str = "models"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None
        self.matrix: np.ndarray | None = None
        self.key: str | None = None

    # ------------------------------------------------------------- model
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _prefix(self, kind: str, texts: list[str]) -> list[str]:
        # e5 models expect "query: " / "passage: " prefixes
        if "e5" in self.model_name.lower():
            return [f"{kind}: {t}" for t in texts]
        return texts

    # ------------------------------------------------------------- build
    def build(self, texts: list[str]) -> None:
        digest = hashlib.sha256((self.model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        self.key = digest
        cache = self.cache_dir / f"kb_index_{digest}.npz"
        if cache.exists():
            self.matrix = np.load(cache)["matrix"]
            log.info("dense KB index loaded from cache (%d docs)", len(self.matrix))
            return
        emb = self.model.encode(self._prefix("passage", texts), batch_size=32, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        self.matrix = emb.astype(np.float32)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, matrix=self.matrix)
            for old in self.cache_dir.glob("kb_index_*.npz"):
                if old != cache:
                    old.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not cache KB index: %s", exc)
        log.info("dense KB index built (%d docs)", len(self.matrix))

    # ------------------------------------------------------------- query
    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self.matrix is None:
            return []
        q = self.model.encode(self._prefix("query", [query]), normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def fuse(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion of several (doc_index, score) rankings -> (doc_index, fused_score)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
