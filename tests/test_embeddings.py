"""Dense index + reciprocal-rank fusion with a stub encoder (no model download)."""
from __future__ import annotations

import numpy as np

from krishidisha.services.embeddings import DenseIndex, fuse


class StubEncoder:
    """Deterministic 8-d embeddings from character histograms; good enough to test ranking plumbing."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        out = []
        for t in texts:
            t = t.split(": ", 1)[-1].lower()
            v = np.array([t.count(c) for c in "aeiourstn"], dtype=np.float32)[:8]
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.stack(out)


def test_dense_index_builds_caches_and_ranks(tmp_path):
    idx = DenseIndex("stub-model", cache_dir=tmp_path)
    idx._model = StubEncoder()
    docs = ["wheat rust treatment", "tomato late blight", "sugarcane red rot", "rice blast"]
    idx.build(docs)
    assert idx.matrix.shape == (4, 8)
    assert list(tmp_path.glob("kb_index_*.npz"))
    hits = idx.search("wheat rust", k=2)
    assert len(hits) == 2 and all(isinstance(i, int) for i, _ in hits)
    # second build with the same corpus loads from cache
    idx2 = DenseIndex("stub-model", cache_dir=tmp_path)
    idx2._model = StubEncoder()
    idx2.build(docs)
    assert np.allclose(idx.matrix, idx2.matrix)


def test_fuse_prefers_documents_ranked_by_both_legs():
    dense = [(3, 0.9), (1, 0.8), (2, 0.5)]
    tfidf = [(1, 0.7), (0, 0.6), (3, 0.2)]
    fused = fuse([dense, tfidf])
    order = [i for i, _ in fused]
    assert order[0] in (1, 3) and set(order) == {0, 1, 2, 3}
    assert order.index(1) < order.index(2) and order.index(3) < order.index(0)
