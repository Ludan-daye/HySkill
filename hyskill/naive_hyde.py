"""Naive HyDE baseline: unstructured passage generation, single-vector retrieval.

Faithful port of HyDE (Gao et al., ACL 2023) to the skill corpus: average the
K passage embeddings with the query embedding (eq. 8) and rank the full-text
corpus index by inner product. No field structure, no BM25, no fusion.
"""

import hashlib
from pathlib import Path

import numpy as np

from hyskill.embedder import Embedder


class NaiveHydeRetriever:
    def __init__(self, generator, st_model=None,
                 encoder_name: str = "BAAI/bge-base-en-v1.5", emb_cache_dir=None):
        self._generator = generator
        self._embedder = Embedder(model=st_model, model_name=encoder_name)
        self._encoder_id = encoder_name if st_model is None else "injected"
        self._emb_cache = Path(emb_cache_dir) if emb_cache_dir else None

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._ids = list(corpus_ids)
        cache_file = None
        if self._emb_cache:
            # key over ids + a content sample: catches corpus swaps and edits cheaply
            sample = "".join(corpus_texts[:100])[:20000]
            raw = "\n".join(corpus_ids) + "|" + sample + "|" + self._encoder_id
            key = hashlib.sha256(raw.encode()).hexdigest()
            cache_file = self._emb_cache / f"naive_hyde-{key}.npz"
            if cache_file.exists():
                self._emb = np.load(cache_file, allow_pickle=False)["full"]
                return
        self._emb = self._embedder.encode(list(corpus_texts))
        if cache_file is not None:
            self._emb_cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, full=self._emb)

    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        results = []
        q_vecs = self._embedder.encode(list(queries))
        for query, q_vec in zip(queries, q_vecs):
            docs = self._generator.generate(query)
            vecs = [q_vec]
            if docs:
                vecs = list(self._embedder.encode(docs)) + [q_vec]
            v = np.mean(vecs, axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            scores = self._emb @ v
            order = np.argsort(-scores)[:top_k]
            results.append([(self._ids[j], float(scores[j])) for j in order])
        return results
