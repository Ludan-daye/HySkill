"""Naive HyDE baseline: unstructured passage generation, single-vector retrieval.

Faithful port of HyDE (Gao et al., ACL 2023) to the skill corpus: average the
K passage embeddings with the query embedding (eq. 8) and rank the full-text
corpus index by inner product. No field structure, no BM25, no fusion.
"""

import numpy as np

from hyskill.embedder import Embedder


class NaiveHydeRetriever:
    def __init__(self, generator, st_model=None,
                 encoder_name: str = "BAAI/bge-base-en-v1.5"):
        self._generator = generator
        self._embedder = Embedder(model=st_model, model_name=encoder_name)

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._ids = list(corpus_ids)
        self._emb = self._embedder.encode(list(corpus_texts))

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
