"""Two-stage retrieval: fusion recall then single-vector precision rerank.

Operationalises pilot Law A — 4-path fusion wins R@10 (broad recall),
single-vector imagination wins R@1 (clean precision). Stage 1 recalls
`recall_k` candidates with the fused ranking; stage 2 reranks just those
candidates by cosine against the whole hypothetical document vector
(mean of K generated docs + query anchor, as in NaiveHyde).
Generation is cache-hit on stage 2 (same generator, same keys).
"""

import numpy as np

from hyskill.retriever import HySkillRetriever


class TwoStageRetriever(HySkillRetriever):
    def __init__(self, *args, recall_k: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self._recall_k = recall_k

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        super().build_index(corpus_ids, corpus_texts)
        self._id_pos = {sid: j for j, sid in enumerate(self._ids)}

    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        recalled = super().retrieve(queries, self._recall_k)
        q_vecs = self._embedder.encode(list(queries))
        results = []
        for query, q_vec, cands in zip(queries, q_vecs, recalled):
            docs = self._generator.generate(query)  # cache hit after stage 1
            vecs = [q_vec]
            if docs:
                vecs = list(self._embedder.encode(docs)) + [q_vec]
            v = np.mean(vecs, axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            scored = [(sid, float(self._full_emb[self._id_pos[sid]] @ v))
                      for sid, _ in cands]
            scored.sort(key=lambda kv: -kv[1])
            results.append(scored[:top_k])
        return results
