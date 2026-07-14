"""BM25 backend: bm25s sparse scoring with rank_bm25 fallback.

bm25s (method="robertson" = the same Okapi BM25 weighting as rank_bm25's
BM25Okapi) scores 26k docs in milliseconds via a sparse term-document
matrix, vs seconds per query for rank_bm25's pure-Python loop. Absolute
scores differ by a constant idf-normalisation factor, but rankings match —
and every consumer here (RRF paths, hybrid top-k) uses rank order only.
"""

import numpy as np


class FastBM25:
    def __init__(self, token_lists: list[list[str]]):
        try:
            import bm25s
            self._impl = bm25s.BM25(method="robertson")
            self._impl.index(token_lists, show_progress=False)
            self._backend = "bm25s"
        except ImportError:
            from rank_bm25 import BM25Okapi
            self._impl = BM25Okapi(token_lists)
            self._backend = "rank_bm25"

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        return np.asarray(self._impl.get_scores(tokens))


class FastBM25Retriever:
    """SR-Agents-protocol BM25 retriever on the FastBM25 backend — same
    ranking as `sragents retrieve --retriever bm25` at a fraction of the
    time; used to produce llm_rerank's candidate files."""

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._ids = list(corpus_ids)
        self._bm25 = FastBM25([t.lower().split() for t in corpus_texts])

    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        out = []
        for q in queries:
            scores = self._bm25.get_scores(q.lower().split())
            order = np.argsort(-scores)[:top_k]
            out.append([(self._ids[int(j)], float(scores[int(j)])) for j in order])
        return out
