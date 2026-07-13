"""HySkillRetriever — hypothetical-skill multi-view retrieval (SR-Agents protocol).

Pipeline per query:
  1. generate K hypothetical SKILL.md docs (query-side HyDE);
  2. parse fields, embed, per-field centroid; query vector mixed into the
     meta centroid with weight 1/(K+1)  (HyDE eq. 8 adapted);
  3. rank per path: meta<->meta, body<->body, code<->code (code-bearing
     skills only), BM25(query -> full text);
  4. RRF-fuse paths; return top_k.
Fallback when generation fails: dense query -> full-text ranking fused with BM25.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from hyskill.embedder import Embedder
from hyskill.fusion import rrf
from hyskill.parser import parse_fields, parse_generated

_PATH_DEPTH = 100  # per-path ranking depth fed into RRF


def _tok(text: str) -> list[str]:
    return text.lower().split()


def corpus_cache_key(corpus_path: str, encoder_id: str, ids: list[str]) -> str:
    """Cache key for corpus embeddings: corpus file identity + encoder + id set."""
    p = Path(corpus_path)
    st = p.stat()
    raw = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{encoder_id}|{len(ids)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class HySkillRetriever:
    def __init__(self, corpus_path: str, generator, st_model=None,
                 encoder_name: str = "BAAI/bge-base-en-v1.5", rrf_k: int = 60,
                 emb_cache_dir=None):
        self._corpus_path = corpus_path
        self._generator = generator
        self._embedder = Embedder(model=st_model, model_name=encoder_name)
        self._encoder_id = encoder_name if st_model is None else "injected"
        self._rrf_k = rrf_k
        self._emb_cache = Path(emb_cache_dir) if emb_cache_dir else None

    # ---------------------------------------------------------- indexing
    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._ids = list(corpus_ids)
        cache_file = None
        if self._emb_cache:
            key = corpus_cache_key(self._corpus_path, self._encoder_id, corpus_ids)
            cache_file = self._emb_cache / f"hyskill-{key}.npz"
            if cache_file.exists():
                z = np.load(cache_file, allow_pickle=False)
                self._meta_emb, self._body_emb = z["meta"], z["body"]
                self._code_emb, self._full_emb = z["code"], z["full"]
                self._code_ids = list(z["code_ids"])
                self._bm25 = BM25Okapi([_tok(t) for t in corpus_texts])
                return

        raw = {s["skill_id"]: s for s in json.loads(Path(self._corpus_path).read_text())}
        fields = [parse_fields(raw[i].get("name", ""), raw[i].get("description", ""),
                               raw[i].get("content", "")) if i in raw
                  else {"meta": t[:200], "body": t, "code": ""}
                  for i, t in zip(corpus_ids, corpus_texts)]

        self._meta_emb = self._embedder.encode([f["meta"] or " " for f in fields])
        self._body_emb = self._embedder.encode([f["body"] or " " for f in fields])
        self._code_ids = [i for i, f in zip(corpus_ids, fields) if f["code"]]
        self._code_emb = self._embedder.encode(
            [f["code"] for f in fields if f["code"]])
        self._full_emb = self._embedder.encode(list(corpus_texts))
        self._bm25 = BM25Okapi([_tok(t) for t in corpus_texts])

        if cache_file is not None:
            self._emb_cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file, meta=self._meta_emb, body=self._body_emb,
                code=self._code_emb, full=self._full_emb,
                code_ids=np.array(self._code_ids, dtype=str))

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _rank(query_vec: np.ndarray, matrix: np.ndarray, ids: list[str]) -> list[str]:
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ query_vec
        order = np.argsort(-scores)[:_PATH_DEPTH]
        return [ids[j] for j in order]

    def _bm25_rank(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(_tok(query))
        order = np.argsort(-scores)[:_PATH_DEPTH]
        return [self._ids[j] for j in order]

    # ---------------------------------------------------------- retrieval
    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        results = []
        q_vecs = self._embedder.encode(list(queries))
        for query, q_vec in zip(queries, q_vecs):
            docs = self._generator.generate(query)
            if not docs:
                paths = [self._rank(q_vec, self._full_emb, self._ids),
                         self._bm25_rank(query)]
                results.append(rrf(paths, k=self._rrf_k, top_k=top_k))
                continue
            parsed = [parse_generated(d) for d in docs]
            centroids = {}
            for field in ("meta", "body", "code"):
                texts = [p[field] for p in parsed if p[field]]
                if texts:
                    centroids[field] = self._embedder.encode(texts).mean(axis=0)
            k = len(docs)
            if "meta" in centroids:  # query anchor into meta path (HyDE eq. 8)
                centroids["meta"] = (centroids["meta"] * k + q_vec) / (k + 1)
            else:
                centroids["meta"] = q_vec
            for f in centroids:
                n = np.linalg.norm(centroids[f])
                if n > 0:
                    centroids[f] = centroids[f] / n

            paths = [self._rank(centroids["meta"], self._meta_emb, self._ids)]
            if "body" in centroids:
                paths.append(self._rank(centroids["body"], self._body_emb, self._ids))
            if "code" in centroids:
                paths.append(self._rank(centroids["code"], self._code_emb, self._code_ids))
            paths.append(self._bm25_rank(query))
            results.append(rrf(paths, k=self._rrf_k, top_k=top_k))
        return results
