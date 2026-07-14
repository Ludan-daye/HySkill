import numpy as np
import pytest

from hyskill.bm25 import FastBM25

DOCS = ["the cat sat on the mat", "dogs chase cats in the park",
        "python code for sorting lists", "matrix algebra and vectors",
        "the mat was red and the cat black"]
TOKS = [d.lower().split() for d in DOCS]


def test_backend_is_bm25s():
    assert FastBM25(TOKS)._backend == "bm25s"


def test_ranking_matches_rank_bm25():
    pytest.importorskip("rank_bm25")
    from rank_bm25 import BM25Okapi
    q = "cat on mat".lower().split()
    fast = FastBM25(TOKS).get_scores(q)
    slow = np.asarray(BM25Okapi(TOKS).get_scores(q))
    assert fast.shape == slow.shape == (len(DOCS),)
    assert list(np.argsort(-fast)) == list(np.argsort(-slow))


def test_no_match_scores_zero():
    scores = FastBM25(TOKS).get_scores(["zzz", "qqq"])
    assert float(scores.max()) == 0.0


def test_fast_bm25_retriever_protocol():
    from hyskill.bm25 import FastBM25Retriever
    r = FastBM25Retriever()
    r.build_index([f"s{i}" for i in range(len(DOCS))], DOCS)
    hits = r.retrieve(["cat on the mat", "python sorting"], top_k=2)
    assert len(hits) == 2 and len(hits[0]) == 2
    assert hits[0][0][0] == "s0"          # cat/mat doc first
    assert hits[1][0][0] == "s2"          # python doc first
    assert all(isinstance(s, float) for _, s in hits[0])
