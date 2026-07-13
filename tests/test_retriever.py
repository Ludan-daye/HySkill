import hashlib
import json
from pathlib import Path

import numpy as np

from hyskill.retriever import HySkillRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.json"


class FakeST:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.lower()[:40].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(16)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class EchoGenerator:
    """Returns a hypothetical skill that reuses the query's words, so BM25 and
    (loosely) dense paths favour lexically-overlapping corpus skills."""

    n_failures = 0

    def __init__(self, k=2):
        self._k = k

    def generate(self, query):
        md = ("---\nname: hyp\ndescription: " + query[:60] + "\n---\n"
              "1. do it\n\n```python\nsolve()\n```")
        return [md] * self._k


class EmptyGenerator:
    n_failures = 0

    def generate(self, query):
        return []


def _make(gen):
    corpus = json.loads(FIXTURE.read_text())
    r = HySkillRetriever(corpus_path=str(FIXTURE), generator=gen,
                         st_model=FakeST())
    ids = [s["skill_id"] for s in corpus]
    texts = ["\n".join([s["name"], s["description"], s["content"]]) for s in corpus]
    r.build_index(ids, texts)
    return r


def test_protocol_shapes_and_sorting():
    r = _make(EchoGenerator())
    out = r.retrieve(["convert scanned pdf to markdown tables", "compute lah numbers"], top_k=3)
    assert len(out) == 2
    for ranking in out:
        assert 0 < len(ranking) <= 3
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)
        for sid, _ in ranking:
            assert sid.startswith("s_")


def test_bm25_path_lifts_lexical_match():
    r = _make(EchoGenerator())
    out = r.retrieve(["compute Lah numbers ordered partitions"], top_k=6)
    top_ids = [sid for sid, _ in out[0][:3]]
    assert "s_lah" in top_ids


def test_fallback_on_empty_generation_still_returns():
    r = _make(EmptyGenerator())
    out = r.retrieve(["clean malformed csv file"], top_k=4)
    assert len(out[0]) == 4  # falls back to query-only retrieval


def test_codeless_skills_survivable():
    r = _make(EchoGenerator())
    out = r.retrieve(["outline an argumentative essay thesis points"], top_k=6)
    assert "s_essay" in [sid for sid, _ in out[0]]
