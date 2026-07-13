import hashlib
import json
from pathlib import Path

import numpy as np

from hyskill.naive_hyde import NaiveHydeRetriever

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


class OneDoc:
    n_failures = 0

    def generate(self, query):
        return ["a passage about " + query]


def test_naive_hyde_protocol():
    corpus = json.loads(FIXTURE.read_text())
    r = NaiveHydeRetriever(generator=OneDoc(), st_model=FakeST())
    ids = [s["skill_id"] for s in corpus]
    texts = ["\n".join([s["name"], s["description"], s["content"]]) for s in corpus]
    r.build_index(ids, texts)
    out = r.retrieve(["clean csv"], top_k=3)
    assert len(out) == 1 and len(out[0]) == 3
    scores = [s for _, s in out[0]]
    assert scores == sorted(scores, reverse=True)
