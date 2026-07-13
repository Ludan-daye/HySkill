import hashlib
import json
from pathlib import Path

import numpy as np

from hyskill.two_stage import TwoStageRetriever

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
    n_failures = 0

    def generate(self, query):
        md = ("---\nname: hyp\ndescription: " + query[:60] + "\n---\n"
              "1. do it\n\n```python\nsolve()\n```")
        return [md, md]


def _make(recall_k=4):
    corpus = json.loads(FIXTURE.read_text())
    r = TwoStageRetriever(corpus_path=str(FIXTURE), generator=EchoGenerator(),
                          st_model=FakeST(), recall_k=recall_k)
    ids = [s["skill_id"] for s in corpus]
    texts = ["\n".join([s["name"], s["description"], s["content"]]) for s in corpus]
    r.build_index(ids, texts)
    return r


def test_protocol_and_sorted_scores():
    r = _make()
    out = r.retrieve(["clean malformed csv file", "compute lah numbers"], top_k=3)
    assert len(out) == 2
    for ranking in out:
        assert 0 < len(ranking) <= 3
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)


def test_stage2_output_subset_of_stage1_recall():
    r = _make(recall_k=3)
    out = r.retrieve(["compute lah numbers ordered partitions"], top_k=6)
    # stage 2 can only reorder/truncate the recall_k candidates
    assert len(out[0]) == 3
