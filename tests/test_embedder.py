import numpy as np

from hyskill.embedder import Embedder


class FakeST:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(8)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_encode_shape_and_norm():
    e = Embedder(model=FakeST())
    v = e.encode(["a", "b"])
    assert v.shape == (2, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_encode_empty_returns_empty():
    e = Embedder(model=FakeST())
    assert e.encode([]).shape[0] == 0
