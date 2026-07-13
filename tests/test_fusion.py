from hyskill.fusion import rrf


def test_rrf_hand_example():
    # Path A ranks: s1, s2 ; Path B ranks: s2, s3
    # k=0 for easy math: s1 = 1/1 = 1.0 ; s2 = 1/2 + 1/1 = 1.5 ; s3 = 1/2 = 0.5
    out = rrf([["s1", "s2"], ["s2", "s3"]], k=0, top_k=3)
    assert [i for i, _ in out] == ["s2", "s1", "s3"]
    assert abs(dict(out)["s2"] - 1.5) < 1e-9


def test_rrf_absent_id_gets_no_contribution():
    out = rrf([["a"], ["b"]], k=0, top_k=10)
    assert dict(out)["a"] == dict(out)["b"] == 1.0


def test_rrf_top_k_truncates():
    out = rrf([["a", "b", "c"]], k=60, top_k=2)
    assert len(out) == 2
