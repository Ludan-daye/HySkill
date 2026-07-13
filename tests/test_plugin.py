import pytest

sragents_base = pytest.importorskip("sragents.retrieve.base")


def test_factories_registered():
    import hyskill.plugin  # noqa: F401  (import side-effect registers)
    names = sragents_base.list_retrievers()
    assert "hyskill" in names and "naive_hyde" in names and "dense" in names


def test_hyskill_factory_accepts_string_kwargs(tmp_path):
    import hyskill.plugin  # noqa: F401
    r = sragents_base.get(
        "hyskill",
        corpus_path="tests/fixtures/tiny_corpus.json",
        mock_generator="1", k_samples="2", rrf_k="60",
        cache_dir=str(tmp_path),
    )
    assert r is not None


def test_naive_hyde_factory_accepts_string_kwargs(tmp_path):
    import hyskill.plugin  # noqa: F401
    r = sragents_base.get(
        "naive_hyde",
        mock_generator="1", k_samples="1", cache_dir=str(tmp_path),
    )
    assert r is not None
