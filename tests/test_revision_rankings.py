"""Cache-backed ranking recomputation for the IJCAI-27 revision."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from tests.test_naive_hyde import FakeST

FIXTURES = Path(__file__).parent / "fixtures"
TINY_CORPUS = FIXTURES / "tiny_corpus.json"
K4_FIXTURE = FIXTURES / "revision_k4_retrieval_top50.jsonl"
K2_FIXTURE = FIXTURES / "revision_k2_retrieval_top50.jsonl"


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _install_jsonl_gz(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wb") as handle:
        handle.write(source.read_bytes())


def test_cached_generator_returns_prefix() -> None:
    from hyskill.revision_rankings import CachedImaginationGenerator

    rows = {"q": {"skill": ["s0", "s1", "s2", "s3"]}}
    generator = CachedImaginationGenerator(rows, "skill", 2)
    assert generator.generate("q") == ["s0", "s1"]
    assert generator.n_failures == 0


def test_cached_generator_missing_query_raises() -> None:
    from hyskill.revision_rankings import CachedImaginationGenerator

    generator = CachedImaginationGenerator({}, "skill", 2)
    with pytest.raises(KeyError):
        generator.generate("missing")


def test_load_imagination_cache_collapses_identical_queries(tmp_path: Path) -> None:
    from hyskill.revision_rankings import load_imagination_cache

    imaginations = {
        "sentence": ["same sentence"],
        "passage": ["same passage"],
        "skill": ["same skill"],
    }
    path = tmp_path / "cache.jsonl.gz"
    _write_gzip_jsonl(
        path,
        [
            {"query": "shared", "imaginations": imaginations},
            {"query": "shared", "imaginations": imaginations},
            {"query": "other", "imaginations": imaginations},
        ],
    )
    loaded = load_imagination_cache(path)
    assert set(loaded) == {"shared", "other"}
    assert loaded["shared"] == imaginations


def test_load_imagination_cache_rejects_conflicting_queries(tmp_path: Path) -> None:
    from hyskill.revision_rankings import load_imagination_cache

    path = tmp_path / "cache.jsonl.gz"
    _write_gzip_jsonl(
        path,
        [
            {
                "query": "shared",
                "imaginations": {
                    "sentence": ["s"],
                    "passage": ["first"],
                    "skill": ["skill"],
                },
            },
            {
                "query": "shared",
                "imaginations": {
                    "sentence": ["s"],
                    "passage": ["second"],
                    "skill": ["skill"],
                },
            },
        ],
    )
    with pytest.raises(ValueError, match="shared"):
        load_imagination_cache(path)


def test_metrics_multi_gold() -> None:
    from hyskill.revision_rankings import instance_metrics

    metrics = instance_metrics(
        {
            "model": "m",
            "domain": "theoremqa",
            "arm": "naive_passage",
            "instance_id": "i1",
            "gold_skill_ids": ["a", "b"],
            "ranked_ids": ["x", "b", "a"],
            "omitted_count": None,
        }
    )
    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics["hit@1"] == 0
    assert metrics["hit@3"] == 1
    assert metrics["Recall@5"] == 1.0
    assert metrics["nDCG@10"] == expected_ndcg
    assert set(metrics) == {
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@50",
        "nDCG@1",
        "nDCG@5",
        "nDCG@10",
        "nDCG@50",
    }


def test_recompute_naive_passage_tiny(tmp_path: Path) -> None:
    from sragents.cli.retrieve import _build_query
    from sragents.corpus import load_corpus, skill_text
    from hyskill.naive_hyde import NaiveHydeRetriever
    from hyskill.revision_rankings import CachedImaginationGenerator, recompute_rankings

    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    annotated = {
        "instance_id": "theoremqa_tiny_0",
        "dataset": "theoremqa",
        "question": "clean the csv",
        "skill_annotations": ["s_csv"],
    }
    skipped = {
        "instance_id": "theoremqa_tiny_skip",
        "dataset": "theoremqa",
        "question": "this instance has no gold skill",
        "skill_annotations": [],
    }
    (instances_dir / "theoremqa.json").write_text(
        json.dumps([annotated, skipped]),
        encoding="utf-8",
    )
    query = _build_query(annotated)
    imaginations = {
        "sentence": ["one sentence skill"],
        "passage": ["a passage about clean csv", "another passage about tables"],
        "skill": ["---\nname: cleaner\ndescription: clean csv\n---\n1. parse\n"],
    }
    cache_path = tmp_path / "cache.jsonl.gz"
    _write_gzip_jsonl(cache_path, [{"query": query, "imaginations": imaginations}])
    rows = recompute_rankings(
        tag="tiny",
        domain="theoremqa",
        variant="naive_passage",
        encoder_name="BAAI/bge-base-en-v1.5",
        k=2,
        corpus_path=TINY_CORPUS,
        instances_dir=instances_dir,
        cache_path=cache_path,
        emb_cache_dir=tmp_path / "emb",
        st_model=FakeST(),
    )
    assert [row["instance_id"] for row in rows] == ["theoremqa_tiny_0"]
    ranked = rows[0]["ranked_ids"]
    assert len(ranked) <= 50
    assert len(ranked) == len(set(ranked))
    assert rows[0]["gold_skill_ids"] == ["s_csv"]
    assert rows[0]["arm"] == "naive_passage"
    assert rows[0]["omitted_count"] is None

    corpus = load_corpus(TINY_CORPUS)
    direct = NaiveHydeRetriever(
        generator=CachedImaginationGenerator({query: imaginations}, "passage", 2),
        st_model=FakeST(),
        encoder_name="BAAI/bge-base-en-v1.5",
        emb_cache_dir=tmp_path / "emb-direct",
    )
    direct.build_index(
        [skill["skill_id"] for skill in corpus],
        [skill_text(skill) for skill in corpus],
    )
    retrieved = direct.retrieve([query], top_k=50)[0]
    assert ranked == [skill_id for skill_id, _score in retrieved]
    scores = [score for _skill_id, score in retrieved]
    assert scores == sorted(scores, reverse=True)


def test_route_picks_validation_winner() -> None:
    from hyskill.k_ablation import FIXED_VARIANTS
    from hyskill.revision_rankings import route

    validation = frozenset({"val"})
    rows_by_variant: dict[str, list[dict[str, object]]] = {}
    for variant in FIXED_VARIANTS:
        if variant == "naive_passage":
            ranked_rows = [
                ["g", "x"],
                ["x", "g"],
            ]
        else:
            ranked_rows = [
                ["x", "g"],
                ["g", "x"],
            ]
        rows_by_variant[variant] = [
            {
                "model": "m",
                "domain": "theoremqa",
                "arm": variant,
                "instance_id": instance_id,
                "gold_skill_ids": ["g"],
                "ranked_ids": ranked,
                "omitted_count": None,
            }
            for instance_id, ranked in zip(("val", "test"), ranked_rows)
        ]
    pick, routed = route(rows_by_variant, validation)
    assert pick == "naive_passage"
    assert [row["arm"] for row in routed] == ["routed", "routed"]
    assert [row["ranked_ids"] for row in routed] == [["g", "x"], ["x", "g"]]
    assert rows_by_variant["naive_passage"][0]["arm"] == "naive_passage"
    routed[0]["ranked_ids"].append("mutated")
    assert rows_by_variant["naive_passage"][0]["ranked_ids"] == ["g", "x"]


def test_load_committed_rankings_schemas(tmp_path: Path) -> None:
    from hyskill.revision_rankings import load_committed_rankings

    tag = "demo-tag"
    _install_jsonl_gz(K4_FIXTURE, tmp_path / tag / "k4" / "retrieval_top50.jsonl.gz")
    _install_jsonl_gz(K2_FIXTURE, tmp_path / tag / "k2" / "retrieval_top50.jsonl.gz")
    bm25 = load_committed_rankings(tag, "bm25", root=tmp_path)
    rerank = load_committed_rankings(tag, "llm_rerank", root=tmp_path)
    routed = load_committed_rankings(tag, "routed_k2", root=tmp_path)
    assert len(bm25) == 1
    assert bm25[0]["model"] == tag
    assert bm25[0]["domain"] == "theoremqa"
    assert bm25[0]["arm"] == "bm25"
    assert bm25[0]["instance_id"] == "ex_bm25"
    assert bm25[0]["gold_skill_ids"] == ["gold_b"]
    assert bm25[0]["ranked_ids"] == ["skill_b", "skill_a"]
    assert bm25[0]["omitted_count"] is None
    assert [row["instance_id"] for row in rerank] == ["ex_rerank"]
    assert rerank[0]["arm"] == "llm_rerank"
    assert rerank[0]["gold_skill_ids"] == ["gold_r"]
    assert rerank[0]["ranked_ids"] == ["skill_r"]
    assert rerank[0]["omitted_count"] is None
    assert len(routed) == 1
    assert routed[0]["model"] == "demo-tag"
    assert routed[0]["domain"] == "champ"
    assert routed[0]["arm"] == "routed_k2"
    assert routed[0]["instance_id"] == "ex_k2"
    assert routed[0]["gold_skill_ids"] == ["gold_k"]
    assert routed[0]["ranked_ids"] == ["skill_k1", "skill_k2"]
    assert routed[0]["omitted_count"] is None


def test_runtime_rerank_missing_omitted_count_raises(tmp_path: Path) -> None:
    from hyskill.revision_rankings import load_runtime_rerank_rankings

    decisions = tmp_path / "glm4-9b" / "decisions"
    decisions.mkdir(parents=True)
    row = {
        "instance_id": "theoremqa_00000",
        "domain": "theoremqa",
        "model": "glm4-9b",
        "arm": "always_rerank",
        "reranked_candidate_ids": ["theoremqa_000"],
    }
    (decisions / "theoremqa-always_rerank.decisions.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="omitted_candidate_count"):
        load_runtime_rerank_rankings(tmp_path, "glm4-9b")
