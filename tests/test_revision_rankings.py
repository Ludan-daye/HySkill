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


_PUBLISHED_METRICS: tuple[str, ...] = tuple(
    f"{family}@{depth}" for depth in (1, 5, 10, 50) for family in ("Recall", "nDCG")
)
_GATE_TAG: str = "demo-tag"
_GATE_DOMAIN: str = "theoremqa"
_GATE_ENCODER: str = "sentence-transformers/all-MiniLM-L6-v2"
_GATE_SPLITS: tuple[tuple[str, str], ...] = (
    ("val-1", "validation"),
    ("test-1", "test"),
)


def _gate_metric_values(variant: str, recall_at_1: float) -> dict[str, float]:
    values: dict[str, float] = {name: 0.5 for name in _PUBLISHED_METRICS}
    values["nDCG@10"] = 0.8 if variant == "naive_passage" else 0.1
    values["Recall@1"] = recall_at_1
    values["hit@50"] = 1.0
    return values


def _gate_recomputed_rows(recall_at_1: float) -> list[dict[str, object]]:
    from hyskill.k_ablation import ALL_VARIANTS

    rows: list[dict[str, object]] = []
    for variant in ALL_VARIANTS:
        metrics: dict[str, float] = _gate_metric_values(variant, recall_at_1)
        for instance_id, split in _GATE_SPLITS:
            row: dict[str, object] = {
                "model": _GATE_TAG,
                "domain": _GATE_DOMAIN,
                "arm": variant,
                "instance_id": instance_id,
                "split": split,
                "encoder": _GATE_ENCODER,
                "k_samples": 2,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def _mean_metric(rows: list[dict[str, object]], metric: str) -> float:
    if not rows:
        raise AssertionError(f"test fixture has no rows for metric={metric}")
    return sum(float(row[metric]) for row in rows) / len(rows)


def _published_from_recomputed(
    recomputed: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the fixture the way the published K=2 table is aggregated."""

    domains: list[str] = sorted({str(row["domain"]) for row in recomputed})
    variants: list[str] = sorted({str(row["arm"]) for row in recomputed})
    published: list[dict[str, object]] = [
        {
            "schema_version": 1,
            "tag": _GATE_TAG,
            "model": _GATE_TAG,
            "domain": _GATE_DOMAIN,
            "k_samples": 1,
            "variant": "naive_passage",
            "split": "test",
            "unit": "instances",
            "n": 1,
            "metric": "Recall@1",
            "value": 9.0,
        }
    ]
    for domain in domains:
        for variant in variants:
            domain_rows: list[dict[str, object]] = [
                row
                for row in recomputed
                if row["domain"] == domain and row["arm"] == variant
            ]
            for split in ("all", "validation", "test"):
                selected: list[dict[str, object]] = (
                    domain_rows
                    if split == "all"
                    else [row for row in domain_rows if row["split"] == split]
                )
                for metric in _PUBLISHED_METRICS:
                    published.append(
                        {
                            "schema_version": 1,
                            "tag": _GATE_TAG,
                            "model": _GATE_TAG,
                            "domain": domain,
                            "k_samples": 2,
                            "variant": variant,
                            "split": split,
                            "unit": "instances",
                            "n": len(selected),
                            "metric": metric,
                            "value": _mean_metric(selected, metric),
                        }
                    )
    for variant in variants:
        for split in ("all", "validation", "test"):
            for metric in _PUBLISHED_METRICS:
                selected_all: list[dict[str, object]] = []
                domain_means: list[float] = []
                for domain in domains:
                    domain_rows = [
                        row
                        for row in recomputed
                        if row["domain"] == domain and row["arm"] == variant
                    ]
                    selected = (
                        domain_rows
                        if split == "all"
                        else [row for row in domain_rows if row["split"] == split]
                    )
                    selected_all.extend(selected)
                    domain_means.append(_mean_metric(selected, metric))
                published.append(
                    {
                        "schema_version": 1,
                        "tag": _GATE_TAG,
                        "model": _GATE_TAG,
                        "domain": "__all_micro__",
                        "k_samples": 2,
                        "variant": variant,
                        "split": split,
                        "unit": "instances",
                        "n": len(selected_all),
                        "metric": metric,
                        "value": _mean_metric(selected_all, metric),
                    }
                )
                published.append(
                    {
                        "schema_version": 1,
                        "tag": _GATE_TAG,
                        "model": _GATE_TAG,
                        "domain": "__all_macro__",
                        "k_samples": 2,
                        "variant": variant,
                        "split": split,
                        "unit": "domains",
                        "n": len(domains),
                        "metric": metric,
                        "value": sum(domain_means) / len(domain_means),
                    }
                )
    return published


def _replace_published_value(
    published: list[dict[str, object]],
    *,
    domain: str,
    variant: str,
    split: str,
    metric: str,
    value: float,
) -> None:
    matches: list[dict[str, object]] = [
        row
        for row in published
        if row["k_samples"] == 2
        and row["domain"] == domain
        and row["variant"] == variant
        and row["split"] == split
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one published cell, found {len(matches)}: "
            f"domain={domain}, variant={variant}, split={split}, metric={metric}"
        )
    matches[0]["value"] = value


def _routed_rankings(ranked_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "model": _GATE_TAG,
            "domain": _GATE_DOMAIN,
            "arm": "routed",
            "instance_id": instance_id,
            "gold_skill_ids": ["gold"],
            "ranked_ids": list(ranked_ids),
            "omitted_count": None,
        }
        for instance_id, _split in _GATE_SPLITS
    ]


def _committed_routed(ranked_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = _routed_rankings(ranked_ids)
    for row in rows:
        row["arm"] = "routed_k2"
    return rows


def _router() -> dict[str, object]:
    return {"domains": {_GATE_DOMAIN: {"pick": "naive_passage"}}}


def _bm25_ok() -> dict[str, object]:
    return {"explanation": "unit-test fixture", "models": {}}


def _run_gate(
    recomputed: list[dict[str, object]],
    published: list[dict[str, object]],
    recomputed_rankings: list[dict[str, object]],
    committed_routed: list[dict[str, object]],
) -> dict[str, object]:
    from scripts.check_revision_reproduction import reproduction_report

    return reproduction_report(
        tag=_GATE_TAG,
        recomputed_metrics=recomputed,
        published_metrics=published,
        router_decisions=_router(),
        recomputed_rankings=recomputed_rankings,
        committed_routed=committed_routed,
        bm25_report=_bm25_ok(),
    )


def test_gate_passes_on_equal_metrics() -> None:
    ranked: list[str] = ["gold", "other"]
    recomputed: list[dict[str, object]] = _gate_recomputed_rows(0.5)
    published: list[dict[str, object]] = _published_from_recomputed(recomputed)
    report: dict[str, object] = _run_gate(
        recomputed,
        published,
        _routed_rankings(ranked),
        _committed_routed(ranked),
    )
    assert report["passed"] is True
    assert report["metric_mismatches"] == []
    assert report["router_pick_mismatches"] == []
    assert report["routed_top50_overlap"] == {_GATE_DOMAIN: 1.0}


def test_gate_fails_on_fourth_decimal_only_when_third_differs() -> None:
    ranked: list[str] = ["gold", "other"]
    close_rows: list[dict[str, object]] = _gate_recomputed_rows(0.5034)
    close_published: list[dict[str, object]] = _published_from_recomputed(close_rows)
    _replace_published_value(
        close_published,
        domain=_GATE_DOMAIN,
        variant="naive_passage",
        split="test",
        metric="Recall@1",
        value=0.5031,
    )
    close_report: dict[str, object] = _run_gate(
        close_rows,
        close_published,
        _routed_rankings(ranked),
        _committed_routed(ranked),
    )
    assert close_report["passed"] is True
    assert close_report["metric_mismatches"] == []

    far_rows: list[dict[str, object]] = _gate_recomputed_rows(0.504)
    far_published: list[dict[str, object]] = _published_from_recomputed(far_rows)
    _replace_published_value(
        far_published,
        domain=_GATE_DOMAIN,
        variant="naive_passage",
        split="test",
        metric="Recall@1",
        value=0.503,
    )
    far_report: dict[str, object] = _run_gate(
        far_rows,
        far_published,
        _routed_rankings(ranked),
        _committed_routed(ranked),
    )
    assert far_report["passed"] is False
    mismatches: list[dict[str, object]] = far_report["metric_mismatches"]  # type: ignore[assignment]
    assert len(mismatches) == 1
    assert mismatches[0]["domain"] == _GATE_DOMAIN
    assert mismatches[0]["variant"] == "naive_passage"
    assert mismatches[0]["split"] == "test"
    assert mismatches[0]["metric"] == "Recall@1"
    assert mismatches[0]["recomputed"] == 0.504
    assert mismatches[0]["published"] == 0.503


def test_gate_tolerates_tie_reorder() -> None:
    from hyskill.revision_rankings import instance_metrics

    original: list[str] = ["gold", "tie_a", "tie_b"]
    swapped: list[str] = ["gold", "tie_b", "tie_a"]
    base: dict[str, object] = {
        "model": _GATE_TAG,
        "domain": _GATE_DOMAIN,
        "arm": "routed",
        "instance_id": "val-1",
        "gold_skill_ids": ["gold"],
        "ranked_ids": original,
        "omitted_count": None,
    }
    original_metrics: dict[str, float] = instance_metrics({**base, "ranked_ids": original})
    swapped_metrics: dict[str, float] = instance_metrics({**base, "ranked_ids": swapped})
    assert original != swapped
    assert original_metrics == swapped_metrics

    recomputed: list[dict[str, object]] = _gate_recomputed_rows(0.5)
    for row in recomputed:
        if row["arm"] != "routed":
            continue
        for metric, value in swapped_metrics.items():
            row[metric] = value
    published: list[dict[str, object]] = _published_from_recomputed(recomputed)
    report: dict[str, object] = _run_gate(
        recomputed,
        published,
        _routed_rankings(swapped),
        _committed_routed(original),
    )
    assert report["passed"] is True
    assert report["metric_mismatches"] == []
    assert report["router_pick_mismatches"] == []
    assert report["routed_top50_overlap"] == {_GATE_DOMAIN: 1.0}


def test_gate_rejects_mismatched_encoder() -> None:
    from scripts.check_revision_reproduction import reproduction_report

    ranked: list[str] = ["gold", "other"]
    recomputed: list[dict[str, object]] = _gate_recomputed_rows(0.5)
    published: list[dict[str, object]] = _published_from_recomputed(recomputed)
    with pytest.raises(ValueError, match="BAAI/bge-base-en-v1.5"):
        reproduction_report(
            tag=_GATE_TAG,
            recomputed_metrics=recomputed,
            published_metrics=published,
            router_decisions=_router(),
            recomputed_rankings=_routed_rankings(ranked),
            committed_routed=_committed_routed(ranked),
            bm25_report=_bm25_ok(),
            expected_encoder="BAAI/bge-base-en-v1.5",
        )
