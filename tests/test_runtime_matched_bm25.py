from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

import hyskill.runtime_matched_bm25 as runtime_matched_bm25
from hyskill.bm25 import FastBM25Retriever
from hyskill.runtime_matched_bm25 import (
    FROZEN_BM25_BACKEND,
    FROZEN_BM25S_VERSION,
    FROZEN_RETRIEVER_NAME,
    FROZEN_SCORING_METHOD,
    FROZEN_SRAGENTS_REVISION,
    FROZEN_TOKENIZATION,
    FROZEN_TOP_K,
    BM25Artifact,
    JsonObject,
    JsonValue,
    NativeBM25Runtime,
    RuntimeMatchedBM25Error,
    build_bm25_artifact,
    build_bm25_manifest,
    load_native_bm25_runtime,
    ordered_candidate_id_payload,
    read_json,
    require_list,
    require_object,
    sha256_file,
    sha256_json,
    validate_bm25_artifact,
    validated_corpus,
    validated_instances,
    write_json,
)


class _FakeBackend:
    def __init__(self, backend: str) -> None:
        self._backend: str = backend


class _FakeRetriever:
    def __init__(self, backend: str) -> None:
        self._ids: list[str] = []
        self._bm25: _FakeBackend = _FakeBackend(backend)

    def build_index(
        self,
        corpus_ids: list[str],
        corpus_texts: list[str],
    ) -> None:
        assert len(corpus_ids) == len(corpus_texts)
        self._ids = list(corpus_ids)

    def retrieve(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[tuple[str, float]]]:
        return [
            [
                (skill_id, float(top_k - rank))
                for rank, skill_id in enumerate(self._ids[:top_k])
            ]
            for _query in queries
        ]


def _fixture_corpus() -> list[JsonObject]:
    return [
        {
            "skill_id": f"skill_{index:03d}",
            "name": f"Skill {index}",
            "description": f"Description {index}",
            "content": f"Content {index}",
        }
        for index in range(55)
    ]


def _fixture_instances() -> list[JsonObject]:
    return [
        {
            "instance_id": "theoremqa_00000",
            "dataset": "theoremqa",
            "question": "Question zero",
            "skill_annotations": ["skill_000"],
        },
        {
            "instance_id": "theoremqa_00001",
            "dataset": "theoremqa",
            "question": "Question one",
            "skill_annotations": ["skill_001"],
        },
    ]


def _fixture_runtime() -> NativeBM25Runtime:
    return {
        "build_query": lambda instance: cast(str, instance["question"]),
        "skill_text": lambda skill: "\n".join(
            cast(str, skill[field])
            for field in ("name", "description", "content")
        ),
        "create_retriever": lambda: _FakeRetriever(FROZEN_BM25_BACKEND),
        "retriever_name": FROZEN_RETRIEVER_NAME,
        "retriever_backend": FROZEN_BM25_BACKEND,
        "bm25s_version": FROZEN_BM25S_VERSION,
        "tokenization": FROZEN_TOKENIZATION,
        "scoring_method": FROZEN_SCORING_METHOD,
        "revision": FROZEN_SRAGENTS_REVISION,
        "source_root": "/pinned/SR-Agents/src",
    }


def _write_fixture_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, list[JsonObject], list[JsonObject]]:
    corpus_path: Path = tmp_path / "corpus.json"
    instances_path: Path = tmp_path / "instances.json"
    corpus: list[JsonObject] = _fixture_corpus()
    instances: list[JsonObject] = _fixture_instances()
    write_json(corpus_path, corpus)
    write_json(instances_path, instances)
    return corpus_path, instances_path, corpus, instances


def test_build_is_deterministic_and_hashes_ordered_top50(
    tmp_path: Path,
) -> None:
    corpus_path, instances_path, corpus, instances = _write_fixture_inputs(
        tmp_path
    )
    first: BM25Artifact = build_bm25_artifact(
        instances,
        corpus,
        "theoremqa",
        _fixture_runtime(),
        sha256_file(corpus_path),
        sha256_file(instances_path),
        FROZEN_TOP_K,
    )
    second: BM25Artifact = build_bm25_artifact(
        instances,
        corpus,
        "theoremqa",
        _fixture_runtime(),
        sha256_file(corpus_path),
        sha256_file(instances_path),
        FROZEN_TOP_K,
    )
    assert first == second
    assert [candidate["skill_id"] for candidate in first["results"][0]["retrieved"]] == [
        f"skill_{index:03d}" for index in range(FROZEN_TOP_K)
    ]
    assert first["metadata"]["ordered_candidate_ids_sha256"] == sha256_json(
        ordered_candidate_id_payload(first["results"])
    )


def test_validate_checks_coverage_duplicates_order_and_sha(
    tmp_path: Path,
) -> None:
    corpus_path, instances_path, corpus, instances = _write_fixture_inputs(
        tmp_path
    )
    artifact: BM25Artifact = build_bm25_artifact(
        instances,
        corpus,
        "theoremqa",
        _fixture_runtime(),
        sha256_file(corpus_path),
        sha256_file(instances_path),
        FROZEN_TOP_K,
    )
    artifact_path: Path = tmp_path / "bm25.json"
    manifest_path: Path = tmp_path / "bm25.manifest.json"
    legacy_path: Path = tmp_path / "legacy-bm25.json"
    code_path: Path = tmp_path / "builder.py"
    code_path.write_text("pass\n", encoding="utf-8")
    write_json(artifact_path, artifact)
    manifest = build_bm25_manifest(
        artifact,
        artifact_path,
        corpus_path,
        instances_path,
        _fixture_runtime(),
        (code_path,),
    )
    write_json(manifest_path, manifest)
    write_json(
        legacy_path,
        {
            "metadata": {
                "dataset": "theoremqa",
                "retriever": FROZEN_RETRIEVER_NAME,
                "top_k": FROZEN_TOP_K,
                "corpus_size": len(corpus),
                "n_queries": len(instances),
            },
            "results": artifact["results"],
        },
    )
    report = validate_bm25_artifact(
        require_object(read_json(artifact_path), "artifact"),
        require_object(read_json(manifest_path), "manifest"),
        require_object(read_json(legacy_path), "legacy"),
        artifact_path,
        manifest_path,
        legacy_path,
        corpus_path,
        instances_path,
        "theoremqa",
    )
    assert report["valid"] is True
    assert report["instance_count"] == 2
    assert report["duplicate_candidate_ids"] == 0
    assert report["order_mismatches"] == 0

    legacy_value: JsonValue = read_json(legacy_path)
    legacy_payload: JsonObject = require_object(legacy_value, "legacy")
    legacy_results: list[JsonValue] = require_list(
        cast(JsonValue, legacy_payload["results"]),
        "legacy.results",
    )
    first_record: JsonObject = require_object(legacy_results[0], "first")
    retrieved: list[JsonValue] = require_list(
        cast(JsonValue, first_record["retrieved"]),
        "first.retrieved",
    )
    retrieved[0], retrieved[1] = retrieved[1], retrieved[0]
    write_json(legacy_path, legacy_payload)
    with pytest.raises(
        RuntimeMatchedBM25Error,
        match="ordered candidate IDs differ",
    ):
        validate_bm25_artifact(
            require_object(read_json(artifact_path), "artifact"),
            require_object(read_json(manifest_path), "manifest"),
            require_object(read_json(legacy_path), "legacy"),
            artifact_path,
            manifest_path,
            legacy_path,
            corpus_path,
            instances_path,
            "theoremqa",
        )


def test_rejects_duplicate_corpus_and_wrong_domain() -> None:
    corpus: list[JsonObject] = _fixture_corpus()
    corpus.append(dict(corpus[0]))
    with pytest.raises(RuntimeMatchedBM25Error, match="Duplicate identifier"):
        validated_corpus(cast(list[JsonValue], corpus))
    instances: list[JsonObject] = _fixture_instances()
    with pytest.raises(RuntimeMatchedBM25Error, match="domain mismatch"):
        validated_instances(cast(list[JsonValue], instances), "logicbench")


def test_rejects_rank_bm25_fallback() -> None:
    corpus: list[JsonObject] = _fixture_corpus()
    instances: list[JsonObject] = _fixture_instances()
    runtime: NativeBM25Runtime = _fixture_runtime()
    runtime["create_retriever"] = lambda: _FakeRetriever("rank_bm25")
    with pytest.raises(
        RuntimeMatchedBM25Error,
        match="fallback is forbidden",
    ):
        build_bm25_artifact(
            instances,
            corpus,
            "theoremqa",
            runtime,
            "1" * 64,
            "2" * 64,
            FROZEN_TOP_K,
        )


def test_loader_uses_legacy_fast_bm25_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout: Path = tmp_path / "SR-Agents"
    source_root: Path = checkout / "src"
    (source_root / "sragents" / "cli").mkdir(parents=True)
    retrieve_module: ModuleType = ModuleType("sragents.cli.retrieve")
    retrieve_module.__file__ = str(
        source_root / "sragents" / "cli" / "retrieve.py"
    )
    retrieve_module._build_query = lambda instance: cast(  # type: ignore[attr-defined]
        str,
        instance["question"],
    )
    corpus_module: ModuleType = ModuleType("sragents.corpus")
    corpus_module.__file__ = str(source_root / "sragents" / "corpus.py")
    corpus_module.skill_text = lambda skill: cast(  # type: ignore[attr-defined]
        str,
        skill["content"],
    )
    bm25s_module: ModuleType = ModuleType("bm25s")
    original_import_module = importlib.import_module

    def fake_import_module(module_name: str) -> ModuleType:
        modules: dict[str, ModuleType] = {
            "sragents.cli.retrieve": retrieve_module,
            "sragents.corpus": corpus_module,
            "bm25s": bm25s_module,
        }
        if module_name in modules:
            return modules[module_name]
        return cast(ModuleType, original_import_module(module_name))

    monkeypatch.setattr(
        runtime_matched_bm25,
        "git_revision",
        lambda _checkout: FROZEN_SRAGENTS_REVISION,
    )
    monkeypatch.setattr(
        runtime_matched_bm25.importlib,
        "import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        runtime_matched_bm25.importlib.metadata,
        "version",
        lambda package_name: (
            FROZEN_BM25S_VERSION
            if package_name == "bm25s"
            else pytest.fail(f"Unexpected package: {package_name}")
        ),
    )
    runtime: NativeBM25Runtime = load_native_bm25_runtime(
        checkout,
        FROZEN_SRAGENTS_REVISION,
    )
    retriever = runtime["create_retriever"]()
    assert isinstance(retriever, FastBM25Retriever)
    assert runtime["retriever_name"] == "fast_bm25"
    assert runtime["retriever_backend"] == "bm25s"
