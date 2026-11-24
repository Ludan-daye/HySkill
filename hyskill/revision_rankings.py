"""Recompute retrieval rankings from a committed imagination cache.

The generator never calls a model. Every hypothetical document comes from
the jsonl.gz cache that ``scripts/export_full_imagination_cache.py`` wrote,
so a later analysis can rebuild rankings on a new encoder without sampling
again.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterator, Literal, TypedDict, cast

from sragents.cli.retrieve import _build_query
from sragents.corpus import load_corpus, skill_text

from hyskill.k_ablation import (
    FIXED_VARIANTS,
    VARIANT_RETRIEVERS,
    RetrievalRecord,
    expected_router_pick,
    metric_for_record,
    require_float,
    require_integer,
    require_list,
    require_object,
    require_string,
)
from hyskill.naive_hyde import NaiveHydeRetriever
from hyskill.retriever import HySkillRetriever
from hyskill.two_stage import TwoStageRetriever


TOP_K: int = 50
RRF_K: int = 60
RECALL_K: int = 50
HIT_DEPTHS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
RANK_METRIC_DEPTHS: tuple[int, ...] = (1, 5, 10, 50)
IMAGINATION_TEMPLATES: tuple[str, ...] = ("sentence", "passage", "skill")
VARIANT_TEMPLATE: dict[str, str] = {
    "naive_sentence": "sentence",
    "naive_passage": "passage",
    "naive_skill": "skill",
    "hyskill": "skill",
    "two_stage": "skill",
}
CommittedArm = Literal["bm25", "llm_rerank", "routed_k2"]
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_ENCODERS_PATH: Path = _REPO_ROOT / "results" / "revision-ijcai27" / "encoders.json"
_BENCH_INSTANCES: Path = (
    _REPO_ROOT / "external" / "SR-Agents" / "data" / "bench" / "instances"
)


# omitted_count is null unless a reranker recorded how many candidates it appended.
RankingRow = TypedDict(
    "RankingRow",
    {
        "model": str,
        "domain": str,
        "arm": str,
        "instance_id": str,
        "gold_skill_ids": list[str],
        "ranked_ids": list[str],
        "omitted_count": int | None,
    },
)


class _PreparedQuery(TypedDict):
    instance_id: str
    query: str
    gold_skill_ids: list[str]


class CachedImaginationGenerator:
    """Return cached hypothetical documents for one template.

    ``rows`` maps query text to the per-template imagination lists. A query
    that was never cached raises ``KeyError``; there is no live-generation
    fallback. ``n_failures`` stays 0 because a miss is an error, not a
    counted generation failure.
    """

    def __init__(
        self,
        rows: dict[str, dict[str, list[str]]],
        template: str,
        k: int,
    ) -> None:
        if k < 0:
            raise ValueError(f"k must be non-negative: k={k}")
        if not template:
            raise ValueError("template must be a non-empty string")
        self._rows = rows
        self._template = template
        self._k = k
        self.n_failures = 0

    def generate(self, query: str) -> list[str]:
        """Return the first ``k`` cached documents for ``query``."""

        try:
            imaginations = self._rows[query]
        except KeyError as error:
            raise KeyError(
                f"query is not in the imagination cache: query={query!r}"
            ) from error
        try:
            documents = imaginations[self._template]
        except KeyError as error:
            raise KeyError(
                f"imagination template is missing: template={self._template}"
            ) from error
        return list(documents[: self._k])


def load_imagination_cache(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load query text to ``{sentence, passage, skill}`` imagination lists.

    The committed caches contain 3,970 rows and 3,968 unique queries. Two
    instances can share a query. Identical imagination text collapses to one
    row. Different text for that same query is a cache contradiction.
    """

    if not path.is_file():
        raise FileNotFoundError(f"imagination cache does not exist: path={path}")
    loaded: dict[str, dict[str, list[str]]] = {}
    for line_number, row in _iter_json_objects(path, gzip_file=True):
        context: str = f"{path}:{line_number}"
        query: str = require_string(row.get("query"), f"{context}.query")
        imaginations: dict[str, list[str]] = _parse_imaginations(
            row.get("imaginations"), f"{context}.imaginations"
        )
        previous: dict[str, list[str]] | None = loaded.get(query)
        if previous is None:
            loaded[query] = imaginations
            continue
        if previous != imaginations:
            raise ValueError(
                "imagination cache has conflicting text for one query: "
                f"context={context}, query={query!r}"
            )
    return loaded


def recompute_rankings(
    tag: str,
    domain: str,
    variant: str,
    encoder_name: str,
    k: int,
    corpus_path: Path,
    instances_dir: Path,
    cache_path: Path,
    emb_cache_dir: Path,
    st_model: object | None = None,
) -> list[RankingRow]:
    """Rank one domain with one fixed variant, using cached imaginations."""

    if variant not in FIXED_VARIANTS:
        raise ValueError(
            "variant must be a fixed variant: "
            f"variant={variant!r}, allowed={list(FIXED_VARIANTS)}"
        )
    if k <= 0:
        raise ValueError(f"k must be a positive integer: k={k}")
    if not tag:
        raise ValueError("tag must be a non-empty string")
    template: str = VARIANT_TEMPLATE[variant]
    cache: dict[str, dict[str, list[str]]] = load_imagination_cache(cache_path)
    prepared: list[_PreparedQuery] = _load_annotated_queries(instances_dir / f"{domain}.json")
    for item in prepared:
        query: str = item["query"]
        if query not in cache:
            raise KeyError(
                "query is not in the imagination cache: "
                f"domain={domain}, instance_id={item['instance_id']}"
            )
        available: int = len(cache[query][template])
        if available < k:
            raise ValueError(
                "imagination cache has fewer samples than k: "
                f"domain={domain}, instance_id={item['instance_id']}, "
                f"template={template}, k={k}, actual={available}"
            )
    generator = CachedImaginationGenerator(cache, template, k)
    retriever_encoder: str = encoder_name
    if st_model is None:
        retriever_encoder, _revision = resolve_encoder(encoder_name, None)
        _force_local_encoder_load()
    retriever = _build_retriever(
        variant=variant,
        generator=generator,
        encoder_name=retriever_encoder,
        emb_cache_dir=emb_cache_dir,
        corpus_path=corpus_path,
        st_model=st_model,
    )
    corpus: list[dict] = load_corpus(corpus_path)
    corpus_ids: list[str] = [
        require_string(skill.get("skill_id"), f"corpus[{index}].skill_id")
        for index, skill in enumerate(corpus)
    ]
    corpus_texts: list[str] = [skill_text(skill) for skill in corpus]
    retriever.build_index(corpus_ids, corpus_texts)
    retrieved_lists = retriever.retrieve([item["query"] for item in prepared], TOP_K)
    if len(retrieved_lists) != len(prepared):
        raise RuntimeError(
            "retriever returned a different number of rankings: "
            f"domain={domain}, variant={variant}, expected={len(prepared)}, "
            f"actual={len(retrieved_lists)}"
        )
    rows: list[RankingRow] = []
    for item, retrieved in zip(prepared, retrieved_lists):
        ranked_ids: list[str] = [skill_id for skill_id, _score in retrieved]
        if len(ranked_ids) > TOP_K:
            raise ValueError(
                "retriever returned more than top_k skills: "
                f"instance_id={item['instance_id']}, actual={len(ranked_ids)}"
            )
        _require_unique_ids(ranked_ids, f"ranking:{item['instance_id']}")
        rows.append(
            {
                "model": tag,
                "domain": domain,
                "arm": variant,
                "instance_id": item["instance_id"],
                "gold_skill_ids": list(item["gold_skill_ids"]),
                "ranked_ids": ranked_ids,
                "omitted_count": None,
            }
        )
    return rows


def route(
    rows_by_variant: dict[str, list[RankingRow]],
    validation: frozenset[str],
) -> tuple[str, list[RankingRow]]:
    """Pick the validation nDCG@10 winner and copy its rows as ``routed``."""

    if set(rows_by_variant) != set(FIXED_VARIANTS):
        raise ValueError(
            "route requires every fixed variant: "
            f"actual={sorted(rows_by_variant)}"
        )
    if not validation:
        raise ValueError("validation id set is empty")
    scores: dict[str, float] = {}
    for variant in FIXED_VARIANTS:
        by_id: dict[str, RankingRow] = _index_rows(rows_by_variant[variant], variant)
        missing: list[str] = sorted(validation - set(by_id))
        if missing:
            raise ValueError(
                "variant is missing validation instances: "
                f"variant={variant}, missing_sample={missing[:5]}"
            )
        total: float = sum(_ndcg_at_10(by_id[instance_id]) for instance_id in validation)
        scores[variant] = total / len(validation)
    pick, _degenerate = expected_router_pick(scores)
    routed: list[RankingRow] = [_copy_routed(row) for row in rows_by_variant[pick]]
    return pick, routed


def load_committed_rankings(
    tag: str,
    arm: CommittedArm,
    root: Path = Path("community-results"),
) -> list[RankingRow]:
    """Load published BM25, LLM-rerank, or K=2 routed top-50 rankings.

    K=4 rows store ``gold`` and ``top50`` triples ``[skill_id, score, flag]``.
    K=2 routed rows store ``gold_skill_ids`` and ``retrieved``. Neither source
    records an omission count, so ``omitted_count`` is ``None``.
    """

    if not tag:
        raise ValueError("tag must be a non-empty string")
    base: Path = Path(root)
    if arm in ("bm25", "llm_rerank"):
        return _load_k4_rankings(base / tag / "k4" / "retrieval_top50.jsonl.gz", tag, arm)
    if arm == "routed_k2":
        return _load_k2_rankings(base / tag / "k2" / "retrieval_top50.jsonl.gz", tag)
    raise ValueError(f"unsupported committed ranking arm: arm={arm!r}")


def load_runtime_rerank_rankings(raw_root: Path, tag: str) -> list[RankingRow]:
    """Load runtime-matched ``always_rerank`` decision rankings.

    Decision rows have no gold. Gold skill ids are joined from the benchmark
    instance file. A missing instance, a missing annotation, or a missing
    ``omitted_candidate_count`` raises. Zero is a real omission count and is
    preserved.
    """

    if not tag:
        raise ValueError("tag must be a non-empty string")
    decisions_dir: Path = Path(raw_root) / tag / "decisions"
    if not decisions_dir.is_dir():
        raise FileNotFoundError(
            f"runtime rerank decisions directory does not exist: path={decisions_dir}"
        )
    paths: list[Path] = sorted(
        path for path in decisions_dir.glob("*always_rerank*.jsonl") if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            "no always_rerank decision files found: "
            f"directory={decisions_dir}"
        )
    gold_by_domain: dict[str, dict[str, list[str]]] = {}
    rows: list[RankingRow] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, raw in _iter_json_objects(path, gzip_file=False):
            context: str = f"{path}:{line_number}"
            instance_id: str = require_string(raw.get("instance_id"), f"{context}.instance_id")
            if instance_id in seen:
                raise ValueError(
                    "duplicate runtime rerank instance: "
                    f"tag={tag}, instance_id={instance_id}, path={path}"
                )
            seen.add(instance_id)
            domain: str = require_string(raw.get("domain"), f"{context}.domain")
            model: str = require_string(raw.get("model"), f"{context}.model")
            if model != tag:
                raise ValueError(
                    "runtime rerank model does not match tag: "
                    f"context={context}, tag={tag}, model={model}"
                )
            arm: str = require_string(raw.get("arm"), f"{context}.arm")
            if arm != "always_rerank":
                raise ValueError(
                    "runtime rerank row has an unexpected arm: "
                    f"context={context}, arm={arm!r}"
                )
            if "omitted_candidate_count" not in raw or raw["omitted_candidate_count"] is None:
                raise ValueError(
                    "rerank decision is missing omitted_candidate_count: "
                    f"context={context}"
                )
            omitted_count: int = require_integer(
                raw.get("omitted_candidate_count"),
                f"{context}.omitted_candidate_count",
            )
            ranked_raw: list[object] = require_list(
                raw.get("reranked_candidate_ids"),
                f"{context}.reranked_candidate_ids",
            )
            ranked_ids: list[str] = [
                require_string(skill_id, f"{context}.reranked_candidate_ids[{index}]")
                for index, skill_id in enumerate(ranked_raw)
            ]
            _require_unique_ids(ranked_ids, context)
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": arm,
                    "instance_id": instance_id,
                    "gold_skill_ids": _gold_annotations(domain, instance_id, gold_by_domain),
                    "ranked_ids": ranked_ids,
                    "omitted_count": omitted_count,
                }
            )
    return rows


def instance_metrics(row: RankingRow) -> dict[str, float]:
    """Return hit, Recall, and nDCG at the revision depths for one ranking.

    Recall and nDCG use ``hyskill.k_ablation.metric_for_record``. Hit is 1
    when any gold id occurs in the first ``k`` ranks.
    """

    gold: set[str] = set(row["gold_skill_ids"])
    ranked: list[str] = row["ranked_ids"]
    metrics: dict[str, float] = {}
    for depth in HIT_DEPTHS:
        window: list[str] = ranked[:depth]
        metrics[f"hit@{depth}"] = 1.0 if any(skill_id in gold for skill_id in window) else 0.0
    record: RetrievalRecord = {
        "instance_id": row["instance_id"],
        "gold_skill_ids": list(row["gold_skill_ids"]),
        "retrieved": [{"skill_id": skill_id, "score": 0.0} for skill_id in ranked],
    }
    for depth in RANK_METRIC_DEPTHS:
        for family in ("Recall", "nDCG"):
            metric_name: str = f"{family}@{depth}"
            metrics[metric_name] = metric_for_record(record, metric_name)
    return metrics


def resolve_encoder(encoder_name: str, st_model: object | None) -> tuple[str, str]:
    """Return ``(retriever encoder id, revision)``.

    An injected encoder, including FakeST, keeps the caller-supplied name and
    records revision ``injected``. A named encoder must appear in
    ``results/revision-ijcai27/encoders.json``. The retriever id is then the
    local snapshot directory so sentence-transformers loads from disk.
    """

    if st_model is not None:
        return encoder_name, "injected"
    if not encoder_name:
        raise ValueError("encoder name must be a non-empty string")
    if not _ENCODERS_PATH.is_file():
        raise FileNotFoundError(f"encoder manifest does not exist: path={_ENCODERS_PATH}")
    payload: dict[str, object] = require_object(
        json.loads(_ENCODERS_PATH.read_text(encoding="utf-8")),
        str(_ENCODERS_PATH),
    )
    encoders: list[object] = require_list(payload.get("encoders"), f"{_ENCODERS_PATH}.encoders")
    matches: list[dict[str, object]] = []
    for index, item in enumerate(encoders):
        record: dict[str, object] = require_object(item, f"{_ENCODERS_PATH}.encoders[{index}]")
        name: str = require_string(record.get("name"), f"{_ENCODERS_PATH}.encoders[{index}].name")
        if name == encoder_name:
            matches.append(record)
    if not matches:
        raise ValueError(
            f"encoder is not recorded in {_ENCODERS_PATH}: encoder={encoder_name!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"encoder is recorded more than once: encoder={encoder_name!r}, count={len(matches)}"
        )
    record = matches[0]
    revision: str = require_string(record.get("revision"), "encoder.revision")
    local_dir: Path = Path(require_string(record.get("local_dir"), "encoder.local_dir"))
    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"encoder snapshot does not exist: encoder={encoder_name}, path={local_dir}"
        )
    return str(local_dir), revision


def _force_local_encoder_load() -> None:
    """Point Hugging Face reads at the mirror and refuse a hub fallback."""

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _build_retriever(
    variant: str,
    generator: CachedImaginationGenerator,
    encoder_name: str,
    emb_cache_dir: Path,
    corpus_path: Path,
    st_model: object | None,
) -> NaiveHydeRetriever | HySkillRetriever:
    """Construct the retriever named by ``VARIANT_RETRIEVERS`` with plugin defaults."""

    retriever_name: str = VARIANT_RETRIEVERS[variant]
    if retriever_name == "naive_hyde":
        return NaiveHydeRetriever(
            generator=generator,
            st_model=st_model,
            encoder_name=encoder_name,
            emb_cache_dir=emb_cache_dir,
        )
    if retriever_name == "hyskill":
        return HySkillRetriever(
            corpus_path=str(corpus_path),
            generator=generator,
            st_model=st_model,
            encoder_name=encoder_name,
            rrf_k=RRF_K,
            emb_cache_dir=emb_cache_dir,
        )
    if retriever_name == "two_stage":
        return TwoStageRetriever(
            corpus_path=str(corpus_path),
            generator=generator,
            st_model=st_model,
            encoder_name=encoder_name,
            rrf_k=RRF_K,
            emb_cache_dir=emb_cache_dir,
            recall_k=RECALL_K,
        )
    raise ValueError(
        "unsupported retriever for variant: "
        f"variant={variant}, retriever={retriever_name}"
    )


def _load_annotated_queries(path: Path) -> list[_PreparedQuery]:
    """Build retrieval queries the same way ``sragents retrieve`` does.

    Instances with an empty ``skill_annotations`` list are skipped. Query
    text comes from ``sragents.cli.retrieve._build_query``, which is what the
    committed imagination cache was exported with.
    """

    if not path.is_file():
        raise FileNotFoundError(f"instance file does not exist: path={path}")
    raw_instances: list[object] = require_list(
        json.loads(path.read_text(encoding="utf-8")),
        f"instances:{path}",
    )
    prepared: list[_PreparedQuery] = []
    seen: set[str] = set()
    for index, raw_instance in enumerate(raw_instances):
        context: str = f"instances:{path}[{index}]"
        instance: dict[str, object] = require_object(raw_instance, context)
        gold_raw: object = instance.get("skill_annotations", [])
        if not gold_raw:
            continue
        gold_list: list[object] = require_list(gold_raw, f"{context}.skill_annotations")
        instance_id: str = require_string(instance.get("instance_id"), f"{context}.instance_id")
        if instance_id in seen:
            raise ValueError(
                f"duplicate instance id: path={path}, instance_id={instance_id}"
            )
        seen.add(instance_id)
        gold_skill_ids: list[str] = [
            require_string(skill_id, f"{context}.skill_annotations[{gold_index}]")
            for gold_index, skill_id in enumerate(gold_list)
        ]
        try:
            query: str = _build_query(cast(dict, instance))
        except KeyError as error:
            raise KeyError(
                "instance is missing a field required to build the query: "
                f"context={context}, field={error}"
            ) from error
        if not isinstance(query, str) or not query:
            raise ValueError(
                "SR-Agents produced an empty query: "
                f"context={context}, instance_id={instance_id}"
            )
        prepared.append(
            {
                "instance_id": instance_id,
                "query": query,
                "gold_skill_ids": gold_skill_ids,
            }
        )
    if not prepared:
        raise ValueError(f"no annotated instances found: path={path}")
    return prepared


def _parse_imaginations(value: object, context: str) -> dict[str, list[str]]:
    raw: dict[str, object] = require_object(value, context)
    parsed: dict[str, list[str]] = {}
    for template in IMAGINATION_TEMPLATES:
        samples: list[object] = require_list(raw.get(template), f"{context}.{template}")
        parsed[template] = [
            require_string(sample, f"{context}.{template}[{index}]")
            for index, sample in enumerate(samples)
        ]
    return parsed


def _index_rows(rows: list[RankingRow], variant: str) -> dict[str, RankingRow]:
    by_id: dict[str, RankingRow] = {}
    for row in rows:
        instance_id: str = row["instance_id"]
        if instance_id in by_id:
            raise ValueError(
                "duplicate instance id in variant rankings: "
                f"variant={variant}, instance_id={instance_id}"
            )
        by_id[instance_id] = row
    return by_id


def _ndcg_at_10(row: RankingRow) -> float:
    record: RetrievalRecord = {
        "instance_id": row["instance_id"],
        "gold_skill_ids": list(row["gold_skill_ids"]),
        "retrieved": [{"skill_id": skill_id, "score": 0.0} for skill_id in row["ranked_ids"]],
    }
    return metric_for_record(record, "nDCG@10")


def _copy_routed(row: RankingRow) -> RankingRow:
    return {
        "model": row["model"],
        "domain": row["domain"],
        "arm": "routed",
        "instance_id": row["instance_id"],
        "gold_skill_ids": list(row["gold_skill_ids"]),
        "ranked_ids": list(row["ranked_ids"]),
        "omitted_count": row["omitted_count"],
    }


def _load_k4_rankings(path: Path, tag: str, arm: str) -> list[RankingRow]:
    rows: list[RankingRow] = []
    seen: set[str] = set()
    for line_number, raw in _iter_json_objects(path, gzip_file=True):
        context: str = f"{path}:{line_number}"
        variant: str = require_string(raw.get("variant"), f"{context}.variant")
        if variant != arm:
            continue
        instance_id: str = require_string(raw.get("instance_id"), f"{context}.instance_id")
        if instance_id in seen:
            raise ValueError(
                "duplicate committed ranking: "
                f"path={path}, arm={arm}, instance_id={instance_id}"
            )
        seen.add(instance_id)
        domain: str = require_string(raw.get("domain"), f"{context}.domain")
        gold_raw: list[object] = require_list(raw.get("gold"), f"{context}.gold")
        gold_skill_ids: list[str] = [
            require_string(skill_id, f"{context}.gold[{index}]")
            for index, skill_id in enumerate(gold_raw)
        ]
        top50: list[object] = require_list(raw.get("top50"), f"{context}.top50")
        ranked_ids: list[str] = []
        for index, item in enumerate(top50):
            triple: list[object] = require_list(item, f"{context}.top50[{index}]")
            if len(triple) != 3:
                raise ValueError(
                    "top50 entry must be a skill/score/flag triple: "
                    f"context={context}.top50[{index}], actual={triple!r}"
                )
            ranked_ids.append(require_string(triple[0], f"{context}.top50[{index}][0]"))
            require_float(triple[1], f"{context}.top50[{index}][1]")
            require_integer(triple[2], f"{context}.top50[{index}][2]")
        _require_unique_ids(ranked_ids, context)
        rows.append(
            {
                "model": tag,
                "domain": domain,
                "arm": arm,
                "instance_id": instance_id,
                "gold_skill_ids": gold_skill_ids,
                "ranked_ids": ranked_ids,
                "omitted_count": None,
            }
        )
    if not rows:
        raise ValueError(f"committed rankings contain no rows for arm={arm}: path={path}")
    return rows


def _load_k2_rankings(path: Path, tag: str) -> list[RankingRow]:
    rows: list[RankingRow] = []
    seen: set[str] = set()
    for line_number, raw in _iter_json_objects(path, gzip_file=True):
        context: str = f"{path}:{line_number}"
        variant: str = require_string(raw.get("variant"), f"{context}.variant")
        if variant != "routed":
            raise ValueError(
                f"k2 retrieval row is not routed: context={context}, variant={variant!r}"
            )
        model: str = require_string(raw.get("model"), f"{context}.model")
        if model != tag:
            raise ValueError(
                "k2 retrieval model does not match tag: "
                f"context={context}, tag={tag}, model={model}"
            )
        instance_id: str = require_string(raw.get("instance_id"), f"{context}.instance_id")
        if instance_id in seen:
            raise ValueError(
                "duplicate committed ranking: "
                f"path={path}, instance_id={instance_id}"
            )
        seen.add(instance_id)
        domain: str = require_string(raw.get("domain"), f"{context}.domain")
        gold_raw: list[object] = require_list(
            raw.get("gold_skill_ids"), f"{context}.gold_skill_ids"
        )
        gold_skill_ids: list[str] = [
            require_string(skill_id, f"{context}.gold_skill_ids[{index}]")
            for index, skill_id in enumerate(gold_raw)
        ]
        retrieved: list[object] = require_list(raw.get("retrieved"), f"{context}.retrieved")
        ranked_ids: list[str] = []
        for index, item in enumerate(retrieved):
            record: dict[str, object] = require_object(item, f"{context}.retrieved[{index}]")
            ranked_ids.append(
                require_string(record.get("skill_id"), f"{context}.retrieved[{index}].skill_id")
            )
            require_float(record.get("score"), f"{context}.retrieved[{index}].score")
        _require_unique_ids(ranked_ids, context)
        rows.append(
            {
                "model": model,
                "domain": domain,
                "arm": "routed_k2",
                "instance_id": instance_id,
                "gold_skill_ids": gold_skill_ids,
                "ranked_ids": ranked_ids,
                "omitted_count": None,
            }
        )
    if not rows:
        raise ValueError(f"committed rankings contain no routed k2 rows: path={path}")
    return rows


def _gold_annotations(
    domain: str,
    instance_id: str,
    cache: dict[str, dict[str, list[str]]],
) -> list[str]:
    if domain not in cache:
        path: Path = _BENCH_INSTANCES / f"{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(
                "benchmark instances are missing for the gold join: "
                f"domain={domain}, path={path}"
            )
        raw_instances: list[object] = require_list(
            json.loads(path.read_text(encoding="utf-8")),
            f"instances:{path}",
        )
        index: dict[str, list[str]] = {}
        for row_index, raw_instance in enumerate(raw_instances):
            context: str = f"instances:{path}[{row_index}]"
            instance: dict[str, object] = require_object(raw_instance, context)
            iid: str = require_string(instance.get("instance_id"), f"{context}.instance_id")
            if iid in index:
                raise ValueError(
                    f"duplicate benchmark instance id: path={path}, instance_id={iid}"
                )
            annotations: object = instance.get("skill_annotations")
            if not annotations:
                index[iid] = []
                continue
            values: list[object] = require_list(annotations, f"{context}.skill_annotations")
            index[iid] = [
                require_string(skill_id, f"{context}.skill_annotations[{gold_index}]")
                for gold_index, skill_id in enumerate(values)
            ]
        cache[domain] = index
    if instance_id not in cache[domain]:
        raise ValueError(
            "rerank decision instance is missing from benchmark instances: "
            f"domain={domain}, instance_id={instance_id}"
        )
    gold: list[str] = cache[domain][instance_id]
    if not gold:
        raise ValueError(
            "rerank decision instance has no skill annotations: "
            f"domain={domain}, instance_id={instance_id}"
        )
    return list(gold)


def _require_unique_ids(ranked_ids: list[str], context: str) -> None:
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError(f"ranking contains duplicate skill ids: context={context}")


def _iter_json_objects(path: Path, gzip_file: bool) -> Iterator[tuple[int, dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"ranking file does not exist: path={path}")
    opener = gzip.open if gzip_file else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "ranking row is not JSON: "
                    f"path={path}, line={line_number}, error={error}"
                ) from error
            yield line_number, require_object(raw, f"{path}:{line_number}")
