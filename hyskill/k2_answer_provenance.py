"""Strict provenance reconstruction for the completed K=2 answer fleet."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
    sha256_text,
)


ArmName = Literal[
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
]
ProvenanceLevel = Literal[
    "formal_direct",
    "posthoc_structural",
    "formal_retry_after_import",
]
ProvenanceCohort = Literal["formal_k2", "early_raw_k2"]
SourceKey = tuple[str, str, str]


class JsonlRow(TypedDict):
    """One parsed JSONL object and its immutable physical-line identity."""

    value: JsonObject
    line_number: int
    line_sha256: str


class RawSourceData(TypedDict):
    """One validated raw answer file."""

    path: Path
    relative_path: str
    sha256: str
    model: str
    domain: str
    arm: str
    rows: dict[str, JsonlRow]


class RunLogData(TypedDict):
    """One Qwen reference run-history log bound to a raw answer file."""

    path: Path
    relative_path: str
    sha256: str
    model: str
    domain: str
    arm: str
    raw_source_sha256: str


@dataclass(frozen=True)
class ProvenanceContract:
    """Immutable answer support, runtime identity, and count contract."""

    models: tuple[str, ...]
    domains: tuple[str, ...]
    domain_counts: Mapping[str, int]
    answer_arms: Mapping[str, tuple[ArmName, ...]]
    raw_arms: Mapping[str, tuple[ArmName, ...]]
    runtime_models: Mapping[str, str]
    runtime_identities: Mapping[str, JsonObject]
    answer_code_bundles: Mapping[str, str]
    early_raw_models: frozenset[str]
    qwen_reference_model: str
    yi_model: str
    expected_model_cohorts: Mapping[str, Mapping[str, int]]
    expected_model_levels: Mapping[str, Mapping[str, int]]
    expected_model_provisional_rows: Mapping[str, int]
    expected_fleet_cohorts: Mapping[str, int]
    expected_fleet_levels: Mapping[str, int]
    expected_fleet_provisional_rows: int


STANDARD_ANSWER_CODE_BUNDLE: str = (
    "05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682"
)
QWEN9_ANSWER_CODE_BUNDLE: str = (
    "f796f20537fe63a484b4a302ebf3c2d5131d15aaff051404c2362be8afbe8d86"
)


PRODUCTION_RUNTIME_IDENTITIES: dict[str, JsonObject] = {
    "deepseek7b": {
        "chat_template_revision": (
            "tokenizer-config-sha256="
            "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
        ),
        "checkpoint": (
            "modelscope:deepseek-ai/deepseek-llm-7b-chat@snapshots/master;"
            "files-manifest-sha256="
            "25b7f08040a12a38ed6a4fdca625063e18091926a30813d56a3c87e3cbe1f03b"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "deepseek7b",
        "served_model": "deepseek7b",
        "tokenizer_revision": (
            "tokenizer.json-sha256="
            "a08b02921f08548065a7b2ec13b2ffeed873231add60f9c3c7b08b04f2cc212a;"
            "tokenizer-config-sha256="
            "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
        ),
        "vllm_version": "0.19.1",
    },
    "glm4-9b": {
        "chat_template_revision": (
            "tokenizer-config-sha256="
            "f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62"
        ),
        "checkpoint": (
            "modelscope:ZhipuAI/glm-4-9b-chat@snapshots/master;"
            "files-manifest-sha256="
            "cd37e55587031d4dbc51bf768f83268669e196434f209b1bd0e6245991e038be"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "glm4-9b",
        "served_model": "glm4-9b",
        "tokenizer_revision": (
            "tokenizer-config-sha256="
            "f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62;"
            "tokenizer-model-sha256="
            "5a493598071550244b2ee7f26118f3edec2150b9dfa967929a99052ac83fe716"
        ),
        "vllm_version": "0.19.1",
    },
    "llama31-8b": {
        "chat_template_revision": (
            "tokenizer-config-sha256="
            "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
        ),
        "checkpoint": (
            "modelscope:LLM-Research/Meta-Llama-3.1-8B-Instruct@"
            "snapshots/master;files-manifest-sha256="
            "a8e51a9052d5cfe3faea783aa90837c6ba39d04f438eb6eca344a0f4b1e44630"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "llama31-8b",
        "served_model": "llama31-8b",
        "tokenizer_revision": (
            "tokenizer.json-sha256="
            "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4;"
            "tokenizer-config-sha256="
            "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
        ),
        "vllm_version": "0.19.1",
    },
    "mistral7b": {
        "chat_template_revision": (
            "tokenizer-config-sha256="
            "b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d"
        ),
        "checkpoint": (
            "modelscope:LLM-Research/Mistral-7B-Instruct-v0.3@"
            "c8cfccbcfd71d4e3479498c30b2823bab19c4687;"
            "files-manifest-sha256="
            "559840283ece7b8cbbb937d74d5ce47aff520cda4a453a3331ac3e8f26bfa6df"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "mistral7b",
        "served_model": "mistral7b",
        "tokenizer_revision": (
            "tokenizer.json-sha256="
            "60b945759e27a63c3c5c0ca675881f5a73b4aa38b5d1d6818570308d4f1a3c59;"
            "tokenizer-config-sha256="
            "b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d;"
            "tokenizer-model-sha256="
            "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89"
        ),
        "vllm_version": "0.19.1",
    },
    "qwen3.5-4b-reference": {
        "chat_template_revision": (
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a;"
            "chat-template-sha256="
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
        ),
        "checkpoint": (
            "hf:Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a;"
            "files-manifest-sha256="
            "7447e4e49652e2eb494c53d808d9b4e005838b1430aecb6df8181b2105d177dc"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "qwen3.5-4b-reference",
        "served_model": "qwen3.5-4b",
        "tokenizer_revision": (
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a;"
            "tokenizer-config-sha256="
            "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
        ),
        "vllm_version": "0.17.1",
    },
    "qwen35-9b": {
        "chat_template_revision": (
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a;"
            "tokenizer-config-sha256="
            "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
        ),
        "checkpoint": (
            "hf-mirror:Qwen/Qwen3.5-9B@"
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a;"
            "files-manifest-sha256="
            "daf8a250ee437249688f839397f7908ed75e10eba31ab9a5663456c36c46b595"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "qwen35-9b",
        "served_model": "qwen35-9b",
        "tokenizer_revision": (
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a;"
            "tokenizer-json-sha256="
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42;"
            "tokenizer-config-sha256="
            "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
        ),
        "vllm_version": "0.17.1",
    },
    "yi15-9b": {
        "chat_template_revision": (
            "tokenizer-config-sha256="
            "a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4"
        ),
        "checkpoint": (
            "modelscope:01ai/Yi-1.5-9B-Chat@snapshots/master;"
            "files-manifest-sha256="
            "45eb2167b36e6209f26a897a440cf27bf002f4b1368556d9105fbe76341addca"
        ),
        "context_length": 8192,
        "dtype": "bfloat16",
        "model": "yi15-9b",
        "served_model": "yi15-9b",
        "tokenizer_revision": (
            "tokenizer.json-sha256="
            "a13ccc285aea27f5e9a98d40e04e330b01d89db6de7af10b013f56eec8eae8a2;"
            "tokenizer-config-sha256="
            "a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4;"
            "tokenizer-model-sha256="
            "386c49cf943d71aa110361135338c50e38beeff0a66593480421f37b319e1a39"
        ),
        "vllm_version": "0.19.1",
    },
}


PRODUCTION_CONTRACT = ProvenanceContract(
    models=(
        "deepseek7b",
        "glm4-9b",
        "llama31-8b",
        "mistral7b",
        "qwen3.5-4b-reference",
        "qwen35-9b",
        "yi15-9b",
    ),
    domains=("theoremqa", "logicbench", "medcalcbench", "champ"),
    domain_counts={
        "theoremqa": 747,
        "logicbench": 760,
        "medcalcbench": 1100,
        "champ": 223,
    },
    answer_arms={
        "deepseek7b": ("routed_always", "routed_gated"),
        "glm4-9b": (
            "routed_always",
            "routed_gated",
            "routed_select",
        ),
        "llama31-8b": (
            "routed_always",
            "routed_gated",
            "routed_select",
        ),
        "mistral7b": (
            "routed_always",
            "routed_gated",
            "routed_select",
        ),
        "qwen3.5-4b-reference": (
            "routed_always",
            "routed_gated",
            "routed_select",
            "fixed_gated",
        ),
        "qwen35-9b": (
            "routed_always",
            "routed_gated",
            "routed_select",
        ),
        "yi15-9b": ("routed_always", "routed_gated"),
    },
    raw_arms={
        "deepseek7b": ("routed_always", "routed_gated"),
        "glm4-9b": (),
        "llama31-8b": ("routed_always", "routed_gated"),
        "mistral7b": (),
        "qwen3.5-4b-reference": (
            "routed_always",
            "routed_gated",
            "fixed_gated",
        ),
        "qwen35-9b": (),
        "yi15-9b": ("routed_always", "routed_gated"),
    },
    runtime_models={
        "deepseek7b": "deepseek7b",
        "glm4-9b": "glm4-9b",
        "llama31-8b": "llama31-8b",
        "mistral7b": "mistral7b",
        "qwen3.5-4b-reference": "qwen3.5-4b",
        "qwen35-9b": "qwen35-9b",
        "yi15-9b": "yi15-9b",
    },
    runtime_identities=PRODUCTION_RUNTIME_IDENTITIES,
    answer_code_bundles={
        "deepseek7b": STANDARD_ANSWER_CODE_BUNDLE,
        "glm4-9b": STANDARD_ANSWER_CODE_BUNDLE,
        "llama31-8b": STANDARD_ANSWER_CODE_BUNDLE,
        "mistral7b": STANDARD_ANSWER_CODE_BUNDLE,
        "qwen3.5-4b-reference": STANDARD_ANSWER_CODE_BUNDLE,
        "qwen35-9b": QWEN9_ANSWER_CODE_BUNDLE,
        "yi15-9b": STANDARD_ANSWER_CODE_BUNDLE,
    },
    early_raw_models=frozenset(
        {"deepseek7b", "llama31-8b", "yi15-9b"}
    ),
    qwen_reference_model="qwen3.5-4b-reference",
    yi_model="yi15-9b",
    expected_model_cohorts={
        "deepseek7b": {"early_raw_k2": 5660},
        "glm4-9b": {"formal_k2": 8490},
        "llama31-8b": {"early_raw_k2": 5660, "formal_k2": 2830},
        "mistral7b": {"formal_k2": 8490},
        "qwen3.5-4b-reference": {"formal_k2": 11320},
        "qwen35-9b": {"formal_k2": 8490},
        "yi15-9b": {"early_raw_k2": 5660},
    },
    expected_model_levels={
        "deepseek7b": {"posthoc_structural": 5660},
        "glm4-9b": {"formal_direct": 8490},
        "llama31-8b": {
            "posthoc_structural": 5660,
            "formal_direct": 2830,
        },
        "mistral7b": {"formal_direct": 8490},
        "qwen3.5-4b-reference": {"formal_direct": 11320},
        "qwen35-9b": {"formal_direct": 8490},
        "yi15-9b": {
            "posthoc_structural": 5553,
            "formal_retry_after_import": 107,
        },
    },
    expected_model_provisional_rows={
        "deepseek7b": 5660,
        "glm4-9b": 0,
        "llama31-8b": 5660,
        "mistral7b": 0,
        "qwen3.5-4b-reference": 8490,
        "qwen35-9b": 0,
        "yi15-9b": 5553,
    },
    expected_fleet_cohorts={
        "early_raw_k2": 16980,
        "formal_k2": 39620,
    },
    expected_fleet_levels={
        "formal_direct": 39620,
        "posthoc_structural": 16873,
        "formal_retry_after_import": 107,
    },
    expected_fleet_provisional_rows=25363,
)


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return a JSON object or raise a contextual data error."""

    if not isinstance(value, dict):
        raise DownstreamDataError(
            f"Expected JSON object: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a JSON string, including the empty string when valid."""

    if not isinstance(value, str):
        raise DownstreamDataError(
            f"Expected string: context={context}, value={value!r}"
        )
    return value


def require_nonempty_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty JSON string."""

    output: str = require_string(value, context)
    if not output:
        raise DownstreamDataError(
            f"Expected non-empty string: context={context}"
        )
    return output


def require_integer(value: JsonValue | None, context: str) -> int:
    """Return a JSON integer."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected integer: context={context}, value={value!r}"
        )
    return value


def require_string_list(value: JsonValue | None, context: str) -> list[str]:
    """Return a duplicate-free JSON string list."""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise DownstreamDataError(
            f"Expected non-empty-string list: context={context}, "
            f"value={value!r}"
        )
    output: list[str] = cast(list[str], value)
    if len(output) != len(set(output)):
        raise DownstreamDataError(
            f"Duplicate skill IDs: context={context}, value={output}"
        )
    return output


def optional_raw_skill_ids(row: JsonObject, context: str) -> list[str]:
    """Return raw skill IDs, treating an absent field as no loaded skill."""

    if "skill_ids_used" not in row:
        return []
    return require_string_list(
        row.get("skill_ids_used"),
        f"{context}.skill_ids_used",
    )


def evidence_relative_path(path: Path, evidence_root: Path) -> str:
    """Return a normalized path below the explicit preservation root."""

    resolved_path: Path = path.resolve()
    resolved_root: Path = evidence_root.resolve()
    try:
        relative: Path = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise DownstreamDataError(
            "Evidence file is outside the preservation root: "
            f"path={resolved_path}, root={resolved_root}"
        ) from error
    output: str = relative.as_posix()
    if not output or output == "." or ".." in PurePosixPath(output).parts:
        raise DownstreamDataError(
            f"Invalid preservation-relative path: path={output!r}"
        )
    return output


def read_jsonl_rows(path: Path, context: str) -> list[JsonlRow]:
    """Parse a JSONL file while preserving physical-line hashes."""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL source does not exist: context={context}, path={path}"
        )
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise DownstreamDataError(
            f"Blank JSONL line is not allowed: context={context}, path={path}"
        )
    output: list[JsonlRow] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            raw_value: JsonValue = cast(JsonValue, json.loads(line))
        except json.JSONDecodeError as error:
            raise DownstreamDataError(
                "Malformed JSONL row: "
                f"context={context}, path={path}, line={line_number}, "
                f"column={error.colno}, message={error.msg}"
            ) from error
        output.append(
            {
                "value": require_object(
                    raw_value,
                    f"{context}:{line_number}",
                ),
                "line_number": line_number,
                "line_sha256": sha256_text(line),
            }
        )
    return output


def expected_source_keys(contract: ProvenanceContract) -> set[SourceKey]:
    """Return the exact recovered raw-source support."""

    return {
        (model, domain, arm)
        for model in contract.models
        for domain in contract.domains
        for arm in contract.raw_arms[model]
    }


def expected_run_log_keys(contract: ProvenanceContract) -> set[SourceKey]:
    """Return the exact Qwen reference run-history support."""

    model: str = contract.qwen_reference_model
    if model not in contract.models:
        return set()
    return {
        (model, domain, arm)
        for domain in contract.domains
        for arm in contract.raw_arms[model]
    }


def load_raw_source(
    path: Path,
    evidence_root: Path,
    contract: ProvenanceContract,
) -> RawSourceData:
    """Load and validate one recovered raw answer file."""

    parsed_rows: list[JsonlRow] = read_jsonl_rows(path, "raw-answer")
    if not parsed_rows:
        raise DownstreamDataError(
            f"Recovered raw answer file is empty: path={path}"
        )
    first: JsonObject = parsed_rows[0]["value"]
    runtime_model: str = require_nonempty_string(
        first.get("model"),
        f"raw-answer:{path}:1.model",
    )
    domain: str = require_nonempty_string(
        first.get("dataset"),
        f"raw-answer:{path}:1.dataset",
    )
    arm: str = require_nonempty_string(
        first.get("method"),
        f"raw-answer:{path}:1.method",
    )
    matching_models: list[str] = [
        model
        for model in contract.models
        if contract.runtime_models[model] == runtime_model
        and domain in contract.domains
        and arm in contract.raw_arms[model]
    ]
    if len(matching_models) != 1:
        raise DownstreamDataError(
            "Raw answer identity does not resolve to one frozen source: "
            f"path={path}, model={runtime_model}, domain={domain}, arm={arm}, "
            f"matches={matching_models}"
        )
    result_model: str = matching_models[0]
    expected_count: int = contract.domain_counts[domain]
    if len(parsed_rows) != expected_count:
        raise DownstreamDataError(
            "Raw answer row count mismatch: "
            f"path={path}, model={result_model}, domain={domain}, arm={arm}, "
            f"expected={expected_count}, actual={len(parsed_rows)}"
        )
    rows: dict[str, JsonlRow] = {}
    for parsed in parsed_rows:
        row: JsonObject = parsed["value"]
        context: str = (
            f"raw-answer:{path}:{parsed['line_number']}"
        )
        expected_identity: dict[str, str] = {
            "model": runtime_model,
            "dataset": domain,
            "method": arm,
        }
        for field, expected_value in expected_identity.items():
            if row.get(field) != expected_value:
                raise DownstreamDataError(
                    "Mixed raw answer identity: "
                    f"context={context}, field={field}, "
                    f"expected={expected_value!r}, actual={row.get(field)!r}"
                )
        instance_id: str = require_nonempty_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        if not instance_id.startswith(f"{domain}_"):
            raise DownstreamDataError(
                f"Raw instance ID has the wrong domain prefix: "
                f"context={context}, instance_id={instance_id}"
            )
        if instance_id in rows:
            raise DownstreamDataError(
                f"Duplicate raw instance ID: context={context}, "
                f"instance_id={instance_id}"
            )
        require_string(row.get("raw_output"), f"{context}.raw_output")
        optional_raw_skill_ids(row, context)
        rows[instance_id] = parsed
    return {
        "path": path.resolve(),
        "relative_path": evidence_relative_path(path, evidence_root),
        "sha256": sha256_file(path),
        "model": result_model,
        "domain": domain,
        "arm": arm,
        "rows": rows,
    }


def load_raw_sources(
    paths: Sequence[Path],
    evidence_root: Path,
    contract: ProvenanceContract,
) -> dict[SourceKey, RawSourceData]:
    """Load the exact frozen set of recovered raw answer files."""

    output: dict[SourceKey, RawSourceData] = {}
    seen_hashes: set[str] = set()
    for path in paths:
        source: RawSourceData = load_raw_source(
            path.resolve(),
            evidence_root,
            contract,
        )
        key: SourceKey = (
            source["model"],
            source["domain"],
            source["arm"],
        )
        if key in output:
            raise DownstreamDataError(
                f"Duplicate recovered raw source: key={key}"
            )
        if source["sha256"] in seen_hashes:
            raise DownstreamDataError(
                "Recovered raw sources duplicate a whole-file digest: "
                f"sha256={source['sha256']}"
            )
        output[key] = source
        seen_hashes.add(source["sha256"])
    expected: set[SourceKey] = expected_source_keys(contract)
    if set(output) != expected:
        raise DownstreamDataError(
            "Recovered raw-source support mismatch: "
            f"missing={sorted(expected - set(output))}, "
            f"unexpected={sorted(set(output) - expected)}"
        )
    return output


def require_log_line_once(text: str, expected_line: str, path: Path) -> None:
    """Require one exact non-progress line in a recovered run log."""

    occurrences: int = sum(
        line == expected_line for line in text.replace("\r", "\n").splitlines()
    )
    if occurrences != 1:
        raise DownstreamDataError(
            "Qwen run-history line mismatch: "
            f"path={path}, expected={expected_line!r}, "
            f"occurrences={occurrences}"
        )


def qwen_provider_source(model: str, domain: str, arm: str) -> str:
    """Return the frozen Qwen K=2 provider source for one raw arm."""

    if arm == "routed_always":
        return (
            f"results/k-ablation/{model}/routed/k2/"
            f"{domain}-routed.json"
        )
    if arm == "routed_gated":
        return f"results/k2-main/{model}/{domain}-routed-gated.json"
    if arm == "fixed_gated":
        return f"results/k2-main/{model}/{domain}-fixed-gated.json"
    raise DownstreamDataError(
        f"Qwen raw run has an unsupported arm: arm={arm}"
    )


def load_qwen_run_log(
    path: Path,
    evidence_root: Path,
    source: RawSourceData,
    contract: ProvenanceContract,
) -> RunLogData:
    """Validate one Qwen direct-engine run-history log."""

    model: str = contract.qwen_reference_model
    domain: str = source["domain"]
    arm: str = source["arm"]
    runtime_model: str = contract.runtime_models[model]
    expected_name: str = f"{domain}-{arm}.log"
    if path.name != expected_name:
        raise DownstreamDataError(
            f"Qwen run-history filename mismatch: expected={expected_name}, "
            f"actual={path.name}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Qwen run-history log is missing: path={path}"
        )
    text: str = path.read_text(encoding="utf-8")
    provider_source: str = qwen_provider_source(model, domain, arm)
    raw_relative: str = (
        f"results/k2-main/{model}/raw-answers/"
        f"{domain}-{arm}.jsonl"
    )
    expected_count: int = contract.domain_counts[domain]
    expected_lines: tuple[str, ...] = (
        f"Provider: topk {{'source': '{provider_source}', 'k': 1}}",
        "Engine:   direct   {'temperature': 0.7, 'max_tokens': 2048}",
        f"Label:    {arm}",
        f"Model:    {runtime_model}",
        f"  wrote {expected_count} records → {raw_relative}",
        (
            f"  {domain} × {runtime_model} × {arm} — "
            f"{expected_count} records"
        ),
        f"  Saved: {raw_relative.removesuffix('.jsonl')}.eval.json",
    )
    for expected_line in expected_lines:
        require_log_line_once(text, expected_line, path)
    return {
        "path": path.resolve(),
        "relative_path": evidence_relative_path(path, evidence_root),
        "sha256": sha256_file(path),
        "model": model,
        "domain": domain,
        "arm": arm,
        "raw_source_sha256": source["sha256"],
    }


def load_qwen_run_logs(
    paths: Sequence[Path],
    evidence_root: Path,
    raw_sources: Mapping[SourceKey, RawSourceData],
    contract: ProvenanceContract,
) -> dict[SourceKey, RunLogData]:
    """Load the exact twelve-file Qwen formal-direct run history."""

    model: str = contract.qwen_reference_model
    expected: set[SourceKey] = expected_run_log_keys(contract)
    path_by_name: dict[str, Path] = {}
    for path in paths:
        if path.name in path_by_name:
            raise DownstreamDataError(
                f"Duplicate Qwen run-history filename: name={path.name}"
            )
        path_by_name[path.name] = path.resolve()
    output: dict[SourceKey, RunLogData] = {}
    for key in sorted(expected):
        _, domain, arm = key
        expected_name: str = f"{domain}-{arm}.log"
        if expected_name not in path_by_name:
            raise DownstreamDataError(
                f"Missing Qwen run-history log: key={key}, "
                f"expected_name={expected_name}"
            )
        if key not in raw_sources:
            raise DownstreamDataError(
                f"Qwen run history lacks its raw source: key={key}"
            )
        output[key] = load_qwen_run_log(
            path_by_name[expected_name],
            evidence_root,
            raw_sources[key],
            contract,
        )
    used_names: set[str] = {
        Path(log["relative_path"]).name for log in output.values()
    }
    unexpected_names: set[str] = set(path_by_name) - used_names
    if unexpected_names:
        raise DownstreamDataError(
            f"Unexpected Qwen run-history logs: names={sorted(unexpected_names)}"
        )
    if model not in contract.models and paths:
        raise DownstreamDataError(
            "Qwen logs were supplied to a contract without Qwen support"
        )
    return output


def validate_runtime_identity(
    row: JsonObject,
    model: str,
    context: str,
    contract: ProvenanceContract,
) -> None:
    """Validate frozen checkpoint, tokenizer, template, runtime, and code."""

    runtime_identity: JsonObject = require_object(
        row.get("runtime_identity"),
        f"{context}.runtime_identity",
    )
    expected_runtime: JsonObject = contract.runtime_identities[model]
    if canonical_json(runtime_identity) != canonical_json(expected_runtime):
        raise DownstreamDataError(
            "Answer runtime identity mismatch: "
            f"context={context}, expected={expected_runtime}, "
            f"actual={runtime_identity}"
        )
    expected_bundle: str = contract.answer_code_bundles[model]
    if row.get("answer_code_bundle_sha256") != expected_bundle:
        raise DownstreamDataError(
            "Answer code bundle mismatch: "
            f"context={context}, expected={expected_bundle}, "
            f"actual={row.get('answer_code_bundle_sha256')!r}"
        )


def validate_final_row_identity(
    row: JsonObject,
    model: str,
    domain: str,
    arm: str,
    context: str,
    contract: ProvenanceContract,
) -> str:
    """Validate immutable final-answer identity and return its instance ID."""

    expected_runtime_model: str = contract.runtime_models[model]
    expected_identity: dict[str, JsonValue] = {
        "schema_version": "k2-answer-record-v1",
        "dataset": domain,
        "method": arm,
        "model": expected_runtime_model,
        "served_model": expected_runtime_model,
    }
    for field, expected_value in expected_identity.items():
        if row.get(field) != expected_value:
            raise DownstreamDataError(
                "Final answer identity mismatch: "
                f"context={context}, field={field}, "
                f"expected={expected_value!r}, actual={row.get(field)!r}"
            )
    instance_id: str = require_nonempty_string(
        row.get("instance_id"),
        f"{context}.instance_id",
    )
    if not instance_id.startswith(f"{domain}_"):
        raise DownstreamDataError(
            f"Final instance ID has the wrong domain prefix: "
            f"context={context}, instance_id={instance_id}"
        )
    require_string(row.get("raw_output"), f"{context}.raw_output")
    require_string_list(
        row.get("skill_ids_used"),
        f"{context}.skill_ids_used",
    )
    require_string_list(
        row.get("expected_skill_ids"),
        f"{context}.expected_skill_ids",
    )
    category: str = require_nonempty_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if category not in {"success", "method_failure"}:
        raise DownstreamDataError(
            f"Unresolved final answer failure: context={context}, "
            f"failure_category={category}"
        )
    attempts: int = require_integer(
        row.get("engine_attempts"),
        f"{context}.engine_attempts",
    )
    if attempts < 1:
        raise DownstreamDataError(
            f"Answer attempt count must be positive: context={context}, "
            f"attempts={attempts}"
        )
    injection: JsonObject = require_object(
        row.get("actual_injection_state"),
        f"{context}.actual_injection_state",
    )
    require_nonempty_string(
        injection.get("state"),
        f"{context}.actual_injection_state.state",
    )
    require_string_list(
        injection.get("skill_ids"),
        f"{context}.actual_injection_state.skill_ids",
    )
    validate_runtime_identity(row, model, context, contract)
    return instance_id


def validate_raw_fields(
    raw_row: JsonObject,
    final_row: JsonObject,
    context: str,
) -> None:
    """Compare raw and final answer identity and answer text exactly."""

    for field in ("instance_id", "dataset", "method", "model", "raw_output"):
        if raw_row.get(field) != final_row.get(field):
            raise DownstreamDataError(
                "Raw-to-final field mismatch: "
                f"context={context}, field={field}, "
                f"raw={raw_row.get(field)!r}, "
                f"final={final_row.get(field)!r}"
            )


def validate_source_path_suffix(
    source_path: str,
    raw_source: RawSourceData,
    context: str,
) -> None:
    """Require the private source path to identify the recovered raw file."""

    expected_suffix: tuple[str, ...] = (
        "results",
        "k2-main",
        raw_source["model"],
        "raw-answers",
        raw_source["path"].name,
    )
    actual_parts: tuple[str, ...] = PurePosixPath(source_path).parts
    if len(actual_parts) < len(expected_suffix) or tuple(
        actual_parts[-len(expected_suffix) :]
    ) != expected_suffix:
        raise DownstreamDataError(
            "Final answer references the wrong raw source path: "
            f"context={context}, expected_suffix={expected_suffix}, "
            f"actual={source_path!r}"
        )


def validate_structural_reference(
    final_row: JsonObject,
    raw_source: RawSourceData,
    raw_line: JsonlRow,
    context: str,
) -> None:
    """Verify one final provisional_source against recovered bytes."""

    source: JsonObject = require_object(
        final_row.get("provisional_source"),
        f"{context}.provisional_source",
    )
    if source.get("source_sha256") != raw_source["sha256"]:
        raise DownstreamDataError(
            "Raw source file hash mismatch: "
            f"context={context}, expected={raw_source['sha256']}, "
            f"actual={source.get('source_sha256')!r}"
        )
    if source.get("source_line_number") != raw_line["line_number"]:
        raise DownstreamDataError(
            "Raw source line number mismatch: "
            f"context={context}, expected={raw_line['line_number']}, "
            f"actual={source.get('source_line_number')!r}"
        )
    if source.get("source_line_sha256") != raw_line["line_sha256"]:
        raise DownstreamDataError(
            "Raw source line hash mismatch: "
            f"context={context}, expected={raw_line['line_sha256']}, "
            f"actual={source.get('source_line_sha256')!r}"
        )
    source_path: str = require_nonempty_string(
        source.get("source_path"),
        f"{context}.provisional_source.source_path",
    )
    validate_source_path_suffix(
        source_path,
        raw_source,
        context,
    )


def validate_retry_after_import(
    final_row: JsonObject,
    raw_row: JsonObject,
    model: str,
    context: str,
    contract: ProvenanceContract,
) -> None:
    """Validate one of the preregistered 107 Yi formal retries."""

    if model != contract.yi_model:
        raise DownstreamDataError(
            f"Only Yi may be a formal retry after import: context={context}"
        )
    if "provisional_source" in final_row:
        raise DownstreamDataError(
            f"Formal retry must not retain provisional_source: context={context}"
        )
    if (
        final_row.get("failure_category") != "method_failure"
        or final_row.get("engine_attempts") != 3
        or final_row.get("raw_output") != ""
    ):
        raise DownstreamDataError(
            f"Invalid Yi formal retry outcome: context={context}"
        )
    error: JsonObject = require_object(
        final_row.get("error"),
        f"{context}.error",
    )
    if error.get("exception_name") != "EmptyModelOutput":
        raise DownstreamDataError(
            f"Yi formal retry must be EmptyModelOutput: context={context}, "
            f"actual={error.get('exception_name')!r}"
        )
    final_used: list[str] = require_string_list(
        final_row.get("skill_ids_used"),
        f"{context}.skill_ids_used",
    )
    if final_used:
        raise DownstreamDataError(
            f"Failed Yi retry cannot claim loaded skills: context={context}"
        )
    raw_skills: list[str] = optional_raw_skill_ids(raw_row, context)
    expected_skills: list[str] = require_string_list(
        final_row.get("expected_skill_ids"),
        f"{context}.expected_skill_ids",
    )
    injection: JsonObject = require_object(
        final_row.get("actual_injection_state"),
        f"{context}.actual_injection_state",
    )
    requested_skills: list[str] = require_string_list(
        injection.get("skill_ids"),
        f"{context}.actual_injection_state.skill_ids",
    )
    if raw_skills != expected_skills or raw_skills != requested_skills:
        raise DownstreamDataError(
            "Yi retry skill binding mismatch: "
            f"context={context}, raw={raw_skills}, "
            f"expected={expected_skills}, requested={requested_skills}"
        )
    if injection.get("state") != "request_submitted":
        raise DownstreamDataError(
            "Yi retry must record request_submitted injection state: "
            f"context={context}, actual={injection.get('state')!r}"
        )


def count_values(
    rows: Sequence[JsonObject],
    field: str,
) -> dict[str, int]:
    """Count one required string field over provenance rows."""

    output: dict[str, int] = {}
    for index, row in enumerate(rows):
        value: str = require_nonempty_string(
            row.get(field),
            f"provenance.rows[{index}].{field}",
        )
        output[value] = output.get(value, 0) + 1
    return output


def raw_source_public_entry(source: RawSourceData) -> JsonObject:
    """Return one path-safe raw-source manifest entry."""

    return {
        "path": source["relative_path"],
        "sha256": source["sha256"],
        "rows": len(source["rows"]),
        "model": source["model"],
        "domain": source["domain"],
        "arm": source["arm"],
    }


def run_log_public_entry(log: RunLogData) -> JsonObject:
    """Return one path-safe run-history manifest entry."""

    return {
        "path": log["relative_path"],
        "sha256": log["sha256"],
        "model": log["model"],
        "domain": log["domain"],
        "arm": log["arm"],
        "raw_source_sha256": log["raw_source_sha256"],
        "generation": {
            "engine": "direct",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    }


def source_reference_entry(
    source: RawSourceData,
    line: JsonlRow,
) -> JsonObject:
    """Return one private-path-free raw source row identity."""

    return {
        "source_sha256": source["sha256"],
        "source_line_number": line["line_number"],
        "source_line_sha256": line["line_sha256"],
    }


def expected_model_rows(
    model: str,
    contract: ProvenanceContract,
) -> int:
    """Return the exact final-answer row count for one model."""

    per_arm: int = sum(contract.domain_counts.values())
    return per_arm * len(contract.answer_arms[model])


def validate_model_counts(
    model: str,
    rows: Sequence[JsonObject],
    provisional_rows: int,
    contract: ProvenanceContract,
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate exact per-model cohort, level, and source-reference counts."""

    expected_rows: int = expected_model_rows(model, contract)
    if len(rows) != expected_rows:
        raise DownstreamDataError(
            f"Provenance model row count mismatch: model={model}, "
            f"expected={expected_rows}, actual={len(rows)}"
        )
    cohorts: dict[str, int] = count_values(rows, "cohort")
    levels: dict[str, int] = count_values(rows, "provenance_level")
    expected_cohorts: dict[str, int] = dict(
        contract.expected_model_cohorts[model]
    )
    expected_levels: dict[str, int] = dict(
        contract.expected_model_levels[model]
    )
    if cohorts != expected_cohorts:
        raise DownstreamDataError(
            f"Provenance cohort count mismatch: model={model}, "
            f"expected={expected_cohorts}, actual={cohorts}"
        )
    if levels != expected_levels:
        raise DownstreamDataError(
            f"Provenance level count mismatch: model={model}, "
            f"expected={expected_levels}, actual={levels}"
        )
    expected_provisional: int = (
        contract.expected_model_provisional_rows[model]
    )
    if provisional_rows != expected_provisional:
        raise DownstreamDataError(
            f"Provisional-source count mismatch: model={model}, "
            f"expected={expected_provisional}, actual={provisional_rows}"
        )
    return cohorts, levels


def build_model_manifest(
    model: str,
    formal_root: Path,
    evidence_root: Path,
    raw_sources: Mapping[SourceKey, RawSourceData],
    run_logs: Mapping[SourceKey, RunLogData],
    contract: ProvenanceContract,
) -> JsonObject:
    """Reconstruct and verify one model's complete provenance manifest."""

    if model not in contract.models:
        raise DownstreamDataError(
            f"Unknown provenance model: model={model}"
        )
    formal_dir: Path = formal_root / model
    if not formal_dir.is_dir():
        raise NotADirectoryError(
            f"Formal model directory is missing: path={formal_dir}"
        )
    provenance_rows: list[JsonObject] = []
    formal_files: list[JsonObject] = []
    provisional_rows: int = 0
    model_raw_sources: list[RawSourceData] = []
    model_run_logs: list[RunLogData] = []
    for domain in contract.domains:
        for arm in contract.answer_arms[model]:
            file_arm: str = arm.replace("_", "-")
            answer_path: Path = formal_dir / f"{domain}-{file_arm}.jsonl"
            parsed_rows: list[JsonlRow] = read_jsonl_rows(
                answer_path,
                f"formal-answer:{model}:{domain}:{arm}",
            )
            expected_count: int = contract.domain_counts[domain]
            if len(parsed_rows) != expected_count:
                raise DownstreamDataError(
                    "Formal answer row count mismatch: "
                    f"model={model}, domain={domain}, arm={arm}, "
                    f"expected={expected_count}, actual={len(parsed_rows)}"
                )
            answer_sha256: str = sha256_file(answer_path)
            formal_files.append(
                {
                    "path": evidence_relative_path(
                        answer_path,
                        evidence_root,
                    ),
                    "sha256": answer_sha256,
                    "rows": len(parsed_rows),
                    "domain": domain,
                    "arm": arm,
                }
            )
            key: SourceKey = (model, domain, arm)
            raw_source: RawSourceData | None = raw_sources.get(key)
            if arm in contract.raw_arms[model] and raw_source is None:
                raise DownstreamDataError(
                    f"Formal answer arm lacks required raw source: key={key}"
                )
            if arm not in contract.raw_arms[model] and raw_source is not None:
                raise DownstreamDataError(
                    f"Unexpected raw source for formal answer arm: key={key}"
                )
            if raw_source is not None:
                model_raw_sources.append(raw_source)
                final_ids: set[str] = {
                    require_nonempty_string(
                        parsed["value"].get("instance_id"),
                        f"formal-answer:{key}:{parsed['line_number']}",
                    )
                    for parsed in parsed_rows
                }
                if final_ids != set(raw_source["rows"]):
                    raise DownstreamDataError(
                        "Raw/final instance coverage mismatch: "
                        f"key={key}, "
                        f"missing={sorted(set(raw_source['rows']) - final_ids)[:20]}, "
                        f"unexpected={sorted(final_ids - set(raw_source['rows']))[:20]}"
                    )
            if model == contract.qwen_reference_model and raw_source is not None:
                if key not in run_logs:
                    raise DownstreamDataError(
                        f"Qwen formal-direct arm lacks run history: key={key}"
                    )
                model_run_logs.append(run_logs[key])
            seen_ids: set[str] = set()
            for parsed in sorted(
                parsed_rows,
                key=lambda item: require_nonempty_string(
                    item["value"].get("instance_id"),
                    f"formal-answer:{key}:{item['line_number']}",
                ),
            ):
                row: JsonObject = parsed["value"]
                context: str = (
                    f"formal-answer:{model}:{domain}:{arm}:"
                    f"{parsed['line_number']}"
                )
                instance_id: str = validate_final_row_identity(
                    row,
                    model,
                    domain,
                    arm,
                    context,
                    contract,
                )
                if instance_id in seen_ids:
                    raise DownstreamDataError(
                        f"Duplicate final answer instance: context={context}, "
                        f"instance_id={instance_id}"
                    )
                seen_ids.add(instance_id)
                has_structural_source: bool = (
                    "provisional_source" in row
                )
                raw_reference: JsonObject | None = None
                skill_binding: str | None = None
                if raw_source is not None:
                    raw_line: JsonlRow = raw_source["rows"][instance_id]
                    raw_row: JsonObject = raw_line["value"]
                    validate_raw_fields(raw_row, row, context)
                    raw_reference = source_reference_entry(
                        raw_source,
                        raw_line,
                    )
                    if has_structural_source:
                        validate_structural_reference(
                            row,
                            raw_source,
                            raw_line,
                            context,
                        )
                        raw_skills: list[str] = optional_raw_skill_ids(
                            raw_row,
                            context,
                        )
                        final_skills: list[str] = require_string_list(
                            row.get("skill_ids_used"),
                            f"{context}.skill_ids_used",
                        )
                        if raw_skills != final_skills:
                            raise DownstreamDataError(
                                "Raw-to-final loaded-skill mismatch: "
                                f"context={context}, raw={raw_skills}, "
                                f"final={final_skills}"
                            )
                        skill_binding = (
                            "raw.skill_ids_used="
                            "final.skill_ids_used"
                        )
                    else:
                        validate_retry_after_import(
                            row,
                            raw_row,
                            model,
                            context,
                            contract,
                        )
                        skill_binding = (
                            "raw.skill_ids_used="
                            "final.expected_skill_ids="
                            "final.actual_injection_state.skill_ids"
                        )
                elif has_structural_source:
                    raise DownstreamDataError(
                        "Final answer references a raw source outside the "
                        f"frozen raw cohort: context={context}"
                    )
                level: ProvenanceLevel
                cohort: ProvenanceCohort
                if (
                    model in contract.early_raw_models
                    and arm in {"routed_always", "routed_gated"}
                ):
                    cohort = "early_raw_k2"
                    level = (
                        "posthoc_structural"
                        if has_structural_source
                        else "formal_retry_after_import"
                    )
                else:
                    cohort = "formal_k2"
                    level = "formal_direct"
                if model == contract.qwen_reference_model and raw_source is not None:
                    if not has_structural_source:
                        raise DownstreamDataError(
                            "Qwen run-history row lacks its structural "
                            f"source reference: context={context}"
                        )
                if has_structural_source:
                    provisional_rows += 1
                provenance_row: JsonObject = {
                    "schema_version": "k2-answer-provenance-row-v1",
                    "model": model,
                    "domain": domain,
                    "arm": arm,
                    "instance_id": instance_id,
                    "provenance_level": level,
                    "cohort": cohort,
                    "formal_source": {
                        "source_sha256": answer_sha256,
                        "source_line_number": parsed["line_number"],
                        "source_line_sha256": parsed["line_sha256"],
                    },
                    "verified_equal_fields": [
                        "instance_id",
                        "dataset",
                        "method",
                        "model",
                        "raw_output",
                    ]
                    if raw_reference is not None
                    else [],
                }
                if raw_reference is not None:
                    provenance_row["raw_source"] = raw_reference
                if skill_binding is not None:
                    provenance_row["skill_identity_binding"] = skill_binding
                provenance_rows.append(provenance_row)
    cohorts, levels = validate_model_counts(
        model,
        provenance_rows,
        provisional_rows,
        contract,
    )
    unique_raw_sources: dict[str, RawSourceData] = {
        source["sha256"]: source for source in model_raw_sources
    }
    unique_run_logs: dict[str, RunLogData] = {
        log["sha256"]: log for log in model_run_logs
    }
    return {
        "schema_version": "k2-answer-provenance-manifest-v1",
        "model": model,
        "raw_sources_verified": bool(unique_raw_sources),
        "run_history_verified": bool(unique_run_logs),
        "runtime_identity": contract.runtime_identities[model],
        "answer_code_bundle_sha256": contract.answer_code_bundles[model],
        "source_files": [
            raw_source_public_entry(source)
            for source in sorted(
                unique_raw_sources.values(),
                key=lambda item: (
                    contract.domains.index(item["domain"]),
                    contract.answer_arms[model].index(
                        cast(ArmName, item["arm"])
                    ),
                ),
            )
        ],
        "run_history_files": [
            run_log_public_entry(log)
            for log in sorted(
                unique_run_logs.values(),
                key=lambda item: (
                    contract.domains.index(item["domain"]),
                    contract.answer_arms[model].index(
                        cast(ArmName, item["arm"])
                    ),
                ),
            )
        ],
        "formal_files": formal_files,
        "rows": provenance_rows,
        "cohort_counts": cohorts,
        "provenance_level_counts": levels,
        "provisional_source_rows": provisional_rows,
        "verification": {
            "raw_to_final_fields": [
                "instance_id",
                "dataset",
                "method",
                "model",
                "raw_output",
                "skill identity",
            ],
            "file_hash_algorithm": "sha256",
            "line_hash_algorithm": "sha256(utf8(line_without_newline))",
        },
    }


def json_bytes(payload: JsonObject) -> bytes:
    """Serialize one deterministic manifest payload."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def merge_counts(
    totals: dict[str, int],
    counts: Mapping[str, int],
) -> None:
    """Add one count mapping into a mutable local accumulator."""

    for name, value in counts.items():
        totals[name] = totals.get(name, 0) + value


def build_fleet_payloads(
    formal_root: Path,
    evidence_root: Path,
    raw_source_paths: Sequence[Path],
    qwen_run_log_paths: Sequence[Path],
    builder_code_files: Sequence[JsonObject],
    builder_code_bundle_sha256: str,
    contract: ProvenanceContract,
) -> dict[str, JsonObject]:
    """Build and globally validate all per-model and fleet manifests."""

    raw_sources: dict[SourceKey, RawSourceData] = load_raw_sources(
        raw_source_paths,
        evidence_root,
        contract,
    )
    run_logs: dict[SourceKey, RunLogData] = load_qwen_run_logs(
        qwen_run_log_paths,
        evidence_root,
        raw_sources,
        contract,
    )
    payloads: dict[str, JsonObject] = {}
    fleet_cohorts: dict[str, int] = {}
    fleet_levels: dict[str, int] = {}
    fleet_provisional_rows: int = 0
    total_rows: int = 0
    model_entries: list[JsonObject] = []
    for model in contract.models:
        payload: JsonObject = build_model_manifest(
            model,
            formal_root,
            evidence_root,
            raw_sources,
            run_logs,
            contract,
        )
        filename: str = f"answer-provenance.{model}.json"
        payload["builder_code_files"] = list(builder_code_files)
        payload["builder_code_bundle_sha256"] = (
            builder_code_bundle_sha256
        )
        payload_bytes: bytes = json_bytes(payload)
        payload_sha256: str = sha256_text(
            payload_bytes.decode("utf-8")
        )
        payloads[filename] = payload
        row_count: int = len(
            cast(list[JsonValue], payload["rows"])
        )
        total_rows += row_count
        merge_counts(
            fleet_cohorts,
            cast(dict[str, int], payload["cohort_counts"]),
        )
        merge_counts(
            fleet_levels,
            cast(dict[str, int], payload["provenance_level_counts"]),
        )
        fleet_provisional_rows += cast(
            int,
            payload["provisional_source_rows"],
        )
        model_entries.append(
            {
                "model": model,
                "path": filename,
                "sha256": payload_sha256,
                "rows": row_count,
            }
        )
    expected_total_rows: int = sum(
        expected_model_rows(model, contract)
        for model in contract.models
    )
    if total_rows != expected_total_rows:
        raise DownstreamDataError(
            f"Fleet provenance row count mismatch: "
            f"expected={expected_total_rows}, actual={total_rows}"
        )
    expected_cohorts: dict[str, int] = dict(
        contract.expected_fleet_cohorts
    )
    expected_levels: dict[str, int] = dict(
        contract.expected_fleet_levels
    )
    if fleet_cohorts != expected_cohorts:
        raise DownstreamDataError(
            f"Fleet provenance cohort mismatch: "
            f"expected={expected_cohorts}, actual={fleet_cohorts}"
        )
    if fleet_levels != expected_levels:
        raise DownstreamDataError(
            f"Fleet provenance level mismatch: "
            f"expected={expected_levels}, actual={fleet_levels}"
        )
    if (
        fleet_provisional_rows
        != contract.expected_fleet_provisional_rows
    ):
        raise DownstreamDataError(
            "Fleet provisional-source count mismatch: "
            f"expected={contract.expected_fleet_provisional_rows}, "
            f"actual={fleet_provisional_rows}"
        )
    fleet_payload: JsonObject = {
        "schema_version": "k2-answer-provenance-fleet-v1",
        "models": model_entries,
        "rows": total_rows,
        "raw_source_files": len(raw_sources),
        "qwen_run_history_files": len(run_logs),
        "cohort_counts": fleet_cohorts,
        "provenance_level_counts": fleet_levels,
        "provisional_source_rows": fleet_provisional_rows,
        "builder_code_files": list(builder_code_files),
        "builder_code_bundle_sha256": builder_code_bundle_sha256,
        "accepted_public_pack": False,
    }
    payloads["answer-provenance.fleet.json"] = fleet_payload
    return payloads
