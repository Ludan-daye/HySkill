"""Pure calibration and payload helpers for the runtime-matched Gate audit."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal, TypeAlias, TypedDict, cast

from hyskill.runtime_matched_bm25 import git_revision, require_module_under
from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    FROZEN_SRAGENTS_REVISION,
    JsonLike,
    JsonValue,
    answer_payload_hash,
    require_sha256,
)


GateDecision = Literal["blocked_s1", "skipped_s2", "kept"]
GateArm = Literal["routed_gated", "fixed_gated"]
GateTaskKey: TypeAlias = tuple[str, str, GateArm]
JsonObject: TypeAlias = dict[str, JsonValue]

ANSWER_PAYLOAD_SCHEMA_VERSION: str = "runtime-matched-gate-answer-payload-v1"
GATE_DIFF_ROW_SCHEMA_VERSION: str = "runtime-matched-gate-diff-row-v1"
GATE_DECISION_SCHEMA_VERSION: str = "runtime-matched-gate-decision-v1"
GATE_AUDIT_SCHEMA_VERSION: str = "runtime-matched-gate-audit-v1"
GATE_RERUN_MANIFEST_SCHEMA_VERSION: str = (
    "runtime-matched-gate-rerun-manifest-v1"
)
GATE_COMBINED_RERUN_SCHEMA_VERSION: str = (
    "runtime-matched-gate-combined-rerun-manifest-v1"
)
GATE_RERUN_ANSWER_SCHEMA_VERSION: str = "runtime-matched-gate-answer-v1"
GATE_MERGE_REPORT_SCHEMA_VERSION: str = "runtime-matched-gate-merge-report-v1"
ANSWER_TEMPERATURE: float = 0.7
ANSWER_MAX_TOKENS: int = 2048
ANSWER_THINKING: bool = False
P_MIN: float = 0.9
RULE_DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
RULE_DOMAIN_COUNTS: dict[str, int] = {
    "theoremqa": 747,
    "logicbench": 760,
    "medcalcbench": 1100,
    "champ": 223,
}
FIXED_GATE_MODEL: str = "qwen3.5-4b-reference"
FROZEN_CORPUS_SHA256: str = (
    "16ee509ae5bea8c2e17167dffecd89100a7d8dfa31256c3742426758c7169b5e"
)
FROZEN_INSTANCE_SHA256: dict[str, str] = {
    "theoremqa": (
        "c969a7291e23361ba9f377e464be76093804deb628b964fb846c6eff6b28deeb"
    ),
    "logicbench": (
        "af5055caac041ea08cee47622f5d922b47b2ba0a8e3a60b87349599eeff1bdfe"
    ),
    "medcalcbench": (
        "814f6b082f56cf89dd9c50be7ab87b5bcb0e41633138951e42b38d6923c3244e"
    ),
    "champ": (
        "d61346716cede953afb352e739e170b96d2bfb98824edd91b8783dc3526c7cec"
    ),
}


def expected_gate_task_keys() -> tuple[GateTaskKey, ...]:
    """Return the exact 28 routed plus four fixed Gate task identities."""

    routed: tuple[GateTaskKey, ...] = tuple(
        (model, domain, "routed_gated")
        for model in sorted(FROZEN_K2_RUNTIME_REFERENCES)
        for domain in RULE_DOMAINS
    )
    fixed: tuple[GateTaskKey, ...] = tuple(
        (FIXED_GATE_MODEL, domain, "fixed_gated")
        for domain in RULE_DOMAINS
    )
    return routed + fixed


def expected_gate_row_count() -> int:
    """Return the exact logical row count across all 32 Gate tasks."""

    return sum(
        RULE_DOMAIN_COUNTS[domain]
        for _, domain, _ in expected_gate_task_keys()
    )


class RuntimeMatchedGateError(ValueError):
    """Raised when Gate evidence violates the frozen K=2 protocol."""


class GateSignal(TypedDict):
    """One frozen K=2 Gate signal."""

    instance_id: str
    top1: str
    S1: float
    S2: float
    rel_truth_wrong: bool


class GateThresholds(TypedDict):
    """One pair of S1/S2 thresholds."""

    tau1: float | None
    tau2: float | None


class RenderedAnswerPayload(TypedDict):
    """Model-visible direct-answer request components and their hash."""

    messages: list[JsonObject]
    loaded_skills: list[JsonObject]
    tools: list[JsonObject]
    generation: JsonObject
    answer_payload_hash: str


class NativeGateRuntime(TypedDict):
    """Pinned native functions used to render direct-answer payloads."""

    build_prompt: "BuildPrompt"
    get_extra_body: "GetExtraBody"
    revision: str
    source_root: str


class GateAuditResult(TypedDict):
    """Pure output for one model-domain-arm Gate audit."""

    new_thresholds: GateThresholds
    diff_rows: list[JsonObject]
    decision_rows: list[JsonObject]


BuildPrompt = Callable[[dict[str, object], list[str] | None], tuple[str, str]]
GetExtraBody = Callable[[str, bool], dict[str, object] | None]


def load_native_gate_runtime(
    checkout: Path,
    expected_revision: str,
) -> NativeGateRuntime:
    """Load prompt rendering only from the explicitly pinned checkout."""

    checkout_path: Path = checkout.resolve()
    observed_revision: str = git_revision(checkout_path)
    if observed_revision != expected_revision:
        raise RuntimeError(
            "SR-Agents revision mismatch for Gate audit: "
            f"path={checkout_path}, expected={expected_revision}, "
            f"observed={observed_revision}"
        )
    if observed_revision != FROZEN_SRAGENTS_REVISION:
        raise RuntimeError(
            "Gate audit revision differs from the frozen K=2 source: "
            f"expected={FROZEN_SRAGENTS_REVISION}, "
            f"observed={observed_revision}"
        )
    source_root: Path = checkout_path / "src"
    if not source_root.is_dir():
        raise NotADirectoryError(
            f"SR-Agents source directory does not exist: path={source_root}"
        )
    source_root_text: str = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    prompts_module: ModuleType = importlib.import_module("sragents.prompts")
    llm_module: ModuleType = importlib.import_module("sragents.llm")
    for module in (prompts_module, llm_module):
        require_module_under(module, source_root)
    return {
        "build_prompt": cast(
            BuildPrompt,
            getattr(prompts_module, "build_prompt"),
        ),
        "get_extra_body": cast(
            GetExtraBody,
            getattr(llm_module, "get_extra_body"),
        ),
        "revision": observed_revision,
        "source_root": source_root_text,
    }


def require_object(
    value: JsonValue | None,
    context: str,
) -> JsonObject:
    """Return one JSON object or raise with source context."""

    if not isinstance(value, dict):
        raise RuntimeMatchedGateError(
            "Expected JSON object: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_list(
    value: JsonValue | None,
    context: str,
) -> list[JsonValue]:
    """Return one JSON list or raise with source context."""

    if not isinstance(value, list):
        raise RuntimeMatchedGateError(
            "Expected JSON list: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_string(
    value: JsonValue | None,
    context: str,
) -> str:
    """Return one non-empty JSON string."""

    if not isinstance(value, str) or not value:
        raise RuntimeMatchedGateError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_string_list(
    value: JsonValue | None,
    context: str,
) -> list[str]:
    """Return one duplicate-free list of non-empty strings."""

    values: list[str] = [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(require_list(value, context))
    ]
    if len(values) != len(set(values)):
        raise RuntimeMatchedGateError(
            f"String list contains duplicates: context={context}, values={values}"
        )
    return values


def require_boolean(
    value: JsonValue | None,
    context: str,
) -> bool:
    """Return one JSON Boolean."""

    if not isinstance(value, bool):
        raise RuntimeMatchedGateError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_number(
    value: JsonValue | None,
    context: str,
) -> float:
    """Return one finite JSON number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeMatchedGateError(
            f"Expected finite number: context={context}, value={value!r}"
        )
    return float(value)


def require_optional_number(
    value: JsonValue | None,
    context: str,
) -> float | None:
    """Return one finite number or an explicit null."""

    if value is None:
        return None
    return require_number(value, context)


def require_gate_arm(value: str) -> GateArm:
    """Return one supported K=2 Gate arm."""

    if value not in ("routed_gated", "fixed_gated"):
        raise RuntimeMatchedGateError(
            "Unknown Gate arm: "
            f"value={value!r}, allowed=['fixed_gated', 'routed_gated']"
        )
    return cast(GateArm, value)


def require_probability(value: float, field_name: str) -> float:
    """Return one closed-unit-interval probability or raise."""

    if value < 0.0 or value > 1.0:
        raise RuntimeMatchedGateError(
            f"Probability is outside [0, 1]: field={field_name}, value={value}"
        )
    return value


def pick_tau(
    values: Sequence[float],
    is_positive: Sequence[bool],
    p_min: float,
) -> float | None:
    """Reproduce the frozen Gate threshold selection exactly."""

    require_probability(p_min, "p_min")
    if len(values) != len(is_positive):
        raise RuntimeMatchedGateError(
            "Threshold inputs have different lengths: "
            f"values={len(values)}, labels={len(is_positive)}"
        )
    if not values:
        raise RuntimeMatchedGateError("Threshold calibration support is empty")
    pairs: list[tuple[float, bool]] = sorted(zip(values, is_positive))
    best: float | None = None
    positives: int = 0
    for index, (value, positive) in enumerate(pairs, start=1):
        if positive:
            positives += 1
        if positives / index >= p_min:
            best = value
    return best


def calibrate_thresholds(
    signals: Sequence[GateSignal],
    validation_ids: frozenset[str],
    bare_correct: Mapping[str, bool],
    p_min: float,
) -> GateThresholds:
    """Calibrate S1 from relevance and S2 from fresh Bare correctness."""

    if not validation_ids:
        raise RuntimeMatchedGateError("Gate validation ID set is empty")
    signal_by_id: dict[str, GateSignal] = {}
    for signal in signals:
        instance_id: str = signal["instance_id"]
        if instance_id in signal_by_id:
            raise RuntimeMatchedGateError(
                f"Duplicate Gate signal: instance_id={instance_id}"
            )
        signal_by_id[instance_id] = signal
    missing_signals: list[str] = sorted(validation_ids - signal_by_id.keys())
    if missing_signals:
        raise RuntimeMatchedGateError(
            "Validation IDs are missing Gate signals: "
            f"sample={missing_signals[:20]}"
        )
    missing_bare: list[str] = sorted(validation_ids - bare_correct.keys())
    if missing_bare:
        raise RuntimeMatchedGateError(
            "Validation IDs are missing fresh Bare labels: "
            f"sample={missing_bare[:20]}"
        )
    validation_signals: list[GateSignal] = [
        signal_by_id[instance_id] for instance_id in sorted(validation_ids)
    ]
    return {
        "tau1": pick_tau(
            [signal["S1"] for signal in validation_signals],
            [signal["rel_truth_wrong"] for signal in validation_signals],
            p_min,
        ),
        "tau2": pick_tau(
            [signal["S2"] for signal in validation_signals],
            [bare_correct[signal["instance_id"]] for signal in validation_signals],
            p_min,
        ),
    }


def gate_decision(
    signal: GateSignal,
    tau1: float | None,
    tau2: float | None,
) -> GateDecision:
    """Apply the frozen ordered S1-then-S2 Gate rule."""

    if tau1 is not None and signal["S1"] < tau1:
        return "blocked_s1"
    if tau2 is not None and signal["S2"] < tau2:
        return "skipped_s2"
    return "kept"


def expected_skill_ids(
    signal: GateSignal,
    decision: GateDecision,
) -> tuple[str, ...]:
    """Return the exact direct-engine skill injection for one decision."""

    if decision == "kept":
        return (signal["top1"],)
    return ()


def render_direct_answer_payload(
    instance: Mapping[str, JsonLike],
    loaded_skills: Sequence[Mapping[str, JsonLike]],
    served_model: str,
    build_prompt: BuildPrompt,
    get_extra_body: GetExtraBody,
) -> RenderedAnswerPayload:
    """Render the native direct-engine request without calling a model."""

    instance_object: dict[str, object] = dict(instance)
    skill_objects: list[JsonObject] = [dict(skill) for skill in loaded_skills]
    skill_texts: list[str] = []
    tools: list[JsonObject] = []
    for index, skill in enumerate(skill_objects):
        content: JsonValue | None = skill.get("content")
        if not isinstance(content, str):
            raise RuntimeMatchedGateError(
                "Loaded skill has invalid content: "
                f"index={index}, value_type={type(content).__name__}"
            )
        skill_texts.append(content)
        raw_tools: JsonValue | None = skill.get("tools")
        if raw_tools is None:
            continue
        if not isinstance(raw_tools, list):
            raise RuntimeMatchedGateError(
                "Loaded skill tools must be a list: "
                f"index={index}, value_type={type(raw_tools).__name__}"
            )
        for tool_index, raw_tool in enumerate(raw_tools):
            if not isinstance(raw_tool, dict):
                raise RuntimeMatchedGateError(
                    "Loaded skill tool must be an object: "
                    f"skill_index={index}, tool_index={tool_index}, "
                    f"value_type={type(raw_tool).__name__}"
                )
            tools.append(dict(raw_tool))
    system, user = build_prompt(
        instance_object,
        skill_texts if skill_texts else None,
    )
    if not isinstance(system, str) or not isinstance(user, str):
        raise RuntimeMatchedGateError(
            "Native prompt renderer must return two strings: "
            f"system_type={type(system).__name__}, "
            f"user_type={type(user).__name__}"
        )
    messages: list[JsonObject] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    extra_body: dict[str, object] | None = get_extra_body(
        served_model,
        ANSWER_THINKING,
    )
    generation: JsonObject = {
        "temperature": ANSWER_TEMPERATURE,
        "max_tokens": ANSWER_MAX_TOKENS,
        "thinking": ANSWER_THINKING,
        "extra_body": extra_body,
    }
    payload_hash: str = answer_payload_hash(
        ANSWER_PAYLOAD_SCHEMA_VERSION,
        instance,
        messages,
        skill_objects,
        tools,
        generation,
    )
    return {
        "messages": messages,
        "loaded_skills": skill_objects,
        "tools": tools,
        "generation": generation,
        "answer_payload_hash": payload_hash,
    }


def index_instances(
    values: Sequence[JsonValue],
    domain: str,
) -> dict[str, JsonObject]:
    """Validate and index one complete rule-domain instance set."""

    output: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        instance: JsonObject = require_object(
            value,
            f"instances[{index}]",
        )
        instance_id: str = require_string(
            instance.get("instance_id"),
            f"instances[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"Duplicate instance ID: instance_id={instance_id}"
            )
        observed_domain: str = require_string(
            instance.get("dataset"),
            f"instances[{index}].dataset",
        )
        if observed_domain != domain:
            raise RuntimeMatchedGateError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"observed={observed_domain}"
            )
        output[instance_id] = instance
    if not output:
        raise RuntimeMatchedGateError("Gate instance support is empty")
    return output


def index_corpus(
    values: Sequence[JsonValue],
) -> dict[str, JsonObject]:
    """Validate and index the frozen skill corpus."""

    output: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        skill: JsonObject = require_object(value, f"corpus[{index}]")
        skill_id: str = require_string(
            skill.get("skill_id"),
            f"corpus[{index}].skill_id",
        )
        if skill_id in output:
            raise RuntimeMatchedGateError(
                f"Duplicate corpus skill ID: skill_id={skill_id}"
            )
        content: JsonValue | None = skill.get("content")
        if not isinstance(content, str):
            raise RuntimeMatchedGateError(
                "Corpus skill content must be a string: "
                f"skill_id={skill_id}, value_type={type(content).__name__}"
            )
        raw_tools: JsonValue | None = skill.get("tools")
        if raw_tools is not None:
            require_list(raw_tools, f"corpus:{skill_id}.tools")
        output[skill_id] = skill
    if not output:
        raise RuntimeMatchedGateError("Gate corpus is empty")
    return output


def index_signals(
    values: Sequence[JsonValue],
) -> dict[str, GateSignal]:
    """Validate and index one frozen Gate signal set."""

    output: dict[str, GateSignal] = {}
    for index, value in enumerate(values):
        raw_signal: JsonObject = require_object(
            value,
            f"signals[{index}]",
        )
        instance_id: str = require_string(
            raw_signal.get("instance_id"),
            f"signals[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"Duplicate Gate signal: instance_id={instance_id}"
            )
        s1: float = require_number(
            raw_signal.get("S1"),
            f"signals[{index}].S1",
        )
        if s1 < -1.0 or s1 > 1.0:
            raise RuntimeMatchedGateError(
                "S1 cosine is outside [-1, 1]: "
                f"instance_id={instance_id}, value={s1}"
            )
        s2: float = require_probability(
            require_number(
                raw_signal.get("S2"),
                f"signals[{index}].S2",
            ),
            f"signals[{index}].S2",
        )
        output[instance_id] = {
            "instance_id": instance_id,
            "top1": require_string(
                raw_signal.get("top1"),
                f"signals[{index}].top1",
            ),
            "S1": s1,
            "S2": s2,
            "rel_truth_wrong": require_boolean(
                raw_signal.get("rel_truth_wrong"),
                f"signals[{index}].rel_truth_wrong",
            ),
        }
    if not output:
        raise RuntimeMatchedGateError("Gate signal support is empty")
    return output


def index_rows(
    values: Sequence[JsonValue],
    context: str,
) -> dict[str, JsonObject]:
    """Index arbitrary JSON rows by one required unique instance ID."""

    output: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        row: JsonObject = require_object(value, f"{context}[{index}]")
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"{context} contains duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def exact_coverage(
    expected_ids: set[str],
    observed_ids: set[str],
    context: str,
) -> None:
    """Require exact instance coverage."""

    missing: list[str] = sorted(expected_ids - observed_ids)
    unexpected: list[str] = sorted(observed_ids - expected_ids)
    if missing or unexpected:
        raise RuntimeMatchedGateError(
            f"{context} coverage mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def _loaded_skills(
    skill_ids: Sequence[str],
    corpus: Mapping[str, JsonObject],
    instance_id: str,
) -> list[JsonObject]:
    output: list[JsonObject] = []
    for skill_id in skill_ids:
        skill: JsonObject | None = corpus.get(skill_id)
        if skill is None:
            raise RuntimeMatchedGateError(
                "Gate decision refers to a missing corpus skill: "
                f"instance_id={instance_id}, skill_id={skill_id}"
            )
        output.append(skill)
    return output


def _validate_old_answer(
    answer: Mapping[str, JsonValue],
    instance_id: str,
    expected_skill_ids_value: tuple[str, ...],
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> tuple[str, str]:
    if answer.get("schema_version") != "k2-answer-record-v1":
        raise RuntimeMatchedGateError(
            "Old Gate answer has an unexpected schema: "
            f"instance_id={instance_id}, "
            f"schema={answer.get('schema_version')!r}"
        )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("instance_id", instance_id),
        ("dataset", domain),
        ("method", arm),
        ("served_model", served_model),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},"
        f"observed={answer.get(field_name)!r}"
        for field_name, expected in expected_fields
        if answer.get(field_name) != expected
    ]
    runtime_identity: JsonObject = require_object(
        answer.get("runtime_identity"),
        f"old-answer:{instance_id}.runtime_identity",
    )
    if runtime_identity.get("model") != result_tag:
        mismatches.append(
            "runtime_identity.model:"
            f"expected={result_tag!r},"
            f"observed={runtime_identity.get('model')!r}"
        )
    answer_expected: list[str] = require_string_list(
        answer.get("expected_skill_ids"),
        f"old-answer:{instance_id}.expected_skill_ids",
    )
    if answer_expected != list(expected_skill_ids_value):
        mismatches.append(
            "expected_skill_ids:"
            f"expected={list(expected_skill_ids_value)!r},"
            f"observed={answer_expected!r}"
        )
    if mismatches:
        raise RuntimeMatchedGateError(
            "Old Gate answer identity or injection mismatch: "
            f"instance_id={instance_id}, mismatches={mismatches}"
        )
    request_hash: str = require_sha256(
        answer.get("request_hash"),
        f"old-answer:{instance_id}.request_hash",
    )
    failure_category: str = require_string(
        answer.get("failure_category"),
        f"old-answer:{instance_id}.failure_category",
    )
    if failure_category not in ("success", "method_failure"):
        raise RuntimeMatchedGateError(
            "Old Gate answer contains an unresolved outcome: "
            f"instance_id={instance_id}, category={failure_category}"
        )
    raw_output: JsonValue | None = answer.get("raw_output")
    if not isinstance(raw_output, str):
        raise RuntimeMatchedGateError(
            "Old Gate answer raw_output must be a string: "
            f"instance_id={instance_id}"
        )
    skill_ids_used: list[str] = require_string_list(
        answer.get("skill_ids_used"),
        f"old-answer:{instance_id}.skill_ids_used",
    )
    if failure_category == "success":
        if not raw_output.strip():
            raise RuntimeMatchedGateError(
                f"Old successful Gate answer is empty: instance_id={instance_id}"
            )
        if skill_ids_used != list(expected_skill_ids_value):
            raise RuntimeMatchedGateError(
                "Old successful Gate answer used unexpected skills: "
                f"instance_id={instance_id}, "
                f"expected={list(expected_skill_ids_value)}, "
                f"observed={skill_ids_used}"
            )
    elif raw_output:
        raise RuntimeMatchedGateError(
            "Old Gate method failure carries answer text: "
            f"instance_id={instance_id}"
        )
    injection_state: JsonObject = require_object(
        answer.get("actual_injection_state"),
        f"old-answer:{instance_id}.actual_injection_state",
    )
    submitted_skill_ids: list[str] = require_string_list(
        injection_state.get("skill_ids"),
        f"old-answer:{instance_id}.actual_injection_state.skill_ids",
    )
    if submitted_skill_ids != list(expected_skill_ids_value):
        raise RuntimeMatchedGateError(
            "Old Gate request submitted unexpected skills: "
            f"instance_id={instance_id}, "
            f"expected={list(expected_skill_ids_value)}, "
            f"observed={submitted_skill_ids}"
        )
    return request_hash, failure_category


def audit_gate_task(
    instances: Mapping[str, JsonObject],
    corpus: Mapping[str, JsonObject],
    signals: Mapping[str, GateSignal],
    validation_ids: frozenset[str],
    fresh_bare_correct: Mapping[str, bool],
    old_thresholds: GateThresholds,
    old_gated_retrieved_ids: Mapping[str, Sequence[str]],
    old_answers: Mapping[str, JsonObject],
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    p_min: float,
    runtime: NativeGateRuntime,
) -> GateAuditResult:
    """Recalibrate one Gate and compare every model-visible answer payload."""

    expected_ids: set[str] = set(instances)
    for context, observed in (
        ("signals", set(signals)),
        ("fresh-bare", set(fresh_bare_correct)),
        ("old-gated", set(old_gated_retrieved_ids)),
        ("old-answers", set(old_answers)),
    ):
        exact_coverage(expected_ids, observed, context)
    if not validation_ids.issubset(expected_ids):
        raise RuntimeMatchedGateError(
            "Gate validation IDs are outside the instance support: "
            f"sample={sorted(validation_ids - expected_ids)[:20]}"
        )
    new_thresholds: GateThresholds = calibrate_thresholds(
        list(signals.values()),
        validation_ids,
        fresh_bare_correct,
        p_min,
    )
    if new_thresholds["tau1"] != old_thresholds["tau1"]:
        raise RuntimeMatchedGateError(
            "Fresh Bare labels unexpectedly changed tau1. The frozen S1 "
            "signals, relevance labels, or validation IDs are inconsistent: "
            f"old_tau1={old_thresholds['tau1']}, "
            f"new_tau1={new_thresholds['tau1']}"
        )
    diff_rows: list[JsonObject] = []
    decision_rows: list[JsonObject] = []
    for instance_id in sorted(expected_ids):
        instance: JsonObject = instances[instance_id]
        signal: GateSignal = signals[instance_id]
        old_decision: GateDecision = gate_decision(
            signal,
            old_thresholds["tau1"],
            old_thresholds["tau2"],
        )
        new_decision: GateDecision = gate_decision(
            signal,
            new_thresholds["tau1"],
            new_thresholds["tau2"],
        )
        old_expected: tuple[str, ...] = expected_skill_ids(
            signal,
            old_decision,
        )
        new_expected: tuple[str, ...] = expected_skill_ids(
            signal,
            new_decision,
        )
        old_retrieved_ids: list[str] = list(
            old_gated_retrieved_ids[instance_id]
        )
        if old_decision == "kept":
            if not old_retrieved_ids or old_retrieved_ids[0] != signal["top1"]:
                raise RuntimeMatchedGateError(
                    "Old gated retrieval does not preserve the signal top-1: "
                    f"instance_id={instance_id}, signal_top1={signal['top1']}, "
                    f"retrieved_sample={old_retrieved_ids[:3]}"
                )
        elif old_retrieved_ids:
            raise RuntimeMatchedGateError(
                "Old gated retrieval retained candidates for a blocked row: "
                f"instance_id={instance_id}, decision={old_decision}, "
                f"retrieved_sample={old_retrieved_ids[:3]}"
            )
        old_request_hash, old_failure_category = _validate_old_answer(
            old_answers[instance_id],
            instance_id,
            old_expected,
            result_tag,
            served_model,
            domain,
            arm,
        )
        old_payload: RenderedAnswerPayload = render_direct_answer_payload(
            instance,
            _loaded_skills(old_expected, corpus, instance_id),
            served_model,
            runtime["build_prompt"],
            runtime["get_extra_body"],
        )
        new_payload: RenderedAnswerPayload = render_direct_answer_payload(
            instance,
            _loaded_skills(new_expected, corpus, instance_id),
            served_model,
            runtime["build_prompt"],
            runtime["get_extra_body"],
        )
        decision_changed: bool = old_decision != new_decision
        injection_changed: bool = old_expected != new_expected
        payload_changed: bool = (
            old_payload["answer_payload_hash"]
            != new_payload["answer_payload_hash"]
        )
        if injection_changed != payload_changed:
            raise RuntimeMatchedGateError(
                "Gate injection and model-visible payload change disagree: "
                f"instance_id={instance_id}, "
                f"injection_changed={injection_changed}, "
                f"payload_changed={payload_changed}"
            )
        is_validation: bool = instance_id in validation_ids
        row: JsonObject = {
            "schema_version": GATE_DIFF_ROW_SCHEMA_VERSION,
            "model": result_tag,
            "served_model": served_model,
            "domain": domain,
            "arm": arm,
            "instance_id": instance_id,
            "is_validation": is_validation,
            "S1": signal["S1"],
            "S2": signal["S2"],
            "top1_skill_id": signal["top1"],
            "old_tau1": old_thresholds["tau1"],
            "old_tau2": old_thresholds["tau2"],
            "new_tau1": new_thresholds["tau1"],
            "new_tau2": new_thresholds["tau2"],
            "old_decision": old_decision,
            "new_decision": new_decision,
            "old_expected_skill_ids": list(old_expected),
            "new_expected_skill_ids": list(new_expected),
            "decision_changed": decision_changed,
            "injection_changed": injection_changed,
            "old_answer_payload_hash": old_payload["answer_payload_hash"],
            "new_answer_payload_hash": new_payload["answer_payload_hash"],
            "payload_changed": payload_changed,
            "rerun_required": payload_changed,
            "preserve_old_row": not payload_changed,
            "old_request_hash": old_request_hash,
            "old_failure_category": old_failure_category,
        }
        diff_rows.append(row)
        decision_rows.append(
            {
                "schema_version": GATE_DECISION_SCHEMA_VERSION,
                "model": result_tag,
                "served_model": served_model,
                "domain": domain,
                "arm": arm,
                "instance_id": instance_id,
                "is_validation": is_validation,
                "decision": new_decision,
                "expected_skill_ids": list(new_expected),
                "answer_payload_hash": new_payload["answer_payload_hash"],
                "rerun_required": payload_changed,
            }
        )
    return {
        "new_thresholds": new_thresholds,
        "diff_rows": diff_rows,
        "decision_rows": decision_rows,
    }
