#!/usr/bin/env python3
"""Export one model's validated E7 hy_rerank evidence into a path-free public pack.

Reads ``<result-root>/<TAG>/{eval,decisions,audits,runtime,logs}`` across the
four rule-scored domains, verifies every domain audit has ``valid=true`` and
matching file SHA-256 digests, builds canonical path-free public records, scans
every generated file for sensitive tokens, and writes:
  - ``answer_per_instance.jsonl.gz`` (2,830 rows)
  - ``decision_per_instance.jsonl.gz`` (2,830 rows)
  - ``runtime_jobs.jsonl.gz`` (8 rows)
  - ``usage_jobs.jsonl.gz`` (8 rows)
  - ``validation.json``
  - ``README.md``
  - ``manifest.json``
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_execution import canonical_json, sha256_file
from hyskill.runtime_matched_public import (
    JsonObject,
    JsonValue,
    deterministic_jsonl_gzip,
    public_answer_row,
    public_decision_row,
    public_runtime_job,
    public_usage_job,
)
from hyskill.runtime_matched_rerank import HY_RERANK_ARM, require_supported_model
from scripts.export_runtime_matched_baselines import (
    SENSITIVE_TOKENS,
    RuntimeMatchedExportError,
    load_json,
    load_jsonl,
    output_record,
    scan_pack,
    strip_paths,
    write_json,
    write_jsonl_gzip,
)
from scripts.summarize_runtime_matched_usage import (
    UsageLogSpec,
    load_usage_events,
    summarize_events,
)
from scripts.validate_runtime_matched_baselines import (
    DOMAINS,
    NATIVE_AUDIT_SCHEMA_VERSION,
    require_list,
    require_object,
    require_string,
    served_model_for,
    verify_evidence_file,
)

HY_RERANK_PACK_SCHEMA_VERSION: str = "ijcai27-hy-rerank-public-model-pack-v1"
HY_RERANK_VALIDATION_SCHEMA_VERSION: str = "ijcai27-hy-rerank-public-validation-v1"
DOMAIN_ORDER: dict[str, int] = {
    domain: idx for idx, (domain, _total, _heldout) in enumerate(DOMAINS)
}
STAGE_ORDER: dict[str, int] = {"decision": 0, "answer": 1}


def _answer_sort_key(row: Mapping[str, JsonValue]) -> tuple[int, str]:
    domain = require_string(row.get("domain"), "answer.domain")
    return (DOMAIN_ORDER[domain], require_string(row.get("instance_id"), "answer.instance_id"))


def _runtime_sort_key(row: Mapping[str, JsonValue]) -> tuple[int, int, str]:
    job = require_object(row.get("job"), "runtime.job")
    domain = require_string(job.get("domain"), "runtime.job.domain")
    job_id = require_string(job.get("job_id"), "runtime.job.job_id")
    stage = "decision" if "-decision" in job_id else "answer"
    return (DOMAIN_ORDER[domain], STAGE_ORDER[stage], job_id)


def _usage_sort_key(row: Mapping[str, JsonValue]) -> tuple[int, int]:
    domain = require_string(row.get("domain"), "usage.domain")
    stage = require_string(row.get("stage"), "usage.stage")
    return (DOMAIN_ORDER[domain], STAGE_ORDER[stage])


def _hy_rerank_readme(model: str, answer_rows: int, decision_rows: int) -> str:
    return "\n".join(
        (
            f"# IJCAI-27 Revision E7 `hy_rerank` Public Pack: {model}",
            "",
            "This pack contains runtime-matched `hy_rerank` decisions and answers",
            "over the frozen K=2 routed HySkill top-50 candidate pools across the",
            "four rule-scored domains (`theoremqa`, `logicbench`, `medcalcbench`, `champ`).",
            "",
            "## Protocol & Coverage",
            "",
            f"- Public answer rows: `{answer_rows}` (`2,265` held-out test + `565` validation).",
            f"- Public decision rows: `{decision_rows}` (`50` ordered routed candidates per instance).",
            "- Prompt & parser: Frozen SR-Agents `277fd8d` listwise reranker (`temperature=0.0`, `max_tokens=1024`, up to 3 parse attempts, omitted candidates appended in input order).",
            "- Answer generation: Direct engine (`temperature=0.7`, `max_tokens=2048`, `thinking=false`) matched to the K=2 checkpoint, tokenizer, chat template, vLLM version, BF16, and 8,192 context contract.",
            "- Accounting: `method_failure` rows remain in the denominator and score `0`. Raw model text, gold answers, endpoints, host paths, GPU UUIDs, and credentials are stripped.",
            "",
        )
    )


def export_hy_rerank_model_pack(
    result_root: Path,
    model: str,
    output_dir: Path,
) -> JsonObject:
    """Validate and export one model's ``hy_rerank`` public pack."""

    require_supported_model(model)
    served_model = served_model_for(model)
    model_root = (result_root / model).resolve()
    target_dir = output_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"target public pack directory is not empty: path={target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    answer_rows: list[JsonObject] = []
    decision_rows: list[JsonObject] = []
    runtime_rows: list[JsonObject] = []
    usage_rows: list[JsonObject] = []
    domain_audits: dict[str, JsonObject] = {}
    audit_shas: dict[str, str] = {}

    for domain, expected_count, expected_heldout in DOMAINS:
        audit_path = model_root / "audits" / f"{domain}-{HY_RERANK_ARM}.audit.json"
        audit = load_json(audit_path, f"audit:{model}:{domain}")
        if (
            audit.get("schema_version") != NATIVE_AUDIT_SCHEMA_VERSION
            or audit.get("valid") is not True
            or audit.get("model") != model
            or audit.get("served_model") != served_model
            or audit.get("domain") != domain
            or audit.get("arm") != HY_RERANK_ARM
            or audit.get("expected_rows") != expected_count
            or audit.get("observed_decisions") != expected_count
            or audit.get("observed_answers") != expected_count
            or audit.get("unresolved") != 0
        ):
            raise RuntimeMatchedExportError(
                f"hy_rerank domain audit failed validation: path={audit_path}"
            )
        audit_shas[domain] = sha256_file(audit_path)
        artifacts = require_object(audit.get("artifacts"), f"audit:{model}:{domain}.artifacts")

        dec_path = verify_evidence_file(
            model_root,
            "decisions",
            require_object(artifacts.get("decisions"), "artifacts.decisions"),
            f"audit:{model}:{domain}.decisions",
        )
        eval_path = verify_evidence_file(
            model_root,
            "eval",
            require_object(artifacts.get("evaluation"), "artifacts.evaluation"),
            f"audit:{model}:{domain}.evaluation",
        )
        dec_manifest_path = verify_evidence_file(
            model_root,
            "runtime",
            require_object(artifacts.get("decision_manifest"), "artifacts.decision_manifest"),
            f"audit:{model}:{domain}.decision_manifest",
        )
        ans_manifest_path = verify_evidence_file(
            model_root,
            "runtime",
            require_object(artifacts.get("answer_manifest"), "artifacts.answer_manifest"),
            f"audit:{model}:{domain}.answer_manifest",
        )
        verify_evidence_file(
            model_root,
            "answers",
            require_object(artifacts.get("answers"), "artifacts.answers"),
            f"audit:{model}:{domain}.answers",
        )

        usage_obj = require_object(audit.get("usage"), f"audit:{model}:{domain}.usage")
        for stage in ("decision", "answer"):
            stage_usage_audit = require_object(
                usage_obj.get(stage),
                f"audit:{model}:{domain}.usage.{stage}",
            )
            log_path = verify_evidence_file(
                model_root,
                "logs",
                stage_usage_audit,
                f"audit:{model}:{domain}.usage.{stage}",
            )
            spec: UsageLogSpec = {
                "model": model,
                "domain": domain,
                "arm": HY_RERANK_ARM,
                "stage": cast(str, stage),  # type: ignore[typeddict-item]
                "path": log_path,
            }
            events = load_usage_events(spec)
            job_summary: JsonObject = {
                "model": model,
                "domain": domain,
                "arm": HY_RERANK_ARM,
                "stage": cast(str, stage),
                **summarize_events(events),
                "sha256": sha256_file(log_path),
            }
            usage_rows.append(public_usage_job(job_summary))

        raw_decisions = load_jsonl(dec_path, f"decisions:{model}:{domain}")
        if len(raw_decisions) != expected_count:
            raise RuntimeMatchedExportError(
                f"decision count mismatch for {model}:{domain}: {len(raw_decisions)}"
            )
        for row in raw_decisions:
            decision_rows.append(public_decision_row(row, model, domain, HY_RERANK_ARM))

        eval_payload = load_json(eval_path, f"eval:{model}:{domain}")
        eval_details = require_list(eval_payload.get("details"), f"eval:{model}:{domain}.details")
        if len(eval_details) != expected_count:
            raise RuntimeMatchedExportError(
                f"eval detail count mismatch for {model}:{domain}: {len(eval_details)}"
            )
        heldout_seen = 0
        for idx, item in enumerate(eval_details):
            row_obj = require_object(item, f"eval:{model}:{domain}.details[{idx}]")
            pub_a = public_answer_row(row_obj, model, domain, HY_RERANK_ARM)
            if not pub_a["is_validation"]:
                heldout_seen += 1
            answer_rows.append(pub_a)
        if heldout_seen != expected_heldout:
            raise RuntimeMatchedExportError(
                f"held-out count mismatch for {model}:{domain}: "
                f"expected={expected_heldout}, actual={heldout_seen}"
            )

        for m_path in (dec_manifest_path, ans_manifest_path):
            m_obj = load_json(m_path, f"manifest:{m_path}")
            runtime_rows.append(public_runtime_job(m_obj, sha256_file(m_path)))

        domain_audits[domain] = require_object(
            strip_paths(cast(JsonValue, dict(audit))),
            f"public-audit:{model}:{domain}",
        )

    ordered_answers = sorted(answer_rows, key=_answer_sort_key)
    ordered_decisions = sorted(decision_rows, key=_answer_sort_key)
    ordered_runtime = sorted(runtime_rows, key=_runtime_sort_key)
    ordered_usage = sorted(usage_rows, key=_usage_sort_key)

    if (
        len(ordered_answers) != 2830
        or len(ordered_decisions) != 2830
        or len(ordered_runtime) != 8
        or len(ordered_usage) != 8
    ):
        raise RuntimeMatchedExportError(
            f"unexpected pack counts for {model}: answers={len(ordered_answers)}, "
            f"decisions={len(ordered_decisions)}, runtime={len(ordered_runtime)}, "
            f"usage={len(ordered_usage)}"
        )

    write_jsonl_gzip(target_dir / "answer_per_instance.jsonl.gz", ordered_answers)
    write_jsonl_gzip(target_dir / "decision_per_instance.jsonl.gz", ordered_decisions)
    write_jsonl_gzip(target_dir / "runtime_jobs.jsonl.gz", ordered_runtime)
    write_jsonl_gzip(target_dir / "usage_jobs.jsonl.gz", ordered_usage)

    public_validation: JsonObject = {
        "schema_version": HY_RERANK_VALIDATION_SCHEMA_VERSION,
        "model": model,
        "served_model": served_model,
        "arm": HY_RERANK_ARM,
        "valid": True,
        "fresh_only": True,
        "domains": domain_audits,
        "domain_audit_sha256": audit_shas,
    }
    write_json(target_dir / "validation.json", public_validation)

    readme = _hy_rerank_readme(model, len(ordered_answers), len(ordered_decisions))
    (target_dir / "README.md").write_text(readme, encoding="utf-8")

    files: JsonObject = {
        "README.md": output_record(
            target_dir,
            "README.md",
            len(readme.splitlines()),
            "markdown-v1",
            "documentation",
        ),
        "answer_per_instance.jsonl.gz": output_record(
            target_dir,
            "answer_per_instance.jsonl.gz",
            len(ordered_answers),
            "runtime-matched-public-answer-row-v1",
            "per_instance",
        ),
        "decision_per_instance.jsonl.gz": output_record(
            target_dir,
            "decision_per_instance.jsonl.gz",
            len(ordered_decisions),
            "runtime-matched-public-decision-row-v1",
            "per_instance",
        ),
        "runtime_jobs.jsonl.gz": output_record(
            target_dir,
            "runtime_jobs.jsonl.gz",
            len(ordered_runtime),
            "runtime-matched-public-runtime-job-v1",
            "per_job",
        ),
        "usage_jobs.jsonl.gz": output_record(
            target_dir,
            "usage_jobs.jsonl.gz",
            len(ordered_usage),
            "runtime-matched-public-usage-job-v1",
            "per_job",
        ),
        "validation.json": output_record(
            target_dir,
            "validation.json",
            len(DOMAINS),
            HY_RERANK_VALIDATION_SCHEMA_VERSION,
            "audit",
        ),
    }
    manifest: JsonObject = {
        "schema_version": HY_RERANK_PACK_SCHEMA_VERSION,
        "model": model,
        "served_model": served_model,
        "arm": HY_RERANK_ARM,
        "domains": [d for d, _n, _h in DOMAINS],
        "fresh_only": True,
        "files": {name: files[name] for name in sorted(files)},
        "sources": {"domain_audit_sha256": audit_shas},
        "manifest_self_policy": {
            "included_in_files": False,
            "reason": "manifest.json is excluded because a file cannot contain its own stable SHA-256 digest.",
        },
    }
    write_json(target_dir / "manifest.json", manifest)
    scan_pack(target_dir)
    return {
        "model": model,
        "arm": HY_RERANK_ARM,
        "answers": len(ordered_answers),
        "decisions": len(ordered_decisions),
        "manifest_sha256": sha256_file(target_dir / "manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export validated E7 hy_rerank private outputs to a public pack."
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_hy_rerank_model_pack(
        args.result_root,
        str(args.model),
        args.output_dir,
    )
    print(canonical_json(summary), flush=True)


if __name__ == "__main__":
    main()
