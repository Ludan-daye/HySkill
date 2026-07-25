#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s LOCAL_LLAMA_AUDIT_BUNDLE_SHA256\n' "$0" >&2
  exit 2
fi
local_llama_audit_bundle_sha256=$1
if [[ ! "$local_llama_audit_bundle_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  printf \
    'Invalid local Llama audit bundle SHA-256: value=%s\n' \
    "$local_llama_audit_bundle_sha256" >&2
  exit 2
fi

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_base=/root/HySkill-baseline-runtime-matched-20260724-results
llama_root="$result_base/llama31-8b"
glm_root="$result_base/glm4-9b"
python_path="$source_root/.venv/bin/python"
llama_checkpoint=/root/.cache/modelscope/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
glm_checkpoint=/root/.cache/modelscope/models/ZhipuAI--glm-4-9b-chat/snapshots/master
llama_pid_path="$llama_root/runtime/vllm.pid"
llama_endpoint_evidence="$llama_root/runtime/wave-c-final-endpoint-readback.json"
glm_endpoint_log="$glm_root/runtime/vllm.wave-c.log"
glm_pid_path="$glm_root/runtime/vllm.wave-c.pid"
glm_endpoint_evidence="$glm_root/runtime/wave-c-start-endpoint-readback.json"
api_models_url=http://127.0.0.1:8000/v1/models

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
arms=(always_rerank select_bm25)

"$python_path" - "$llama_root" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = {
    "theoremqa": 747,
    "logicbench": 760,
    "medcalcbench": 1100,
    "champ": 223,
}
arms = ("always_rerank", "select_bm25")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for domain, expected_rows in expected.items():
    for arm in arms:
        audit_path = root / "audits" / f"{domain}-{arm}.audit.json"
        if not audit_path.is_file():
            raise SystemExit(f"Missing Llama native audit: path={audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        required = {
            "schema_version": "runtime-matched-native-domain-audit-v1",
            "model": "llama31-8b",
            "served_model": "llama31-8b",
            "domain": domain,
            "arm": arm,
            "expected_rows": expected_rows,
            "observed_decisions": expected_rows,
            "observed_answers": expected_rows,
            "fresh_only": True,
            "reused_same_arm": 0,
            "unresolved": 0,
            "valid": True,
        }
        mismatches = {
            field: {"expected": value, "actual": audit.get(field)}
            for field, value in required.items()
            if audit.get(field) != value
        }
        if mismatches:
            raise SystemExit(
                f"Invalid Llama native audit: path={audit_path}, "
                f"mismatches={mismatches}"
            )
        categories = audit.get("failure_categories")
        if not isinstance(categories, dict):
            raise SystemExit(
                f"Missing Llama failure categories: path={audit_path}"
            )
        for stage in ("decision", "answer"):
            stage_categories = categories.get(stage)
            if not isinstance(stage_categories, dict):
                raise SystemExit(
                    f"Missing Llama stage categories: "
                    f"path={audit_path}, stage={stage}"
                )
            unresolved = sum(
                int(stage_categories.get(category, 0))
                for category in ("infra_transient", "unclassified_error")
            )
            if unresolved != 0:
                raise SystemExit(
                    f"Unresolved Llama outcomes: path={audit_path}, "
                    f"stage={stage}, count={unresolved}"
                )
        artifacts = audit.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SystemExit(
                f"Missing Llama audit artifacts: path={audit_path}"
            )
        for name, record in artifacts.items():
            if not isinstance(record, dict):
                raise SystemExit(
                    f"Malformed Llama audit artifact: "
                    f"path={audit_path}, name={name}"
                )
            artifact_path_value = record.get("path")
            artifact_sha = record.get("sha256")
            if (
                not isinstance(artifact_path_value, str)
                or not isinstance(artifact_sha, str)
            ):
                raise SystemExit(
                    f"Incomplete Llama audit artifact: "
                    f"path={audit_path}, name={name}"
                )
            artifact_path = Path(artifact_path_value)
            if not artifact_path.is_file():
                raise SystemExit(
                    f"Missing Llama bound artifact: "
                    f"audit={audit_path}, artifact={artifact_path}"
                )
            actual_sha = sha256_file(artifact_path)
            if actual_sha != artifact_sha:
                raise SystemExit(
                    f"Llama bound artifact SHA mismatch: "
                    f"audit={audit_path}, artifact={artifact_path}, "
                    f"expected={artifact_sha}, actual={actual_sha}"
                )
        usage = audit.get("usage")
        if not isinstance(usage, dict):
            raise SystemExit(f"Missing Llama usage audit: path={audit_path}")
        for stage in ("decision", "answer"):
            record = usage.get(stage)
            if not isinstance(record, dict):
                raise SystemExit(
                    f"Missing Llama usage stage: "
                    f"path={audit_path}, stage={stage}"
                )
            usage_path_value = record.get("path")
            usage_sha = record.get("sha256")
            if (
                not isinstance(usage_path_value, str)
                or not isinstance(usage_sha, str)
            ):
                raise SystemExit(
                    f"Incomplete Llama usage artifact: "
                    f"path={audit_path}, stage={stage}"
                )
            usage_path = Path(usage_path_value)
            actual_usage_sha = sha256_file(usage_path)
            if actual_usage_sha != usage_sha:
                raise SystemExit(
                    f"Llama usage SHA mismatch: path={usage_path}, "
                    f"expected={usage_sha}, actual={actual_usage_sha}"
                )
PY

audit_files=()
for domain in "${domains[@]}"; do
  for arm in "${arms[@]}"; do
    audit_files+=("$domain-$arm.audit.json")
  done
done
remote_llama_audit_bundle_sha256=$(
  (
    cd "$llama_root/audits"
    sha256sum "${audit_files[@]}"
  ) | sha256sum | cut -d ' ' -f 1
)
if [[ "$remote_llama_audit_bundle_sha256" \
  != "$local_llama_audit_bundle_sha256" ]]; then
  printf \
    'Local/remote Llama audit bundle mismatch: local=%s remote=%s\n' \
    "$local_llama_audit_bundle_sha256" \
    "$remote_llama_audit_bundle_sha256" >&2
  exit 1
fi

"$python_path" - "$llama_root" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

root = str(Path(sys.argv[1]).resolve())
needles = (
    "run_runtime_matched_rerank",
    "run_runtime_matched_select",
    "/run-wave-c-",
    "/launch-wave-c-",
    "/watch-wave-c-",
)
active = []
for proc_path in Path("/proc").iterdir():
    if not proc_path.name.isdigit():
        continue
    try:
        arguments = [
            item.decode("utf-8")
            for item in (proc_path / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    command = " ".join(arguments)
    if root in command and any(needle in command for needle in needles):
        active.append({"pid": int(proc_path.name), "command": command})
if active:
    raise SystemExit(f"Llama native processes are still active: {active}")
PY

mkdir -p \
  "$glm_root/answers" \
  "$glm_root/audits" \
  "$glm_root/decisions" \
  "$glm_root/eval" \
  "$glm_root/logs" \
  "$glm_root/runtime"
glm_new_paths=(
  "$glm_endpoint_log"
  "$glm_pid_path"
  "$glm_endpoint_evidence"
  "$glm_root/logs/wave-c-rerank-decision.wrapper.log"
  "$glm_root/logs/wave-c-rerank-decision.wrapper.pid"
  "$glm_root/logs/wave-c-select-decision.wrapper.log"
  "$glm_root/logs/wave-c-select-decision.wrapper.pid"
)
for domain in "${domains[@]}"; do
  for arm in "${arms[@]}"; do
    glm_new_paths+=(
      "$glm_root/decisions/$domain-$arm.jsonl"
      "$glm_root/answers/$domain-$arm.jsonl"
      "$glm_root/eval/$domain-$arm.eval.json"
      "$glm_root/audits/$domain-$arm.audit.json"
      "$glm_root/runtime/$domain-$arm-decision.runtime-facts.json"
      "$glm_root/runtime/$domain-$arm-decision.manifest.json"
      "$glm_root/runtime/$domain-$arm-answer.runtime-facts.json"
      "$glm_root/runtime/$domain-$arm-answer.manifest.json"
      "$glm_root/logs/$domain-$arm.decision.attempts.jsonl"
      "$glm_root/logs/$domain-$arm.answer.attempts.jsonl"
      "$glm_root/logs/$domain-$arm.decision.canary.log"
      "$glm_root/logs/$domain-$arm.decision.canary.pid"
      "$glm_root/logs/$domain-$arm.decision.full.log"
      "$glm_root/logs/$domain-$arm.decision.full.pid"
      "$glm_root/logs/$domain-$arm.answer.canary.log"
      "$glm_root/logs/$domain-$arm.answer.full.log"
      "$glm_root/logs/$domain-$arm.audit-watcher.log"
      "$glm_root/logs/$domain-$arm.audit-watcher.pid"
      "$glm_root/logs/$domain-$arm.pipeline.log"
      "$glm_root/logs/$domain-$arm.pipeline.pid"
    )
  done
done
for path in "${glm_new_paths[@]}"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to overwrite GLM Wave C evidence: %s\n' "$path" >&2
    exit 1
  fi
done

if [[ ! -s "$llama_pid_path" ]]; then
  printf 'Missing Llama endpoint PID file: %s\n' "$llama_pid_path" >&2
  exit 1
fi
llama_endpoint_pid=$(<"$llama_pid_path")
if [[ -e "$llama_endpoint_evidence" ]]; then
  printf \
    'Refusing to overwrite Llama endpoint evidence: %s\n' \
    "$llama_endpoint_evidence" >&2
  exit 1
fi

"$python_path" - \
  "$llama_endpoint_pid" \
  "$llama_checkpoint" \
  "$api_models_url" \
  "$llama_endpoint_evidence" \
  "$remote_llama_audit_bundle_sha256" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

pid = int(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
models_url = sys.argv[3]
output_path = Path(sys.argv[4])
audit_bundle_sha256 = sys.argv[5]
proc_root = Path(f"/proc/{pid}")
if not proc_root.is_dir():
    raise SystemExit(f"Llama endpoint is not live: pid={pid}")
cmdline = [
    item.decode("utf-8")
    for item in (proc_root / "cmdline").read_bytes().split(b"\0")
    if item
]


def require_flag(flag: str, expected: str) -> None:
    if cmdline.count(flag) != 1:
        raise SystemExit(
            f"Llama endpoint flag count mismatch: flag={flag}, "
            f"command={cmdline}"
        )
    index = cmdline.index(flag)
    if index + 1 >= len(cmdline) or cmdline[index + 1] != expected:
        raise SystemExit(
            f"Llama endpoint flag mismatch: flag={flag}, "
            f"expected={expected!r}, command={cmdline}"
        )


if "vllm" not in " ".join(cmdline) or "serve" not in cmdline:
    raise SystemExit(
        f"Refusing to stop a non-vLLM process: pid={pid}, command={cmdline}"
    )
if checkpoint not in cmdline:
    raise SystemExit(
        f"Llama checkpoint mismatch: expected={checkpoint}, command={cmdline}"
    )
require_flag("--port", "8000")
require_flag("--max-model-len", "8192")
require_flag("--dtype", "bfloat16")
require_flag("--served-model-name", "llama31-8b")
with urllib.request.urlopen(models_url, timeout=10.0) as response:
    response_bytes = response.read()
models = json.loads(response_bytes.decode("utf-8"))
matching = [
    item
    for item in models.get("data", [])
    if isinstance(item, dict) and item.get("id") == "llama31-8b"
]
if len(matching) != 1:
    raise SystemExit(f"Invalid Llama model readback: payload={models}")
model = matching[0]
if model.get("root") != checkpoint or model.get("max_model_len") != 8192:
    raise SystemExit(f"Invalid Llama runtime readback: model={model}")
gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
payload = {
    "schema_version": "runtime-matched-endpoint-switch-evidence-v1",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "llama31-8b-final-readback-before-switch",
    "pid": pid,
    "cmdline": cmdline,
    "models_readback": models,
    "gpu_readback": gpu,
    "native_audit_bundle_sha256": audit_bundle_sha256,
    "valid": True,
}
with output_path.open("x", encoding="utf-8") as output_file:
    output_file.write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    output_file.flush()
    os.fsync(output_file.fileno())
PY

kill -TERM "$llama_endpoint_pid"
for _attempt in $(seq 1 180); do
  if ! kill -0 "$llama_endpoint_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$llama_endpoint_pid" 2>/dev/null; then
  printf \
    'Llama endpoint did not stop after SIGTERM: pid=%s\n' \
    "$llama_endpoint_pid" >&2
  exit 1
fi

for path in "${glm_new_paths[@]}"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to overwrite GLM Wave C evidence: %s\n' "$path" >&2
    exit 1
  fi
done

nohup /root/vllmenv/bin/vllm serve "$glm_checkpoint" \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --enforce-eager \
  --trust-remote-code \
  --served-model-name glm4-9b \
  >"$glm_endpoint_log" 2>&1 &
glm_endpoint_pid=$!
printf '%s\n' "$glm_endpoint_pid" >"$glm_pid_path"

endpoint_ready=0
for _attempt in $(seq 1 240); do
  if curl -fsS "$api_models_url" \
    | grep -q '"id":"glm4-9b"'; then
    endpoint_ready=1
    break
  fi
  if ! kill -0 "$glm_endpoint_pid" 2>/dev/null; then
    printf \
      'GLM endpoint exited during startup: pid=%s log=%s\n' \
      "$glm_endpoint_pid" \
      "$glm_endpoint_log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$endpoint_ready" -ne 1 ]]; then
  printf \
    'GLM endpoint did not become ready within 20 minutes: pid=%s log=%s\n' \
    "$glm_endpoint_pid" \
    "$glm_endpoint_log" >&2
  exit 1
fi

"$python_path" - \
  "$glm_endpoint_pid" \
  "$glm_checkpoint" \
  "$api_models_url" \
  "$glm_endpoint_evidence" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

pid = int(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
models_url = sys.argv[3]
output_path = Path(sys.argv[4])
proc_root = Path(f"/proc/{pid}")
if not proc_root.is_dir():
    raise SystemExit(f"GLM endpoint is not live: pid={pid}")
cmdline = [
    item.decode("utf-8")
    for item in (proc_root / "cmdline").read_bytes().split(b"\0")
    if item
]


def require_flag(flag: str, expected: str) -> None:
    if cmdline.count(flag) != 1:
        raise SystemExit(
            f"GLM endpoint flag count mismatch: flag={flag}, "
            f"command={cmdline}"
        )
    index = cmdline.index(flag)
    if index + 1 >= len(cmdline) or cmdline[index + 1] != expected:
        raise SystemExit(
            f"GLM endpoint flag mismatch: flag={flag}, "
            f"expected={expected!r}, command={cmdline}"
        )


if "vllm" not in " ".join(cmdline) or "serve" not in cmdline:
    raise SystemExit(f"Invalid GLM vLLM process: pid={pid}, command={cmdline}")
if checkpoint not in cmdline:
    raise SystemExit(
        f"GLM checkpoint mismatch: expected={checkpoint}, command={cmdline}"
    )
require_flag("--port", "8000")
require_flag("--max-model-len", "8192")
require_flag("--dtype", "bfloat16")
require_flag("--served-model-name", "glm4-9b")
if cmdline.count("--trust-remote-code") != 1:
    raise SystemExit(f"GLM trust-remote-code mismatch: command={cmdline}")
with urllib.request.urlopen(models_url, timeout=10.0) as response:
    response_bytes = response.read()
models = json.loads(response_bytes.decode("utf-8"))
matching = [
    item
    for item in models.get("data", [])
    if isinstance(item, dict) and item.get("id") == "glm4-9b"
]
if len(matching) != 1:
    raise SystemExit(f"Invalid GLM model readback: payload={models}")
model = matching[0]
if model.get("root") != checkpoint or model.get("max_model_len") != 8192:
    raise SystemExit(f"Invalid GLM runtime readback: model={model}")
gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
payload = {
    "schema_version": "runtime-matched-endpoint-switch-evidence-v1",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "glm4-9b-start-readback-before-wave-c",
    "pid": pid,
    "cmdline": cmdline,
    "models_readback": models,
    "models_readback_sha256": hashlib.sha256(response_bytes).hexdigest(),
    "gpu_readback": gpu,
    "valid": True,
}
with output_path.open("x", encoding="utf-8") as output_file:
    output_file.write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    output_file.flush()
    os.fsync(output_file.fileno())
PY

bash "$repo_root/runtime-staging/glm4-9b/launch-wave-c-decision-canary.sh"

"$python_path" - "$glm_root" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = {
    "theoremqa": 747,
    "logicbench": 760,
    "medcalcbench": 1100,
    "champ": 223,
}
arms = ("always_rerank", "select_bm25")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for domain, expected_rows in expected.items():
    for arm in arms:
        decision_path = root / "decisions" / f"{domain}-{arm}.jsonl"
        log_path = root / "logs" / f"{domain}-{arm}.decision.canary.log"
        manifest_path = (
            root / "runtime" / f"{domain}-{arm}-decision.manifest.json"
        )
        rows = [
            json.loads(line)
            for line in decision_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 5:
            raise SystemExit(
                f"GLM decision canary count mismatch: "
                f"domain={domain}, arm={arm}, actual={len(rows)}"
            )
        if len({row.get("instance_id") for row in rows}) != 5:
            raise SystemExit(
                f"GLM decision canary duplicate IDs: "
                f"domain={domain}, arm={arm}"
            )
        schema_version = (
            "runtime-matched-rerank-decision-v1"
            if arm == "always_rerank"
            else "runtime-matched-select-decision-v1"
        )
        manifest_sha = sha256_file(manifest_path)
        resolved = (
            {"success", "method_failure"}
            if arm == "always_rerank"
            else {"success", "selector_fallback", "method_failure"}
        )
        for row in rows:
            required = {
                "schema_version": schema_version,
                "model": "glm4-9b",
                "served_model": "glm4-9b",
                "domain": domain,
                "arm": arm,
                "stage": "decision",
                "runtime_manifest_sha256": manifest_sha,
            }
            mismatches = {
                field: {"expected": value, "actual": row.get(field)}
                for field, value in required.items()
                if row.get(field) != value
            }
            if mismatches or row.get("failure_category") not in resolved:
                raise SystemExit(
                    f"Invalid GLM decision canary row: "
                    f"domain={domain}, arm={arm}, "
                    f"instance={row.get('instance_id')}, "
                    f"mismatches={mismatches}, "
                    f"category={row.get('failure_category')!r}"
                )
        event = (
            "rerank_decision_complete"
            if arm == "always_rerank"
            else "runtime_matched_select_decisions_complete"
        )
        terminal = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            if payload.get("event") == event:
                terminal.append(payload)
        if len(terminal) != 1:
            raise SystemExit(
                f"GLM canary terminal count mismatch: "
                f"domain={domain}, arm={arm}, actual={len(terminal)}"
            )
        summary = terminal[0]
        if (
            summary.get("run_mode") != "canary"
            or summary.get("expected") != expected_rows
            or summary.get("observed") != 5
            or summary.get("selected_this_run") != 5
            or summary.get("output_sha256") != sha256_file(decision_path)
        ):
            raise SystemExit(
                f"Invalid GLM canary terminal: "
                f"domain={domain}, arm={arm}, summary={summary}"
            )
PY

nohup bash \
  "$repo_root/runtime-staging/glm4-9b/launch-wave-c-rerank-decision-full.sh" \
  >"$glm_root/logs/wave-c-rerank-decision.wrapper.log" 2>&1 &
rerank_wrapper_pid=$!
printf '%s\n' \
  "$rerank_wrapper_pid" \
  >"$glm_root/logs/wave-c-rerank-decision.wrapper.pid"

nohup bash \
  "$repo_root/runtime-staging/glm4-9b/launch-wave-c-select-decision-full.sh" \
  >"$glm_root/logs/wave-c-select-decision.wrapper.log" 2>&1 &
select_wrapper_pid=$!
printf '%s\n' \
  "$select_wrapper_pid" \
  >"$glm_root/logs/wave-c-select-decision.wrapper.pid"

for _attempt in $(seq 1 60); do
  missing_pid_files=0
  for domain in "${domains[@]}"; do
    for arm in "${arms[@]}"; do
      pid_path="$glm_root/logs/$domain-$arm.decision.full.pid"
      if [[ ! -s "$pid_path" ]]; then
        missing_pid_files=$((missing_pid_files + 1))
      fi
    done
  done
  if [[ "$missing_pid_files" -eq 0 ]]; then
    break
  fi
  if ! kill -0 "$rerank_wrapper_pid" 2>/dev/null; then
    printf \
      'GLM Rerank decision wrapper exited before PID publication: pid=%s\n' \
      "$rerank_wrapper_pid" >&2
    exit 1
  fi
  if ! kill -0 "$select_wrapper_pid" 2>/dev/null; then
    printf \
      'GLM Select decision wrapper exited before PID publication: pid=%s\n' \
      "$select_wrapper_pid" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$missing_pid_files" -ne 0 ]]; then
  printf \
    'GLM decision PID files were not published: missing=%s\n' \
    "$missing_pid_files" >&2
  exit 1
fi

for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  for arm in "${arms[@]}"; do
    watcher_log="$glm_root/logs/$domain-$arm.audit-watcher.log"
    watcher_pid="$glm_root/logs/$domain-$arm.audit-watcher.pid"
    if [[ -e "$watcher_log" || -e "$watcher_pid" ]]; then
      printf \
        'Refusing to overwrite GLM audit watcher evidence: domain=%s arm=%s\n' \
        "$domain" \
        "$arm" >&2
      exit 1
    fi
    nohup bash \
      "$repo_root/runtime-staging/glm4-9b/watch-wave-c-native-domain-audit.sh" \
      "$domain" \
      "$arm" \
      >"$watcher_log" 2>&1 &
    printf '%s\n' "$!" >"$watcher_pid"
  done

  rerank_pipeline_log="$glm_root/logs/$domain-always_rerank.pipeline.log"
  rerank_pipeline_pid="$glm_root/logs/$domain-always_rerank.pipeline.pid"
  select_pipeline_log="$glm_root/logs/$domain-select_bm25.pipeline.log"
  select_pipeline_pid="$glm_root/logs/$domain-select_bm25.pipeline.pid"
  for path in \
    "$rerank_pipeline_log" \
    "$rerank_pipeline_pid" \
    "$select_pipeline_log" \
    "$select_pipeline_pid"; do
    if [[ -e "$path" ]]; then
      printf 'Refusing to overwrite GLM pipeline evidence: %s\n' "$path" >&2
      exit 1
    fi
  done
  nohup bash \
    "$repo_root/runtime-staging/glm4-9b/run-wave-c-rerank-answer-pipeline.sh" \
    "$domain" \
    >"$rerank_pipeline_log" 2>&1 &
  printf '%s\n' "$!" >"$rerank_pipeline_pid"
  nohup bash \
    "$repo_root/runtime-staging/glm4-9b/run-wave-c-select-answer-pipeline.sh" \
    "$domain" \
    >"$select_pipeline_log" 2>&1 &
  printf '%s\n' "$!" >"$select_pipeline_pid"
done

printf \
  'GLM Wave C launched: endpoint_pid=%s rerank_wrapper_pid=%s select_wrapper_pid=%s\n' \
  "$glm_endpoint_pid" \
  "$rerank_wrapper_pid" \
  "$select_wrapper_pid"
