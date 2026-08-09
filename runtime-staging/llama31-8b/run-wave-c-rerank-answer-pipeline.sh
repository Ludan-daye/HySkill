#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s DOMAIN\n' "$0" >&2
  exit 2
fi
domain=$1
case "$domain" in
  theoremqa) expected=747 ;;
  logicbench) expected=760 ;;
  medcalcbench) expected=1100 ;;
  *)
    printf 'Unsupported or already-running domain: %s\n' "$domain" >&2
    exit 2
    ;;
esac

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
python_path="$source_root/.venv/bin/python"
decision_path="$result_root/decisions/$domain-always_rerank.jsonl"
decision_log="$result_root/logs/$domain-always_rerank.decision.full.log"
decision_pid_path="$result_root/logs/$domain-always_rerank.decision.full.pid"
answer_path="$result_root/answers/$domain-always_rerank.jsonl"
answer_canary_log="$result_root/logs/$domain-always_rerank.answer.canary.log"

if [[ ! -s "$decision_pid_path" ]]; then
  printf 'Missing decision PID file: %s\n' "$decision_pid_path" >&2
  exit 1
fi
decision_pid=$(<"$decision_pid_path")
while true; do
  observed=0
  if [[ -f "$decision_path" ]]; then
    observed=$(wc -l <"$decision_path")
  fi
  if [[ "$observed" -gt "$expected" ]]; then
    printf \
      'Rerank decision count exceeds protocol: domain=%s expected=%s observed=%s\n' \
      "$domain" \
      "$expected" \
      "$observed" >&2
    exit 1
  fi
  if [[ "$observed" -eq "$expected" ]] \
    && grep -q '"event":"rerank_decision_complete"' "$decision_log"; then
    break
  fi
  if ! kill -0 "$decision_pid" 2>/dev/null; then
    printf \
      'Rerank decision exited before a complete terminal state: domain=%s observed=%s\n' \
      "$domain" \
      "$observed" >&2
    exit 1
  fi
  sleep 10
done

"$python_path" - \
  "$decision_path" \
  "$decision_log" \
  "$domain" \
  "$expected" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

decision_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
domain = sys.argv[3]
expected = int(sys.argv[4])
rows = [
    json.loads(line)
    for line in decision_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != expected:
    raise SystemExit(
        f"Decision count mismatch: domain={domain} "
        f"expected={expected} actual={len(rows)}"
    )
instance_ids = [row.get("instance_id") for row in rows]
if len(instance_ids) != len(set(instance_ids)):
    raise SystemExit(f"Duplicate decision IDs: domain={domain}")
resolved = {"success", "method_failure"}
for row in rows:
    expected_identity = {
        "schema_version": "runtime-matched-rerank-decision-v1",
        "model": "llama31-8b",
        "served_model": "llama31-8b",
        "domain": domain,
        "arm": "always_rerank",
        "stage": "decision",
    }
    for field, expected_value in expected_identity.items():
        if row.get(field) != expected_value:
            raise SystemExit(
                f"Decision identity mismatch: domain={domain} "
                f"instance={row.get('instance_id')} field={field} "
                f"expected={expected_value!r} actual={row.get(field)!r}"
            )
    if row.get("failure_category") not in resolved:
        raise SystemExit(
            f"Unresolved decision: domain={domain} "
            f"instance={row.get('instance_id')} "
            f"category={row.get('failure_category')!r}"
        )
terminal = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if not line.startswith("{"):
        continue
    payload = json.loads(line)
    if payload.get("event") == "rerank_decision_complete":
        terminal.append(payload)
if len(terminal) != 1:
    raise SystemExit(
        f"Expected one decision terminal event: "
        f"domain={domain} actual={len(terminal)}"
    )
summary = terminal[0]
digest = hashlib.sha256(decision_path.read_bytes()).hexdigest()
if (
    summary.get("run_mode") != "full"
    or summary.get("expected") != expected
    or summary.get("observed") != expected
    or summary.get("output_sha256") != digest
):
    raise SystemExit(
        f"Decision terminal mismatch: domain={domain} summary={summary}"
    )
PY

bash "$repo_root/runtime-staging/llama31-8b/build-wave-c-rerank-answer-manifest.sh" \
  "$domain"
bash "$repo_root/runtime-staging/llama31-8b/launch-wave-c-rerank-answer-canary.sh" \
  "$domain"

"$python_path" - \
  "$answer_path" \
  "$answer_canary_log" \
  "$domain" \
  "$expected" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

answer_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
domain = sys.argv[3]
expected = int(sys.argv[4])
rows = [
    json.loads(line)
    for line in answer_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 5:
    raise SystemExit(
        f"Answer canary count mismatch: domain={domain} actual={len(rows)}"
    )
if len({row.get("instance_id") for row in rows}) != 5:
    raise SystemExit(f"Duplicate answer canary IDs: domain={domain}")
for row in rows:
    if (
        row.get("schema_version")
        != "runtime-matched-baseline-answer-v1"
        or row.get("model") != "llama31-8b"
        or row.get("served_model") != "llama31-8b"
        or row.get("domain") != domain
        or row.get("arm") != "always_rerank"
        or row.get("stage") != "answer"
        or row.get("reused_same_arm") is not False
        or row.get("failure_category")
        not in {"success", "method_failure"}
    ):
        raise SystemExit(
            f"Invalid answer canary row: domain={domain} row={row}"
        )
terminal = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if not line.startswith("{"):
        continue
    payload = json.loads(line)
    if payload.get("event") == "rerank_answer_complete":
        terminal.append(payload)
if len(terminal) != 1:
    raise SystemExit(
        f"Expected one answer canary terminal event: "
        f"domain={domain} actual={len(terminal)}"
    )
summary = terminal[0]
digest = hashlib.sha256(answer_path.read_bytes()).hexdigest()
if (
    summary.get("run_mode") != "canary"
    or summary.get("expected") != expected
    or summary.get("observed") != 5
    or summary.get("selected_this_run") != 5
    or summary.get("output_sha256") != digest
):
    raise SystemExit(
        f"Answer canary terminal mismatch: domain={domain} summary={summary}"
    )
PY

exec bash \
  "$repo_root/runtime-staging/llama31-8b/launch-wave-c-rerank-answer-full.sh" \
  "$domain"
