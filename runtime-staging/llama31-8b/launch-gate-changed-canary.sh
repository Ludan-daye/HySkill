#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
generation_path="$repo_root/runtime-staging/llama31-8b/generation.json"
facts_root="$repo_root/runtime-staging/llama31-8b"
audit_root="$result_root/gate-audit"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

write_job_facts() {
  local source_path=$1
  local output_path=$2
  local job_id=$3
  local arm=$4
  "$python_path" - "$source_path" "$output_path" "$job_id" "$arm" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
job_id = sys.argv[3]
arm = sys.argv[4]
payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["job"]["job_id"] = job_id
payload["job"]["arm"] = arm
output_path.write_text(
    json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

domains=(theoremqa champ)
arms=(routed_gated routed_gated)
audit_dirs=(theoremqa-routed champ-routed)
mkdir -p "$result_root/runtime" "$result_root/gate-rerun" "$result_root/logs"

for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  arm=${arms[$index]}
  audit_dir=${audit_dirs[$index]}
  facts_path="$facts_root/$domain-$arm.runtime-facts.json"
  facts_tmp="$facts_path.tmp"
  write_job_facts \
    "$facts_root/$domain.runtime-facts.json" \
    "$facts_tmp" \
    "llama31-8b-$domain-$arm-20260724-v1" \
    "$arm"
  mv "$facts_tmp" "$facts_path"
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$generation_path" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "gate_audit=$audit_root/$audit_dir/audit.json" \
    --artifact "gate_decisions=$audit_root/$audit_dir/new-decisions.jsonl" \
    --artifact "gate_rerun=$audit_root/$audit_dir/rerun-ids.json" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_gate.py \
    --code-file hyskill/runtime_matched_bm25.py \
    --code-file scripts/run_runtime_matched_bare.py \
    --code-file scripts/run_runtime_matched_gate_answers.py \
    --code-file scripts/merge_runtime_matched_gate_answers.py \
    --code-file scripts/summarize_runtime_matched_gate.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/base.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/direct.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/tool_loop.py \
    --repository-root "$repo_root" \
    --output "$result_root/runtime/$domain-$arm.manifest.json"
done

pids=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  arm=${arms[$index]}
  audit_dir=${audit_dirs[$index]}
  "$python_path" "$repo_root/scripts/run_runtime_matched_gate_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --gate-audit "$audit_root/$audit_dir/audit.json" \
    --gate-decisions "$audit_root/$audit_dir/new-decisions.jsonl" \
    --gate-rerun "$audit_root/$audit_dir/rerun-ids.json" \
    --runtime-manifest "$result_root/runtime/$domain-$arm.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/gate-rerun/$domain-$arm.answers.jsonl" \
    --usage-log "$result_root/logs/$domain-$arm.gate-rerun.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-$arm.gate-rerun.attempts.jsonl" \
    --api-base "$api_base" \
    --workers 2 \
    --max-new-records 5 \
    >"$result_root/logs/$domain-$arm.gate-rerun.canary.log" 2>&1 &
  pid=$!
  echo "$pid" >"$result_root/logs/$domain-$arm.gate-rerun.canary.pid"
  pids+=("$pid")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
