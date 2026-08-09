#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/glm4-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
facts_root="$repo_root/runtime-staging/glm4-9b"
rerank_generation="$facts_root/rerank-decision.generation.json"
select_generation="$facts_root/select-decision.generation.json"
sragents_revision=277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

write_job_facts() {
  local source_path=$1
  local output_path=$2
  local job_id=$3
  local arm=$4
  local stage=$5
  "$python_path" - "$source_path" "$output_path" "$job_id" "$arm" "$stage" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
job_id = sys.argv[3]
arm = sys.argv[4]
stage = sys.argv[5]
payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["job"]["job_id"] = job_id
payload["job"]["arm"] = arm
payload["job"]["stage"] = stage
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

build_rerank_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-always_rerank-decision.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-always_rerank-decision.manifest.json"
  write_job_facts \
    "$facts_root/$domain.runtime-facts.json" \
    "$facts_path" \
    "glm4-9b-$domain-always_rerank-decision-20260725-v1" \
    always_rerank \
    decision
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$rerank_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "bm25_candidates=$result_root/bm25/$domain-bm25.json" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_rerank.py \
    --code-file scripts/run_runtime_matched_rerank_decisions.py \
    --code-file external/SR-Agents/src/sragents/corpus.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/retrieve/llm_rerank.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

build_select_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-select_bm25-decision.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-select_bm25-decision.manifest.json"
  write_job_facts \
    "$facts_root/$domain.runtime-facts.json" \
    "$facts_path" \
    "glm4-9b-$domain-select_bm25-decision-20260725-v1" \
    select_bm25 \
    decision
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$select_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "bm25_candidates=$result_root/bm25/$domain-bm25.json" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_select.py \
    --code-file scripts/run_runtime_matched_select_decisions.py \
    --code-file external/SR-Agents/src/sragents/corpus.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/providers/llm_select.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

launch_rerank_canary() {
  local domain=$1
  "$python_path" "$repo_root/scripts/run_runtime_matched_rerank_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$result_root/decisions/$domain-always_rerank.jsonl" \
    --attempt-log "$result_root/logs/$domain-always_rerank.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-always_rerank-decision.manifest.json" \
    --result-tag glm4-9b \
    --model glm4-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 1 \
    --max-new-records 5 \
    --sragents-checkout "$repo_root/external/SR-Agents" \
    --repository-root "$repo_root" \
    --sragents-revision "$sragents_revision" \
    >"$result_root/logs/$domain-always_rerank.decision.canary.log" 2>&1 &
}

launch_select_canary() {
  local domain=$1
  local expected=$2
  "$python_path" "$repo_root/scripts/run_runtime_matched_select_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$result_root/decisions/$domain-select_bm25.jsonl" \
    --attempt-log "$result_root/logs/$domain-select_bm25.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select_bm25-decision.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag glm4-9b \
    --model glm4-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 1 \
    --max-new-records 5 \
    >"$result_root/logs/$domain-select_bm25.decision.canary.log" 2>&1 &
}

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
mkdir -p \
  "$result_root/bm25" \
  "$result_root/decisions" \
  "$result_root/logs" \
  "$result_root/runtime"

for domain in "${domains[@]}"; do
  build_rerank_manifest "$domain"
  build_select_manifest "$domain"
done

pids=()
labels=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected=${counts[$index]}
  launch_rerank_canary "$domain"
  pids+=("$!")
  labels+=("$domain-always_rerank")
  launch_select_canary "$domain" "$expected"
  pids+=("$!")
  labels+=("$domain-select_bm25")
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  label=${labels[$index]}
  printf '%s\n' "$pid" >"$result_root/logs/$label.decision.canary.pid"
  if ! wait "$pid"; then
    printf 'Decision canary failed: label=%s pid=%s\n' "$label" "$pid" >&2
    status=1
  fi
done
exit "$status"
