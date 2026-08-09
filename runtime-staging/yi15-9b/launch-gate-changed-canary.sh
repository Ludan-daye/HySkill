#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/yi15-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
generation_path="$repo_root/runtime-staging/yi15-9b/generation.json"
facts_root="$repo_root/runtime-staging/yi15-9b"
audit_root="$result_root/gate-audit/champ-routed"
domain=champ
arm=routed_gated
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

write_job_facts() {
  local source_path=$1
  local output_path=$2
  "$python_path" - "$source_path" "$output_path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["job"]["job_id"] = "yi15-9b-champ-routed-gated-20260724-v1"
payload["job"]["arm"] = "routed_gated"
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

mkdir -p "$result_root/runtime" "$result_root/gate-rerun" "$result_root/logs"
facts_path="$facts_root/$domain-$arm.runtime-facts.json"
facts_tmp="$facts_path.tmp"
write_job_facts "$facts_root/$domain.runtime-facts.json" "$facts_tmp"
mv "$facts_tmp" "$facts_path"

"$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
  --runtime-facts "$facts_path" \
  --generation "$generation_path" \
  --artifact "instances=$instance_root/$domain.json" \
  --artifact "corpus=$corpus_path" \
  --artifact "gate_audit=$audit_root/audit.json" \
  --artifact "gate_decisions=$audit_root/new-decisions.jsonl" \
  --artifact "gate_rerun=$audit_root/rerun-ids.json" \
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

"$python_path" "$repo_root/scripts/run_runtime_matched_gate_answers.py" \
  --instances "$instance_root/$domain.json" \
  --corpus "$corpus_path" \
  --gate-audit "$audit_root/audit.json" \
  --gate-decisions "$audit_root/new-decisions.jsonl" \
  --gate-rerun "$audit_root/rerun-ids.json" \
  --runtime-manifest "$result_root/runtime/$domain-$arm.manifest.json" \
  --repository-root "$repo_root" \
  --output "$result_root/gate-rerun/$domain-$arm.answers.jsonl" \
  --usage-log "$result_root/logs/$domain-$arm.gate-rerun.usage.jsonl" \
  --attempt-log "$result_root/logs/$domain-$arm.gate-rerun.attempts.jsonl" \
  --api-base "$api_base" \
  --workers 2 \
  --max-new-records 5
