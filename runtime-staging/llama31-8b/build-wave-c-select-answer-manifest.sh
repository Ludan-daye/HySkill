#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s DOMAIN\n' "$0" >&2
  exit 2
fi
domain=$1
case "$domain" in
  theoremqa|logicbench|medcalcbench|champ) ;;
  *)
    printf 'Unsupported domain: %s\n' "$domain" >&2
    exit 2
    ;;
esac

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
generation_path="$repo_root/runtime-staging/llama31-8b/generation.json"
decisions_path="$result_root/decisions/$domain-select_bm25.jsonl"
facts_path="$result_root/runtime/$domain-select_bm25-answer.runtime-facts.json"
manifest_path="$result_root/runtime/$domain-select_bm25-answer.manifest.json"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"

"$python_path" - \
  "$repo_root/runtime-staging/llama31-8b/$domain.runtime-facts.json" \
  "$facts_path" \
  "llama31-8b-$domain-select_bm25-answer-20260725-v1" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
job_id = sys.argv[3]
payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["job"]["job_id"] = job_id
payload["job"]["arm"] = "select_bm25"
payload["job"]["stage"] = "answer"
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

"$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
  --runtime-facts "$facts_path" \
  --generation "$generation_path" \
  --artifact "instances=$instance_root/$domain.json" \
  --artifact "corpus=$corpus_path" \
  --artifact "select_decisions=$decisions_path" \
  --code-file hyskill/runtime_matched_execution.py \
  --code-file hyskill/runtime_matched_select.py \
  --code-file scripts/run_runtime_matched_select_answers.py \
  --code-file external/SR-Agents/src/sragents/llm.py \
  --code-file external/SR-Agents/src/sragents/prompts.py \
  --code-file external/SR-Agents/src/sragents/infer/base.py \
  --code-file external/SR-Agents/src/sragents/infer/engines/direct.py \
  --code-file external/SR-Agents/src/sragents/infer/engines/tool_loop.py \
  --repository-root "$repo_root" \
  --output "$manifest_path"
