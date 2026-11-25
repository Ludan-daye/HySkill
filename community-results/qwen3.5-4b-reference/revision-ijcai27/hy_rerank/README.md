# IJCAI-27 Revision E7 `hy_rerank` Public Pack: qwen3.5-4b-reference

This pack contains runtime-matched `hy_rerank` decisions and answers
over the frozen K=2 routed HySkill top-50 candidate pools across the
four rule-scored domains (`theoremqa`, `logicbench`, `medcalcbench`, `champ`).

## Protocol & Coverage

- Public answer rows: `2830` (`2,265` held-out test + `565` validation).
- Public decision rows: `2830` (`50` ordered routed candidates per instance).
- Prompt & parser: Frozen SR-Agents `277fd8d` listwise reranker (`temperature=0.0`, `max_tokens=1024`, up to 3 parse attempts, omitted candidates appended in input order).
- Answer generation: Direct engine (`temperature=0.7`, `max_tokens=2048`, `thinking=false`) matched to the K=2 checkpoint, tokenizer, chat template, vLLM version, BF16, and 8,192 context contract.
- Accounting: `method_failure` rows remain in the denominator and score `0`. Raw model text, gold answers, endpoints, host paths, GPU UUIDs, and credentials are stripped.
