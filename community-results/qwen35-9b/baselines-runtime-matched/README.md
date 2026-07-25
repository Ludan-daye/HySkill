# Runtime-matched baseline pack: qwen35-9b

This pack contains fresh baseline evidence generated under the same checkpoint, tokenizer, chat template, vLLM, BF16, and 8,192-token runtime contract as the active K=2 experiment.

## Coverage

- Public answer rows: 8490.
- Public decision rows: 5660.
- Bare is available on all four rule-scored domains.
- Native Rerank and BM25 Select are available on all four domains.

## Evidence policy

- No legacy compact baseline row is included.
- Actual service-reported usage is retained without imputation.
- Deterministic method failures remain in the denominator.
- Raw model text, gold answers, endpoints, server paths, GPU identifiers, and credentials are omitted.
