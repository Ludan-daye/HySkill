# 第二轮 IJCAI-27 补充实验数据： glm4-9b

本目录属于 2026-11-24 启动的第二轮 E0–E7；主实验的 Gated 和外部基线仍分别在 `../k2/` 与 `../baselines-runtime-matched/`。第二轮范围、复用关系及两种平均口径以[统一索引](../../revision-ijcai27-fleet/README.md)为准。

This directory contains four encoder metric/ranking pairs (23,820 rows each),
original manifests, and the MiniLM E0 reproduction report. The rankings retain
the validation and held-out instances; split labels are in the metric files.

`hy_rerank/` contains the compact E7 pack. `hy_rerank_source.tar.gz` contains
the exact source answers, decisions, evaluations, audits, runtime evidence, and
logs. Final E7 coverage is 2,830 answers and 2,830 decisions (2,265 held-out).

See the [fleet data index](../../revision-ijcai27-fleet/README.md) for protocols,
source-input archives, diagnostic boundaries, and complete SHA-256 verification.
