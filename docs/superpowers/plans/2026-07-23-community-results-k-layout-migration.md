# GitHub K=2/K=4 结果目录迁移清单

> 日期：2026-07-23  
> 状态：`[~]` 目录契约已冻结；只读审计已完成，数据移动尚未执行。  
> 目标：把历史 K=4 主实验包归档到每模型 `k4/`，把新统一实验包写入
> `k2/`，同时保持联合 K 消融与五档 imagination prefix cache 独立。

## 1. 已验证的当前状态

- GitHub `origin/main` 与本地跟踪分支均为
  `5bc21aca567d671d92472b294f8c867a2a0e43b1`。
- 七个模型目录当前合计约 399 MB。
- 按第 3.1 节的白名单，待归档的 K=4 主实验包共 70 个文件、
  70,917,990 bytes；其余空间主要属于必须保持独立的 prefix cache。
- 历史主实验分析包位于 `community-results/<tag>/` 根部，其
  `summary.json` 使用 `k_samples=4`；Qwen3.5-4B reference 没有根部
  `summary.json`，但 README 与定性样例明确记录三模板 K=4。
- 每模型 `k-ablation/` 是 K={1,2,4,8,10} 的联合汇总，不是 K=4 主实验包。
- `imagination_full_k{1,2,4,8,10}.jsonl.gz` 及同名 manifest 是联合消融的
  完整前缀缓存，不属于 K=2 或 K=4 下游主实验包。

## 2. 目标结构

```text
community-results/<tag>/
├── README.md                         # model-level index
├── k2/                              # active unified main result
├── k4/                              # legacy main-result archive
├── k-ablation/                      # joint K=1/2/4/8/10 evidence
├── imagination_full_k1.*
├── imagination_full_k2.*
├── imagination_full_k4.*
├── imagination_full_k8.*
├── imagination_full_k10.*
└── baselines-native/                # if present; K-independent shared evidence
```

不为尚不存在的数据创建空目录，也不复制同一大型 gzip 来制造两份物理副本。

## 3. 文件分类

### 3.1 归档到 `k4/`

以下根部文件存在时使用 `git mv`，不存在时不得制造占位数据：

```text
summary.json
retrieval_top10.jsonl.gz
retrieval_top50.jsonl.gz
gating_per_instance.jsonl.gz
loading_per_instance.jsonl.gz
metrics_flat.jsonl.gz
router_decisions.json
imagination_samples.jsonl.gz
MANIFEST.md
significance.json
```

现有模型 README 是 K=4 包说明。迁移时先把原文保存为 `k4/README.md`，再在
模型根部创建新的目录索引 README；不能直接保留旧 README 并让其中链接失效。

### 3.2 新写入 `k2/`

K=2 包必须由已验收的服务器结果导出，至少包含：

```text
retrieval_top50.jsonl.gz
router_decisions.json
gating_per_instance.jsonl.gz
loading_per_instance.jsonl.gz
selection_per_instance.jsonl.gz
answer_per_instance.jsonl.gz
answer_metrics.json
metrics_flat.jsonl.gz
significance.json
reuse_manifest.json
manifest.json
README.md
```

上下文不支持 Select 的模型可不含 `selection_per_instance.jsonl.gz`，但
manifest 必须把该臂标为 unavailable 并记录原因，不能用空文件或零值代替。

### 3.3 保持原位

- `k-ablation/`
- `imagination_full_k{1,2,4,8,10}.jsonl.gz`
- `imagination_full_k{1,2,4,8,10}.manifest.json`
- `baselines-native/`

根部未知文件必须进入人工审计队列；迁移过程不得根据扩展名猜测归属。

## 4. 非破坏迁移顺序

1. 冻结远端 commit、模型标签和待迁移文件清单。
2. 对每个根部 K=4 文件记录相对路径、字节数和 SHA-256。
3. 验证全部 gzip 可解压、JSON/JSONL 可解析，旧 manifest 中的行数与哈希一致。
4. 独立导出并验收 K=2 包；K=2 不完整时停止，不移动 K=4。
5. 使用 `git mv` 把清单中的旧主实验文件移入 `<tag>/k4/`，以保留 Git 历史且
   不复制大文件。
6. 把旧模型 README 移为 `<tag>/k4/README.md`，新建根部索引 README。
7. 写入 `<tag>/k2/`，重新验证逐题支持集、gzip、JSON schema 和 manifest。
8. 对 K=4 目标文件复算 SHA-256，要求与步骤 2 逐文件完全一致。
9. 确认 `k-ablation/`、五档 prefix cache 和 `baselines-native/` 的路径与
   SHA-256 均未变化。
10. 更新仓库内所有根部旧链接，再运行分析包验证器；只有全部通过后才允许提交
    和推送。

## 5. 失败与回滚门禁

- 任一 K=4 源文件身份不能证明、目标路径已存在、SHA 不同或 gzip 损坏：停止。
- 任一 K=2 必需文件缺失、重复 instance ID、支持集不完整或 manifest 不一致：
  停止。
- 迁移未提交前用反向 `git mv` 恢复路径；禁止删除、覆盖或重新压缩旧文件。
- 已推送后只允许新提交恢复，禁止 force-push 或重写公开历史。
- 路径迁移与论文数值更新是两个独立审查项；不能因为目录完成就宣称实验完成。

## 6. 必须同步的读写方

路径迁移前的只读审计发现以下主实验读写方仍指向模型根目录，K2M010 必须逐项
修改并验证：

- 写入方：`scripts/summarize_multimodel.py`、`scripts/export_analysis_pack.py`、
  `scripts/export_top50.py`、`scripts/export_loading.py`、
  `scripts/export_reference_pack.py`、`scripts/run_multimodel.sh`；
- 文档与证据入口：根 `README.md`、`paper/STATE.md`、
  `docs/08-multimodel-plan.md`、`docs/09-summary.md`；
- 保持不变的 K 消融写入方：`scripts/export_full_imagination_cache.py` 和
  `scripts/run_k_ablation.sh`，因为它们仍应读写模型根部 prefix cache 及
  `k-ablation/`。

活跃论文与汇总脚本默认读取 `k2/`；只有明确标为 legacy K4 的复核命令才读取
`k4/`。不能用“目标文件不存在就回退根目录”的静默 fallback。

## 7. 完成标准

- 七模型均有可验证的 `k2/` 与 `k4/` 主实验包；
- K=4 迁移前后逐文件 SHA-256 完全一致；
- K=2 manifest 指向 K=2 retrieval、gate、selection 和回答证据；
- 联合 K 消融、五档 prefix cache 和共享 native baseline 未被移动或复制；
- 仓库内链接、导出脚本和验证器全部使用新目录契约；
- Git 状态中不存在未解释的删除、重复大型文件或敏感信息。
