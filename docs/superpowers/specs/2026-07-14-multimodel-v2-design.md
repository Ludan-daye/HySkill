# 多模型实验包 v2 设计（想象+重排全对比、验证集粒度路由、bm25s 提速）

日期：2026-07-14 ｜ 状态：已获用户批准的方向（方案 A），待 spec 评审
前置证据：`scripts/route_pilot.py` 离线回放（服务器 `results/route_pilot.json`，五域零缓存缺失）

## 1. 目标

v1 协作包（`c6e6e05`）只让每个模型跑固定"完整技能"想象。v2 补齐三件事：

1. **每个模型跑想象 × 5 变体 + llm_rerank**——"想象 vs 重排打平但便宜 10 倍"主张跨模型成立；
2. **P1 + P2 全流程**——P2 检索源升级为**域级验证集路由**（用户提案经离线验证：20% 验证集五域全部命中全量冠军，宏平均 nDCG@10 0.591→0.636）；
3. **bm25s 替换 rank_bm25**——四路/两阶段的 BM25 建索引与查询从每题 1.5–7s 降至毫秒级，志愿者不用交 10–20 小时"BM25 税"（也解锁我们自己的 K 消融）。

## 2. 路由协议（定案：L1 域级验证集路由）

- 变体池：naive_sentence / naive_passage / naive_skill / hyskill / two_stage（5 条）；
- 切分：与 gate.py calibrate 完全相同——sorted ids, `random.Random(0).sample(ids, max(1, int(n*0.2)))` 为验证集；
- 规则：每域在验证集上算各变体平均 nDCG@10，取 argmax；产出 `<ds>-routed.json`（= 冠军变体检索文件副本 + `metadata.router = {pick, val_ndcg_per_variant}`）;
- 测试报告只用其余 80%（P2 显著性脚本已支持 test-only）。
- **门控信号不随路由变**：S1/S2 仍用完整技能想象（知识快照角色），路由只改排序来源。想象一次、用三次（排序/路由/门控）。

### 已否决的替代方案（离线回放证据）
- 逐题统一裁判（cos(技能想象质心, 各变体 top-1 全文) argmax）：**数学退化**——该分数即 naive_skill 的排序函数，恒选 naive_skill；
- 逐题难度路由（4 份想象平均两两余弦 ≥τ→单路，否则四路）：阈值标定全部滑向极端，退化为域级选择（宏 0.631 < L1 0.636）；但作为无监督域级信号 4/5 域选对单/融，写进分析节；
- 逐题 ORACLE 上界 0.717（比 L1 高 8 分）：空间存在但免费信号够不着，future work。

## 3. 每模型实验菜单（志愿者视角）

| 阶段 | 内容 | 默认范围 | 预计（4090） |
|---|---|---|---|
| A1 预热 | 3 模板（sentence/passage/skill）× 3,970 题 × K=4 | 全 5 域 | 4–8 h |
| A2 P1 检索 | 5 变体 × 5 域（四路/两阶段走 bm25s） | 全 5 域 | 2–3 h |
| A3 P1 重排 | llm_rerank（串行长 prompt，最贵臂） | `RERANK_DOMAINS` 默认 theoremqa,logicbench,bigcodebench；`all` 可全开 | 2.5–7 h |
| B1 路由 | `route_variant.py` 产出 5 域 routed.json | 全 5 域 | 分钟级 |
| B2 P2 门控 | bare / always(routed top-1) / gated(routed+门控) 三臂 | 4 规则域 | 2–4 h |
| B3 选跑 | select 臂 | `SELECT=1` 才跑 | +2–3 h |
| 汇总 | summary.json（检索全表+路由选择+三臂准确率+门控统计） | — | 秒级 |

总计 ≈ 11–22 h 单张 24GB 卡，全程断点续跑。

## 4. 组件与改动

| 组件 | 改动 | 验证 |
|---|---|---|
| `hyskill/retriever.py` | BM25Okapi → bm25s（依赖加入 pyproject）；构建与查询接口不变 | 新单测：微型语料上 bm25s 与 rank_bm25 top-5 排序一致（或 nDCG 等价）；现有 34 测试全绿 |
| `scripts/route_variant.py`（新） | 读 5 变体检索文件 → val 选冠军 → 写 routed.json；缺变体文件时回退 naive_skill 并在 metadata 标注 | 单测：构造两变体玩具数据，验证选择与回退 |
| `scripts/run_multimodel.sh` | v2 阶段编排（上表 A1–B3），新增 RERANK_DOMAINS/SELECT 环境变量；rerank 走 `--plugin hyskill.plugin`（继承 max_tokens=1024 修补） | bash -n + 冒烟（mock 不适用 rerank，人工检查命令） |
| `scripts/summarize_multimodel.py` | 纳入 5 变体 + rerank 指标 + router picks | 随手动冒烟验证 |
| `scripts/gate.py` | 不动 | — |
| `docs/08-multimodel-plan.md` | 按 v2 重写菜单/耗时/回传格式 | — |

## 5. 错误处理

- 预热 empty>1% → 脚本警告（模板不兼容/端点故障的信号），继续但记录；
- 路由：某域验证集上全变体 nDCG=0（检索全废）→ 选 naive_skill 并标注 `degenerate=true`；
- rerank 中途崩溃 → 重跑同命令续跑（sragents retrieve 对已有输出的跳过语义沿用 v1 行为：文件存在即跳过整域）。

## 6. 我们自己服务器的对照消融（qwen35-9b + 4b 补做）

- routed vs 固定 skill 的 P2 对照（验证路由在做题端确实涨分——检索端 +4.5 nDCG 已证，做题端待量化）；
- qwen3.5-4b 的 routed P2 可直接复用现有缓存与 bare/gated 基建，只需重跑 always/gated 两臂 ×4 域（约 1.5 h）。

## 7. 里程碑

1. bm25s 替换 + 测试（先行，解锁一切）；
2. route_variant.py + 测试；
3. run_multimodel.sh v2 + 文档重写 + 推送（协作者可开跑）；
4. 本地服务器：qwen3.5-4b routed-P2 对照（最快出"路由涨做题分"证据）；
5. qwen35-9b 全菜单（我们认领的那份）。
