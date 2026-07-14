# 多模型协作实验计划 v2（众包版）

> **目的**：现有全部结果出自单一生成模型（Qwen3.5-4B）。本计划把完整对比矩阵——**想象检索（5 变体）vs LLM 重排、粒度路由、加载门控做题**——在 5–6 个不同家族的开源模型上复跑，证明方法与模型无关。
> **形式**：每人认领一个模型 + 一张 ≥24GB 显卡，跑一条命令，回传一个几 KB 的 JSON。全流程断点续跑。
> v1→v2 变化：新增 llm_rerank 对比臂、5 变体全跑、验证集粒度路由（离线验证 5/5 域选中真冠军）、BM25 换 bm25s 后端（快约百倍，四路/两阶段不再有"BM25 税"）。

## 1. 跑什么：一条命令里的六个阶段

| 阶段 | 内容 | 回答什么 | 单卡耗时（4090 级） |
|---|---|---|---|
| 1 预热 | 3 想象模板 × 3,970 题 × K=4（≈4.8 万次短生成，32 并发） | （备料，断点续跑） | 4–8 h |
| 2 想象检索 | **5 变体**（一句话/一段话/完整技能/四路/两阶段）× 5 域 | 想象优于基线、粒度规律是否跨模型成立 | 2–3 h |
| 3 重排对比 | fast_bm25 出候选 → **llm_rerank**（默认 3 争议域，`RERANK_DOMAINS=all` 开全量） | "想象 vs 重排"用你的模型自己排自己比 | 2.5–7 h |
| 4 粒度路由 | 20% 验证集给每域选冠军变体（纯打分，零生成） | 路由机制是否跨模型有效 | 分钟级 |
| 5 门控做题* | bare / always(路由 top-1) / gated(路由+门控) 三臂 × 4 规则域（`SELECT=1` 加跑模型自选臂） | 遮蔽伤害与"门控永不受伤"是否跨模型成立 | 2–4 h |
| 6 汇总 | 自动生成 summary.json | — | 秒级 |

\* 阶段 5 需 `TRACKB=1`；只跑 1–4 也有效（检索面结论），但强烈建议跑满。
总计 ≈ 11–22 h，全程无人值守、可随时断点重跑。

## 2. 数据集（脚本自动使用，无需手动下载）

SRA-Bench（arXiv 2604.24594）：**26,262 技能全库** + 5 域 3,970 个带标注实例
（theoremqa 747 / logicbench 760 / medcalcbench 1,100 / champ 223 / bigcodebench 1,140；做题阶段用前 4 个规则可评域）。数据随 SR-Agents 仓库分发，克隆即得（见 §4 第①步）。

## 3. 模型认领表

| TAG（运行时用） | HuggingFace ID | 家族 | 显存 | 备注 |
|---|---|---|---|---|
| `llama31-8b` | meta-llama/Llama-3.1-8B-Instruct | Meta | ~20GB | 仓库 gated，需 HF token |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.3 | Mistral | ~18GB | |
| `glm4-9b` | THUDM/glm-4-9b-chat | 智谱 | ~22GB | |
| `gemma2-9b` | google/gemma-2-9b-it | Google | ~22GB | 仓库 gated，需 HF token |
| `yi15-9b` | 01-ai/Yi-1.5-9B-Chat | 01.AI | ~22GB | |
| `qwen35-9b` | Qwen/Qwen3.5-9B | 阿里 | ~22GB | 项目方自跑；需 `NO_THINK=1` |

任何 OpenAI 兼容端点都可接入,换成你手头的模型起个新 TAG 即可。思考型模型（Qwen3/3.5、GLM-Z1 等输出 `<think>` 的）必须加 `NO_THINK=1`。

## 4. 环境准备（约 20 分钟）

```bash
# ① 代码 + 基准数据（external/ 不入库，单独克隆）
git clone https://github.com/Ludan-daye/HySkill && cd HySkill
git clone https://github.com/oneal2000/SR-Agents external/SR-Agents

# ② Python 环境（3.10+）
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -e external/SR-Agents sentence-transformers openai
.venv/bin/pytest -q        # 应为 41 passed，验证环境完好

# ③ 起模型端点（vLLM；国内加 HF_ENDPOINT 镜像）
pip install vllm
HF_ENDPOINT=https://hf-mirror.com vllm serve <HuggingFace-ID> \
  --served-model-name <TAG> --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

## 5. 一键运行

```bash
# 推荐：全菜单（检索 + 重排 + 路由 + 门控做题）
(TAG=<TAG> MODEL=<TAG> API_BASE=http://localhost:8000/v1 TRACKB=1 \
  nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)

tail -f run.log
# 阶段标记: TRACKA-VARIANTS-DONE / TRACKA-RERANK-DONE / ROUTE-DONE / TRACKB-DONE / ALL-DONE
```

可选开关：`SELECT=1`（加跑模型自选臂）、`RERANK_DOMAINS=all`（重排开满 5 域）、`WORKERS`/`INFER_WORKERS`/`RERANK_WORKERS`（默认 32/48/8，显存吃紧调低）。

要点：
- **断点续跑**：中断后重跑同一条命令即可——生成按内容寻址缓存、已完成文件自动跳过；
- **MODEL 字符串 = TAG 且全程不变**（进缓存键，中途改名等于重来）；
- `(nohup ... &)` 子壳写法保证 SSH 掉线进程不死。

## 6. 冻结参数面板（跨模型可比的前提，勿改）

| 参数 | 值 |
|---|---|
| 想象模板 / K / 温度 | sentence+passage+skill / 4 / 0.7 |
| 编码器 | all-MiniLM-L6-v2 |
| 检索 top-k / 做题 max-tokens | 50 / 2048 |
| 路由与门控标定 | 同一 20% 验证集、seed 0；路由按 val nDCG@10 argmax；门控拦截精度 ≥0.9 |
| BM25 后端 | bm25s（method=robertson，与 rank_bm25 排序一致，已单测锁定） |

全部硬编码在脚本里，正常使用碰不到。

## 7. 结果回传

跑完自动生成 **`community-results/<TAG>/summary.json`**（各域 × 各变体检索指标、路由选择及比分、三臂准确率、门控统计）。回传：fork 后把 `community-results/<TAG>/` 提 PR（推荐），或直接把文件发给维护者。原始 jsonl **不要**提交,本地留存备显著性复核。

## 8. 常见坑

| 症状 | 处理 |
|---|---|
| 预热大量 empty | 端点挂了或思考型模型忘加 `NO_THINK=1`；`curl $API_BASE/models` 自查 |
| vLLM OOM | `--gpu-memory-utilization` 降到 0.7 或 `WORKERS=16` |
| HF 下载超时 | `export HF_ENDPOINT=https://hf-mirror.com` |
| Llama/Gemma 403 | HF 网页接受协议 + `huggingface-cli login` |
| rerank 报 400/超长 | 已由插件运行时修补（max_tokens 封顶 1024），确认命令带 `--plugin hyskill.plugin` |
| logicbench gated 臂飞快 | 正常——门控大量拦截时该域退化为裸考 |

## 9. 汇总方式（维护者侧）

收齐 N 份 summary.json 后产出三张跨模型矩阵：①"模型 × 域 × 变体"检索 nDCG（验证粒度规律）；②"模型 × 域"想象最优 vs llm_rerank（验证成本-效果结论）；③"模型 × 域"bare/always/gated（验证遮蔽与门控性质）。显著性用各自留存的原始文件按 `scripts/significance.py` / `scripts/phase2_significance.py` 补算。
