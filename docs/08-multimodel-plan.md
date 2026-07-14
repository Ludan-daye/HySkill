# 多模型协作实验计划（众包版）

> **目的**：现有全部结果出自单一生成模型（Qwen3.5-4B），这是投稿最大软肋——审稿人会问"这是 HySkill 的性质还是 Qwen 的性质？"。本计划把想象检索（Track A）和加载门控（Track B）在 **5–6 个不同家族的开源模型**上复跑，证明方法与模型无关。
> **形式**：每人认领一个模型 + 一张 ≥24GB 显卡，跑一条命令，回传一个小 JSON。全流程断点续跑，中断重启无损失。

## 1. 两条轨道：每个模型跑什么

| 轨道 | 内容 | 产出 | 单卡耗时（4090 级） |
|---|---|---|---|
| **Track A（必跑）想象检索** | 用认领模型给 3,970 个任务各想象 4 份技能文档（15,880 次短生成，多线程），然后在 26,262 技能全库上跑 naive_skill 检索 × 5 域 | 每域 Recall@1/5/10/50 + nDCG@10 | 预热 2–4 h + 检索 ~1 h |
| **Track B（选跑）加载门控** | 同一模型兼任做题模型：裸考 → 用它自己的想象算 S1/S2 信号 → 保守标定 → always / gated 两臂做题（4 规则域 × 2,830 题 × 3 臂，48 线程并发） | 每域 bare/always/gated 准确率 + 门控统计 | 3–5 h |

判读标准（写进论文的口径）：
- Track A 复现成功 = 该模型的 naive_skill 在多数域仍显著高于冻结基线（hybrid 五域均值 0.456，各域基线数字见 `docs/05-results.md` §2，**不需要重跑基线**——BM25/dense/hybrid 与生成模型无关）；
- Track B 复现成功 = gated ≥ always 的"永不受伤"性质在该模型上保持。

## 2. 模型认领表

| TAG（运行时用） | HuggingFace ID | 家族 | 显存需求 | 备注 |
|---|---|---|---|---|
| `llama31-8b` | meta-llama/Llama-3.1-8B-Instruct | Meta | ~20GB | 仓库 gated，需 HF token |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.3 | Mistral | ~18GB | |
| `glm4-9b` | THUDM/glm-4-9b-chat | 智谱 | ~22GB | |
| `gemma2-9b` | google/gemma-2-9b-it | Google | ~22GB | 仓库 gated，需 HF token |
| `yi15-9b` | 01-ai/Yi-1.5-9B-Chat | 01.AI | ~22GB | |
| `qwen35-9b` | Qwen/Qwen3.5-9B | 阿里 | ~22GB | **我们自己服务器跑**（规模阶梯），需 `NO_THINK=1` |

任何 OpenAI 兼容端点都能接入——上表只是建议；换成你手头已有的模型也行，起个新 TAG 即可。思考型模型（Qwen3/3.5、GLM-Z1 等带 `<think>` 的）必须加 `NO_THINK=1`。

## 3. 环境准备（约 20 分钟）

```bash
# ① 代码（external/ 不入库，需单独克隆 SR-Agents 基准）
git clone https://github.com/Ludan-daye/HySkill && cd HySkill
git clone https://github.com/oneal2000/SR-Agents external/SR-Agents
#    （版本用 master 最新即可；已知的 llm_rerank max_tokens 上游 bug
#      由我们插件在运行时修补，不依赖特定版本）

# ② Python 环境（3.10+）
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -e external/SR-Agents sentence-transformers openai
.venv/bin/pytest -q        # 应为 34 passed，验证环境完好

# ③ 起模型端点（vLLM，独立 conda/venv 皆可；国内加 HF_ENDPOINT 镜像）
pip install vllm
HF_ENDPOINT=https://hf-mirror.com vllm serve <HuggingFace-ID> \
  --served-model-name <TAG> --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
# 编码器 MiniLM(~90MB) 首次运行自动下载；国内同样可用 HF_ENDPOINT 镜像
```

## 4. 一键运行（多线程内置）

```bash
# Track A（必跑）
(TAG=<TAG> MODEL=<TAG> API_BASE=http://localhost:8000/v1 \
  nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)

# Track A + Track B
(TAG=<TAG> MODEL=<TAG> API_BASE=http://localhost:8000/v1 TRACKB=1 \
  nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)

tail -f run.log     # 阶段标记：TRACKA-DONE / TRACKB-DONE / ALL-DONE
```

要点：
- **并发**：生成阶段 `WORKERS=32` 线程、做题阶段 `INFER_WORKERS=48` 线程同时打端点，vLLM 连续批处理会把 GPU 打满；显存吃紧就调低这两个值和 `--gpu-memory-utilization`；
- **断点续跑**：直接重跑同一条命令——已生成的想象在 `results/hyp_cache` 按内容寻址缓存、已完成的检索/做题文件自动跳过，重启零浪费；
- **MODEL 字符串 = TAG 且全程不变**（它进缓存键，中途改名等于全部重来）;
- SSH 掉线安全：命令模板里的 `(nohup ... &)` 子壳写法保证进程不随会话退出。

## 5. 冻结参数面板（保证跨模型可比，勿改）

| 参数 | 值 |
|---|---|
| 想象模板 / 采样数 / 温度 | skill（完整 SKILL.md）/ K=4 / 0.7 |
| 编码器 | sentence-transformers/all-MiniLM-L6-v2 |
| 检索 top-k / 做题 max-tokens | 50 / 2048 |
| 门控标定 | 20% 验证集、拦截精度 ≥0.9、seed 0（gate.py 默认） |

以上全部已硬编码在 `scripts/run_multimodel.sh`，正常使用不会碰到。

## 6. 结果回传

跑完后脚本自动生成 **`community-results/<TAG>/summary.json`**（几 KB：各域检索指标 + 三臂准确率 + 门控统计）。回传方式二选一：

1. **提 PR**（推荐）：fork 本仓库，把 `community-results/<TAG>/` 提交上来；
2. 直接把 summary.json 发给项目维护者。

大文件（results/multimodel/ 下的原始 jsonl）**不要**提交，先在本地留存，万一需要显著性复核再传。

## 7. 常见坑

| 症状 | 处理 |
|---|---|
| 生成阶段大量 `empty` | 端点挂了或模板不兼容：`curl $API_BASE/models` 检查；思考型模型忘加 `NO_THINK=1` |
| vLLM OOM | 降 `--gpu-memory-utilization` 到 0.7 或 `WORKERS=16` |
| HF 下载超时 | `export HF_ENDPOINT=https://hf-mirror.com` |
| Llama/Gemma 403 | 先在 HF 网页接受协议，`huggingface-cli login` |
| logicbench gated 跑得飞快 | 正常——门控大量拦截时该域退化为裸考，是预期行为 |

## 8. 汇总方式（维护者侧）

收齐 N 个 summary.json 后：主表按"模型 × 域"矩阵呈现 nDCG@10 与 gated−always 差值；论文主张从"Qwen 上成立"升级为"跨 N 个家族成立"。显著性复核用各自留存的原始文件按 `scripts/significance.py`（检索）与 `scripts/phase2_significance.py` 逻辑（做题）补算。
