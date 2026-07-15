# 多模型协作实验计划 v2（众包版）

> **目的**：现有全部结果出自单一生成模型（Qwen3.5-4B）。本计划把完整对比矩阵——**想象检索（5 变体）vs LLM 重排、粒度路由、加载门控做题**——在 5–6 个不同家族的开源模型上复跑，证明方法与模型无关。
> **形式**：每人认领一个模型 + 一张 ≥24GB 显卡，跑一条命令，回传一个几 KB 的 JSON。全流程断点续跑。
> v1→v2 变化：新增 llm_rerank 对比臂、5 变体全跑、验证集粒度路由（离线验证 5/5 域选中真冠军）、BM25 换 bm25s 后端（快约百倍，四路/两阶段不再有"BM25 税"）。

## 1. 跑什么：一条命令里的五个阶段（内部三层并行）

| 阶段 | 内容 | 并行方式 | 墙钟（4090 级） |
|---|---|---|---|
| 1 预热 **∥** 重排 | 预热：3 模板 × 3,970 题 × K=4（32 并发）；**重排流与预热同时跑**（重排不需要想象，只吃端点：fast_bm25 出候选 → llm_rerank，默认 3 争议域） | 预热 32 线程 + 重排 8 线程共打端点 | max(4–8h, 2.5–6h) ≈ **4–8 h** |
| 2 想象检索 | 5 变体 × 5 域 | **5 个域并行流**（生成全缓存命中，只剩向量计算） | ~**1 h**（=最慢的域） |
| 3 粒度路由 | 20% 验证集选各域冠军（纯打分） | — | 分钟级 |
| 4 门控做题* | **6 臂** × 4 规则域：bare / always(路由) / gated(路由+门控) / select(同源消融) / **always_rerank(原装重排基线)** / **select_bm25(原装自选基线)** | **4 个域并行流**（域内按依赖串行），每流 24 并发 | ~**3–5 h** |
| 5 汇总 | summary.json（含成本审计块） | — | 秒级 |

\* 需 `TRACKB=1`；只跑 1–3 也有效（检索面结论），但强烈建议跑满。
**总墙钟 ≈ 7–12 h**（v2 串行版为 11–22h），全程无人值守、断点续跑；各并行流的日志在 `results/multimodel/<TAG>/logs/`。

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

# ②.5 【必做】验证 torch 能用 GPU——cu13 版 torch 在 CUDA 12.x 驱动上会静默回落 CPU，
#      嵌入慢 10 倍以上且极易被 OOM-killer 杀（我们在两台 A100 上都踩过）：
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用！装 cu12x 版：pip install torch==2.11.0+cu126 -f https://mirrors.aliyun.com/pytorch-wheels/cu126/'"

# ③ 起模型端点（vLLM；国内加 HF_ENDPOINT 镜像）
pip install vllm
HF_ENDPOINT=https://hf-mirror.com vllm serve <HuggingFace-ID> \
  --served-model-name <TAG> --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

## 5. 一键运行

```bash
# 推荐：全菜单（检索 + 重排 + 路由 + 门控做题）
(TAG=<TAG> MODEL=<TAG> API_BASE=http://localhost:8000/v1 TRACKB=1 SELECT=1 RERANK_DOMAINS=all \
  nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)

tail -f run.log
# 阶段标记: TRACKA-VARIANTS-DONE / TRACKA-RERANK-DONE / ROUTE-DONE / TRACKB-DONE / ALL-DONE
```

调参开关：`WORKERS`/`INFER_WORKERS`/`RERANK_WORKERS`（默认 32/48/8，显存吃紧调低）。

**对比臂规则（2026-07-15 起，select 与 rerank 上下文允许即必跑）**：上下文 ≥8K 的模型一律 `RERANK_DOMAINS=all SELECT=1`（重排五域全跑 + 自选臂全跑）；4K 上下文模型两臂都跳过（`RERANK_DOMAINS=""`、不加 SELECT——50 候选 prompt 物理塞不进，硬跑会 400），并在回传备注里写明原因。

**基线不混搭规则（2026-07-15 起，脚本已自动执行）**：基线必须整栈原装。脚本会自动跑齐两个纯原装基线臂——`always_rerank`（bm25→重排→top-1 直装）与 `select_bm25`（bm25 候选→官方自选）；`select` 臂（我们的检索 + 官方自选）是**同源消融**，只用于隔离装载层，回传与引用时严禁当基线。正版/被取代版对照表见 `docs/05-results.md` §5.5。

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

**补充规则：成本随行。** 汇总脚本会从你这次运行的真实产物里实测每方法的 token 成本（各想象模板的 prompt/生成长度 × K、重排的 50 候选长 prompt；估算器统一为 chars/3.8），自动写进 summary.json 的 `cost` 块。**任何进入论文的方法对比必须携带同批次实测成本**——效果和账单永远一起报，禁止只报精度不报开销。

## 7. 结果回传（三步）

**① 自动汇总**：跑完 `run_multimodel.sh` 即有 `community-results/<TAG>/summary.json`（各域 × 各变体检索指标、路由选择及比分、三臂准确率、门控统计、成本审计）。

**② 生成分析数据包**（一条命令）：
```bash
.venv/bin/python scripts/export_analysis_pack.py <TAG> <TAG>
```
在 `community-results/<TAG>/` 下生成四个可入库的小文件：`retrieval_top10.jsonl.gz`（每题×每变体的金标+top10+逐题 nDCG）、`gating_per_instance.jsonl.gz`（每题 S1/S2/τ/拦截/**全部六臂**对错——含 always_rerank 与 select_bm25 两个原装基线臂）、`imagination_samples.jsonl.gz`（每域固定 10 题的想象原文,3 模板×K=4,全模型同题可比）、`MANIFEST.md`。

**②.5 完整 top-50 榜单**（2026-07-15 起入库要求）：
```bash
.venv/bin/python scripts/export_top50.py <TAG>
```
生成 `retrieval_top50.jsonl.gz`（每题×每方法的全量 50 深榜单，[skill_id, 分数, 是否金标] 三元组；书名按 id 从语料库反查，不重复存，单模型约 6–8MB）。

**②.6 装载数据**（2026-07-15 起入库要求——**装载数据与得分数据必须全部上传**）：
```bash
.venv/bin/python scripts/export_loading.py <TAG>
```
生成 `loading_per_instance.jsonl.gz`（每题 × 每臂：实际装进上下文的技能 id 列表、金标、是否命中；bare 与门控拦截题的 loaded 为空——空本身就是装载决策记录）。约 100KB/模型。

**③ 提交**：fork 本仓库,把整个 `community-results/<TAG>/` 文件夹提 PR（推荐）,或打包发维护者。你的 TAG 文件夹里已有 README 写明每个文件的意义与状态,照着核对。原始大件（做题 jsonl、日志、想象缓存、检索原始 json）**不要**提交,本地留存备复核——top-50 榜单请用 ②.5 的压缩三元组格式入库,不要提交原始检索 json。

## 8. 常见坑

| 症状 | 处理 |
|---|---|
| 预热大量 empty | 端点挂了或思考型模型忘加 `NO_THINK=1`；`curl $API_BASE/models` 自查；**跑完核对日志里 `WARMUP-DONE jobs=N empty=M` 的 empty 数，不能只看标记** |
| 嵌入奇慢 / 变体阶段进程被杀 | 十有八九是 torch 静默回落 CPU（cu13 wheel 配 12.x 驱动）——跑 §4 ②.5 的断言自查；装 cu12x 版修复 |
| vLLM OOM | `--gpu-memory-utilization` 降到 0.7 或 `WORKERS=16`；40G 卡上 vLLM 0.85 + 多路嵌入流不可共存，0.70 是配方 |
| vLLM 起不来："Engine core initialization failed" | 先查 `nvidia-smi` 有无**孤儿 EngineCore 进程**占着显存（上一代死亡实例的残留），按 PID 杀掉再启动 |
| vLLM JIT "compilation terminated" | `apt install build-essential python3.10-dev` + 启动加 `--enforce-eager` |
| HF 下载超时 / Xet 报 AccessDenied | `export HF_ENDPOINT=https://hf-mirror.com`；仍不行换 ModelScope（`modelscope download`，路径用 `.../snapshots/master`） |
| Llama/Gemma 403 | HF 网页接受协议 + `huggingface-cli login`；或 ModelScope 上找 LLM-Research 等非 gated 镜像仓 |
| rerank 报 400/超长 | 已由插件运行时修补（max_tokens 封顶 1024），确认命令带 `--plugin hyskill.plugin`；**4K 上下文模型请直接跳过重排与自选臂** |
| always_rerank 少量 400 | 已知现象并请如实回传：rerank 偏好长技能，全文+题目偶超 8K（我们实测 6–7%），这些题按错计——本身是"装载税"数据 |
| 双卡跑双模型互相饿死 | 每模型 `CUDA_VISIBLE_DEVICES` 绑独立 GPU（嵌入进程默认全挤 cuda:0） |
| 远程后台启动"没反应" | **点火必须回查**：launch 后 sleep 10 再 `pgrep`/看日志有新行；SSH 抖动会让启动静默失败 |
| logicbench gated 臂飞快 | 正常——门控大量拦截时该域退化为裸考 |

## 9. 汇总方式（维护者侧）

收齐 N 份 summary.json 后产出三张跨模型矩阵：①"模型 × 域 × 变体"检索 nDCG（验证粒度规律）；②"模型 × 域"想象最优 vs llm_rerank（验证成本-效果结论）；③"模型 × 域"bare/always/gated（验证遮蔽与门控性质）。显著性用各自留存的原始文件按 `scripts/significance.py` / `scripts/phase2_significance.py` 补算。
