# HySkill K=2 实验服务器实测清单

> 实测时间：2026-07-23 17:04--17:18（Asia/Shanghai）  
> 适用计划：[K=2 统一装载与消融实验计划](plans/2026-07-23-k2-unified-loading-and-ablation.md)  
> 状态：已确认 4 个可登录物理节点、5 个 GPU 槽；第 5 个物理节点入口尚未提供。

## 1. 登录与安全边界

四个已确认节点均使用 SSH 密码交互认证。本文只保存完整登录命令格式，不保存
口令、token、私有 endpoint 或任何可复用凭据。

| 节点 | 登录命令 | 认证 | 2026-07-23 实测 |
|---|---|---|---|
| S1 | `ssh root@180.127.11.169 -p 32940` | password，交互输入 | 成功 |
| S2 | `ssh vicuna@8.138.30.52 -p 6007` | password，交互输入 | 成功 |
| S3 | `ssh root@180.127.11.167 -p 22624` | password，交互输入 | 成功 |
| N1 | `ssh root@180.127.11.167 -p 27244` | password，交互输入 | 成功 |

本机旧 SSH 配置中还存在
`ssh root@180.127.11.169 -p 25720`，但本次连接在认证前即被远端关闭。
该入口视为失效历史记录，不是 N2，也不得进入调度。若要求 5 个物理服务器，
仍需提供第 5 个可用 IP、端口、用户名和认证方式。

## 2. 物理节点总览

硬件数值均来自本轮 `lscpu`、`free`、`df` 和 `nvidia-smi`，不是计划估计。

| 节点 | 主机名 / 用户 | GPU | CPU | RAM | 主要可用磁盘 | 当前 GPU 状态 |
|---|---|---|---|---:|---:|---|
| S1 | `ubuntu22` / `root` | 1× NVIDIA A100-SXM4-80GB | 8 vCPU，Xeon Platinum 8358P | 15 GiB | 根盘 69 GiB | 空闲，14 MiB |
| S2 | `user-SYS-7049GP-TRT` / `vicuna` | 1× NVIDIA A100 80GB PCIe | 64 vCPU，2× Xeon Gold 5218 | 502 GiB | 根盘 1.1 TiB；数据盘 3.7 TiB | 空闲，18 MiB |
| S3 | `ubuntu22` / `root` | 2× NVIDIA A100-SXM4-40GB | 16 vCPU，Xeon Platinum 8575C | 31 GiB | 根盘 33 GiB | GPU 0 空闲；GPU 1 被 Llama endpoint 占用约 32.8 GiB |
| N1 | `ubuntu22` / `root` | 1× NVIDIA GeForce RTX 4090，驱动报告 49,140 MiB | 8 vCPU，Xeon Platinum 8575C | 15 GiB | 根盘 167 GiB | 空闲，12 MiB |

N1 的型号和显存按驱动原样记录。它不是计划中假定的 A100/L40S；投入正式
Select 长提示前必须先通过 BF16、上下文长度、OOM 和吞吐 pilot，不能根据显存
数值直接视为等价替代。

## 3. 当前 5 个 GPU 槽与任务分配

当前可调度资源是 4 个物理节点上的 5 个 GPU 槽，不是计划原先假定的 6 槽。

| GPU 槽 | 物理节点 | 当前状态 | 冻结任务 |
|---|---|---|---|
| S1-0 | S1 A100 80GB | 空闲 | DeepSeek-7B → Yi-1.5-9B；Gate、Always、Gated 顺序运行 |
| S2-0 | S2 A100 80GB | 空闲 | Qwen3.5-4B；Gate、Select、Always、Gated、fixed 组件消融 |
| S3-0 | S3 A100 40GB | 空闲 | GLM-4-9B；Gate、Select、Always、Gated |
| S3-1 | S3 A100 40GB | Llama 服务仍在运行 | Llama-3.1-8B；Gate、Select、Always、Gated |
| N1-0 | N1 RTX 4090，驱动报告约 48 GiB | 空闲、未部署 | 通过 pilot 后承载 Qwen3.5-9B |

Mistral-7B 当前没有独立 N2。未获得第 5 个物理入口前，最快的无重叠调度是：

1. N1 pilot 通过后运行 Qwen3.5-9B；
2. S3-0 完成 GLM 后切换 Mistral-7B；
3. S3-1 保留给 Llama，避免 GLM、Llama、Mistral 三模型争用同一卡；
4. 不允许两个节点重复提交同一
   `model-domain-arm-instance`。

这会把 Mistral 变成一个串行尾部；获得并验收真正的 N2 后，才可恢复每个
Select 合格模型各占一个独立 GPU 槽的同波并行。

## 4. S1：DeepSeek-7B 与 Yi-1.5-9B

### 4.1 项目与运行时

| 项目 | 实测值 |
|---|---|
| 历史主工作树 | `/root/HySkill` |
| 历史主工作树 HEAD | `291db6e92151b6013745c1e29a2470e05bdc431a`，`main...origin/main` |
| K 消融隔离树 | `/root/HySkill-k-run-20260723` |
| K 消融 HEAD | `b642012bd86ca098f828a0c06f6f926375d74b7f`，detached HEAD |
| K 消融 Python | 3.10.12 |
| K 消融运行时 | PyTorch `2.10.0+cu128`，Transformers `5.13.1` |
| 模型服务环境 | `/root/vllmenv`，vLLM `0.19.1` |

两个工作树均有历史未提交代码或实验产物。不得执行 reset、clean、覆盖式 pull
或在旧结果目录原地重跑。

### 4.2 模型

| 模型 | 已有 checkpoint 路径 | 当前可追溯身份 |
|---|---|---|
| DeepSeek-7B | `/root/.cache/modelscope/models/deepseek-ai--deepseek-llm-7b-chat/snapshots/master` | ModelScope `snapshots/master`；另有 HF snapshot `afbda8b347ec881666061fa67447046fc5164ec8`，不能未经哈希证明就视为同一副本 |
| Yi-1.5-9B | `/root/.cache/modelscope/models/01ai--Yi-1.5-9B-Chat/snapshots/master` | ModelScope `snapshots/master`；另有 HF snapshot `1a0fc698cf883c4f5c325f026ca79f0ebd9955a5`，不能未经哈希证明就视为同一副本 |

正式复用旧回答前，必须冻结实际 ModelScope 目录的 weights、config、tokenizer
和 chat template 哈希；`master` 名称本身不是不可变 revision。

### 4.3 K=2 原始输入与历史 K=4 回答

| 内容 | 路径 / 完整性 |
|---|---|
| DeepSeek K=2 fixed top-50 | `/root/HySkill-k-run-20260723/results/k-ablation/deepseek7b/k2/`；25 JSON |
| DeepSeek K=2 routed top-50 | `/root/HySkill-k-run-20260723/results/k-ablation/deepseek7b/routed/k2/`；5 JSON |
| Yi K=2 fixed top-50 | `/root/HySkill-k-run-20260723/results/k-ablation/yi15-9b/k2/`；25 JSON |
| Yi K=2 routed top-50 | `/root/HySkill-k-run-20260723/results/k-ablation/yi15-9b/routed/k2/`；5 JSON |
| 共享想象缓存 | `/root/HySkill/results/hyp_cache/`；238,080 文件；K 消融树路径解析到同一目录 |
| DeepSeek 历史回答 | `/root/HySkill/results/multimodel/deepseek7b/` |
| Yi 历史回答 | `/root/HySkill/results/multimodel/yi15-9b/` |

DeepSeek 和 Yi 的历史 `bare`、`always`、`gated` 均为 4 个规则域、每臂
2,830 行。两模型没有 Select、BM25+Select 或 Rerank 原始回答；其 50 候选提示
超过 4K 上下文，应继续标为 unavailable。

本轮盘点时 S1 没有 vLLM 或 HySkill 实验进程。

## 5. S2：Qwen3.5-4B 与 Qwen3.5-9B

### 5.1 项目与运行时

| 项目 | 实测值 |
|---|---|
| 历史主工作树 | `/home/vicuna/ludan/HySkill` |
| 历史主工作树 HEAD | `24a31274b6b1df42eea54ba87a151bdcdae233c0`，`main...origin/main` |
| K 消融隔离树 | `/home/vicuna/ludan/HySkill-k-run-20260723` |
| K 消融 HEAD | `b642012bd86ca098f828a0c06f6f926375d74b7f`，detached HEAD |
| K 消融 Python | 3.11.13 |
| K 消融运行时 | PyTorch `2.8.0+cu128`，Transformers `5.5.4` |
| 历史成功 Qwen 服务 | `/home/vicuna/anaconda3/envs/api`：vLLM `0.17.1`、PyTorch `2.10.0+cu128`、Transformers `4.57.6`、BF16、8K context；日志明确记录精确 HF snapshot |

历史主工作树和 K 消融树均有未提交代码或产物，必须只读保留。S2 本轮盘点时
没有活跃模型服务或实验进程。

### 5.2 模型

| 模型 | 精确 checkpoint |
|---|---|
| Qwen3.5-4B | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Qwen3.5-9B | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

### 5.3 K=2 原始输入与历史 K=4 回答

| 内容 | 路径 / 完整性 |
|---|---|
| Qwen4B K=2 fixed top-50 | `/home/vicuna/ludan/HySkill-k-run-20260723/results/k-ablation/qwen3.5-4b-reference/k2/`；25 JSON |
| Qwen4B K=2 routed top-50 | `/home/vicuna/ludan/HySkill-k-run-20260723/results/k-ablation/qwen3.5-4b-reference/routed/k2/`；5 JSON |
| Qwen9B K=2 fixed top-50 | `/home/vicuna/ludan/HySkill-k-run-20260723/results/k-ablation/qwen35-9b/k2/`；25 JSON |
| Qwen9B K=2 routed top-50 | `/home/vicuna/ludan/HySkill-k-run-20260723/results/k-ablation/qwen35-9b/routed/k2/`；5 JSON |
| 共享想象缓存 | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/HySkill/results/hyp_cache/`；238,080 文件 |
| Qwen9B 历史回答 | `/home/vicuna/ludan/HySkill/results/multimodel/qwen35-9b/` |
| Qwen4B routed/fixed 历史回答 | `/home/vicuna/ludan/HySkill/results/phase2/` |
| Qwen4B native baselines | `/home/vicuna/ludan/HySkill/results/multimodel/qwen35-4b-baselines/` |

Qwen9B 的 `bare`、`always`、`gated`、HySkill `select`、`select_bm25` 和
`always_rerank` 均为 4 文件、每臂 2,830 行。Qwen4B `phase2` 中的
`bare`、`always`、`gated`、`select`、`always_r`、`gated_r` 也各有
2,830 行；native `select_bm25` 与 `always_rerank` 各有 2,830 行，位于单独
baseline 目录。

## 6. S3：GLM、Llama 与 Mistral

### 6.1 项目与运行时

| 项目 | 实测值 |
|---|---|
| 历史主工作树 | `/root/HySkill` |
| 历史主工作树 HEAD | `291db6e92151b6013745c1e29a2470e05bdc431a`，`main...origin/main` |
| K 消融隔离树 | `/root/HySkill-k-run-20260723` |
| K 消融 HEAD | `b642012bd86ca098f828a0c06f6f926375d74b7f`，detached HEAD |
| K 消融 Python | 3.10.12 |
| K 消融运行时 | PyTorch `2.11.0+cu126`，Transformers `5.13.1` |
| 模型服务环境 | `/root/vllmenv`，vLLM `0.19.1`、PyTorch `2.10.0+cu128` |

两个工作树均有未提交历史修改或实验产物。根盘只剩 33 GiB，部署前必须核算
新增代码、日志和结果空间；不得在该盘重复下载已有模型。

### 6.2 模型

| 模型 | 已有 checkpoint 路径 | 当前可追溯身份 |
|---|---|---|
| GLM-4-9B | `/root/.cache/modelscope/models/ZhipuAI--glm-4-9b-chat/snapshots/master` | ModelScope `snapshots/master`；另有 HF snapshot `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73`，不能直接等同 |
| Llama-3.1-8B | `/root/.cache/modelscope/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master` | ModelScope `snapshots/master`，需冻结目录哈希 |
| Mistral-7B | `/root/.cache/modelscope/models/LLM-Research--Mistral-7B-Instruct-v0.3/snapshots/c8cfccbcfd71d4e3479498c30b2823bab19c4687` | 精确 ModelScope revision `c8cfccbcfd71d4e3479498c30b2823bab19c4687` |

本轮盘点时 GPU 1 上仍运行：

```text
vllm serve .../Meta-Llama-3.1-8B-Instruct/snapshots/master
--port 8001 --max-model-len 8192 --served-model-name llama31-8b
```

该服务已持续约 29 小时，占用约 32.8 GiB。除非确认它属于当前计划并可直接
复用，否则不能结束、替换或把其他模型部署到 GPU 1。

### 6.3 K=2 原始输入与历史 K=4 回答

| 模型 | K=2 fixed | K=2 routed | 历史 K=4 回答 |
|---|---|---|---|
| GLM | `/root/HySkill-k-run-20260723/results/k-ablation/glm4-9b/k2/`；25 JSON | `/root/HySkill-k-run-20260723/results/k-ablation/glm4-9b/routed/k2/`；5 JSON | `/root/HySkill/results/multimodel/glm4-9b/` |
| Llama | `/root/HySkill-k-run-20260723/results/k-ablation/llama31-8b/k2/`；25 JSON | `/root/HySkill-k-run-20260723/results/k-ablation/llama31-8b/routed/k2/`；5 JSON | `/root/HySkill/results/multimodel/llama31-8b/` |
| Mistral | `/root/HySkill-k-run-20260723/results/k-ablation/mistral7b/k2/`；25 JSON | `/root/HySkill-k-run-20260723/results/k-ablation/mistral7b/routed/k2/`；5 JSON | S3 上不存在 `/root/HySkill/results/multimodel/mistral7b/` |

GLM 和 Llama 的 `bare`、`always`、`gated`、HySkill `select`、
`select_bm25`、`always_rerank` 均为 4 文件、每臂 2,830 行。Mistral 的公开
compact 包已在仓库中，但 S3 上没有历史 raw answer 目录；不得从 compact
汇总反造 raw K=4 回答，迁移或复用前需找到其原始归档来源。

S3 的共享想象缓存为 `/root/HySkill/results/hyp_cache/`，共 357,120 文件；
K 消融树中的缓存路径解析到同一目录。

## 7. N1：新增 Qwen3.5-9B 候选节点

N1 是空白计算节点：

- 尚无 HySkill 工作树、SR-Agents、checkpoint 或 vLLM；
- 系统 Python 为 3.11.7，只有基础 Anaconda；
- GPU 空闲，根盘可用 167 GiB；
- 本轮没有模型或实验进程。

因此 N1 不能直接开始正式实验。部署顺序必须是：

1. 建立隔离项目目录并记录代码 bundle SHA；
2. 从中国镜像下载精确
   `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`，禁止跟随
   浮动 `main`；
3. 建立项目级环境并冻结 Python、vLLM、PyTorch、Transformers 版本；
4. 只传 K=2 routed/gated、规则域 instances/corpus、必要旧基线和后续
   selection 缓存，不传完整想象缓存；
5. 核对传输前后 SHA-256；
6. 先跑 200 题 Select 与 direct-answer pilot，确认 BF16、8K 上下文、无 OOM、
   unresolved error 为零后再加入正式调度。

### 7.1 下载镜像策略

所有节点的外部下载统一走中国镜像，不仅限于 N1：

- Hugging Face checkpoint 使用已记录的中国镜像端点；ModelScope 使用国内
  服务；
- pip、Conda、APT 和容器镜像源均切换到中国镜像，并在部署 manifest 中记录
  实际 URL；
- 禁止静默回退海外源；国内镜像不可用时停止任务并报告；
- 镜像来源不替代身份验证：repo、revision、tokenizer、config、chat template
  和权重仍须逐项冻结或校验哈希；
- 已有本地 checkpoint 优先复用，服务器间传输前后验证 SHA-256，不重复下载。

## 8. 输入身份与目录保护

七模型 K=2 retrieval 输入的共同代码身份是：

```text
b642012+bundle-efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f
```

本轮实测验证每个模型均有 5 域 × 5 fixed JSON 和 5 域 routed JSON，即：

```text
7 models × (25 fixed + 5 routed) = 210 K=2 raw result files
```

执行约束：

- `results/k-ablation/<tag>/k2/` 与 `routed/k2/` 是只读输入；
- 新结果只写独立 `results/k2-main/<tag>/`，绝不覆盖 K=4；
- 服务器 raw K=4 回答在完成 request-hash 审计前只读保留；
- GitHub 公开包使用 `<tag>/k2/` 与 `<tag>/k4/` 隔离，K=4 迁移前后逐文件
  SHA-256 必须一致；
- K 无关 native baselines 与 `k-ablation/` 不塞入 K=2 或 K=4 目录；
- 任何 `snapshots/master` 模型在实际 checkpoint 哈希冻结前不得进入回答复用池。

## 9. 盘点状态

| ID | 状态 | 任务 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|
| INFRA001 | `[x]` | 盘点既有 S1/S2/S3 | 登录、硬件、项目、模型、K2/K4 路径和进程均实测 | 3 节点、4 GPU 槽已核对 | 2026-07-23 |
| INFRA002 | `[x]` | 盘点新增 N1 | 登录、硬件、磁盘和空白环境均实测 | 1 节点、1 GPU 槽已核对 | 2026-07-23 |
| INFRA003 | `[!]` | 获得第 5 个物理节点 N2 | 提供可登录入口并完成同级盘点 | 未提供；旧 `25720` 入口认证前关闭 | 2026-07-23 |
| INFRA004 | `[~]` | 冻结浮动 ModelScope checkpoint | weights/config/tokenizer/chat-template 哈希写入 runtime manifest | DeepSeek、Yi、GLM、Llama 尚待执行 | 2026-07-23 |
| INFRA005 | `[~]` | 执行中国镜像门禁 | 所有外部下载记录国内镜像 URL，且无海外回退 | 约束已同步到部署和 runner 任务；等待 N1 manifest 验证 | 2026-07-23 |
