# K=2 正式实验运行环境冻结表

> 日期：2026-07-24  
> 状态：`[x]` 已从本地保存的 formal answer/selection 逐行证据、服务器盘点、
> K 消融 manifest 和冻结代码复核。  
> 用途：作为 runtime-matched baseline 重跑的唯一环境参照。旧 baseline
> 文档、文件名或 endpoint 名称不能替代本表中的逐项身份。

## 1. 证据边界

K=2 formal 记录能逐题绑定并证明：

- checkpoint 来源、revision 或目录文件 manifest SHA-256；
- tokenizer 与 chat template 身份；
- served-model name；
- vLLM 版本、BF16 dtype 和 8,192-token context；
- answer/selector 代码 bundle；
- 模型、领域、arm、输入决定、请求 hash 和输出。

服务器盘点另外记录了 GPU、Python、PyTorch、Transformers 和工作目录，但这些
字段没有全部进入每条 formal answer。因此 baseline 重跑必须同时做到：

1. 严格匹配 formal 逐题身份；
2. 在新任务开始前把 GPU/driver/CUDA/Python/PyTorch/Transformers 和 endpoint
   启动命令写入新的 job-bound manifest。

## 2. 共同运行协议

| 项目 | K=2 冻结值 |
|---|---|
| 模型服务 | OpenAI-compatible vLLM endpoint |
| dtype | `bfloat16` |
| 量化 | 无 |
| context length | 8,192 |
| 答题引擎 | SR-Agents native `direct` |
| 答题参数 | temperature 0.7，`max_tokens=2048`，thinking off |
| Selector | 50 个 ordered candidates；只展示 name/description |
| Selector 参数 | temperature 0，`max_tokens=64`，thinking off |
| Selector 恢复 | 最多三次解析；仅解析失败才 deterministic rank-1 fallback |
| 工具题 | 保留原 skill tools 和 native tool loop |
| 规则域 | TheoremQA 747；LogicBench 760；MedCalc-Bench 1,100；CHAMP 223 |
| 每模型总题数 | 2,830 |
| calibration split | 每域 sorted IDs，seed 0，20% |
| held-out | 每模型 2,265 |
| gate | `p_min=0.9`；S2 sentence coverage threshold 0.6 |
| 统计 | task bootstrap 10,000，seed 0；fleet 按 model→domain→instance 重采样 |

`max_tokens=2048` 与 8K context 同时生效。因此技能注入后过长的请求会成为
preregistered context-overflow method failure，保留在分母中；不能换到 32K
endpoint 恢复，也不能过滤。

## 3. 模型与 endpoint 身份

所有 formal answer 行在各自模型内只出现一个 runtime identity。

| Result tag | Served model | Checkpoint | Files manifest SHA-256 | vLLM | Answer bundle |
|---|---|---|---|---:|---|
| `deepseek7b` | `deepseek7b` | ModelScope `deepseek-ai/deepseek-llm-7b-chat@snapshots/master` | `25b7f08040a12a38ed6a4fdca625063e18091926a30813d56a3c87e3cbe1f03b` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| `glm4-9b` | `glm4-9b` | ModelScope `ZhipuAI/glm-4-9b-chat@snapshots/master` | `cd37e55587031d4dbc51bf768f83268669e196434f209b1bd0e6245991e038be` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| `llama31-8b` | `llama31-8b` | ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct@snapshots/master` | `a8e51a9052d5cfe3faea783aa90837c6ba39d04f438eb6eca344a0f4b1e44630` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| `mistral7b` | `mistral7b` | ModelScope `LLM-Research/Mistral-7B-Instruct-v0.3@c8cfccbcfd71d4e3479498c30b2823bab19c4687` | `559840283ece7b8cbbb937d74d5ce47aff520cda4a453a3331ac3e8f26bfa6df` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| `qwen3.5-4b-reference` | `qwen3.5-4b` | HF `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `7447e4e49652e2eb494c53d808d9b4e005838b1430aecb6df8181b2105d177dc` | 0.17.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| `qwen35-9b` | `qwen35-9b` | China HF mirror `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `daf8a250ee437249688f839397f7908ed75e10eba31ab9a5663456c36c46b595` | 0.17.1 | `f796f20537fe63a484b4a302ebf3c2d5131d15aaff051404c2362be8afbe8d86` |
| `yi15-9b` | `yi15-9b` | ModelScope `01ai/Yi-1.5-9B-Chat@snapshots/master` | `45eb2167b36e6209f26a897a440cf27bf002f4b1368556d9105fbe76341addca` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |

All seven identities additionally record `dtype=bfloat16` and
`context_length=8192`.

## 4. Tokenizer 与 chat template 身份

| Model | Tokenizer identity | Chat-template identity |
|---|---|---|
| DeepSeek-7B | `tokenizer.json=a08b02921f08548065a7b2ec13b2ffeed873231add60f9c3c7b08b04f2cc212a`; `tokenizer_config=9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086` | `tokenizer_config=9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086` |
| GLM-4-9B | `tokenizer.model=5a493598071550244b2ee7f26118f3edec2150b9dfa967929a99052ac83fe716`; `tokenizer_config=f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62` | `tokenizer_config=f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62` |
| Llama-3.1-8B | `tokenizer.json=79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`; `tokenizer_config=177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424` | `tokenizer_config=177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424` |
| Mistral-7B | `tokenizer.json=60b945759e27a63c3c5c0ca675881f5a73b4aa38b5d1d6818570308d4f1a3c59`; `tokenizer.model=37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89`; `tokenizer_config=b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d` | `tokenizer_config=b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d` |
| Qwen3.5-4B | revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; `tokenizer_config=316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` | `chat_template=a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| Qwen3.5-9B | revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; `tokenizer.json=5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`; `tokenizer_config=316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` | revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; `tokenizer_config=316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| Yi-1.5-9B | `tokenizer.json=a13ccc285aea27f5e9a98d40e04e330b01d89db6de7af10b013f56eec8eae8a2`; `tokenizer.model=386c49cf943d71aa110361135338c50e38beeff0a66593480421f37b319e1a39`; `tokenizer_config=a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4` | `tokenizer_config=a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4` |

K=2 selector bundle is
`7e485106b2cb9cd6828fc61b5cae5927bbe70ea24deabf22174997f0b6015b16`
for GLM/Llama/Mistral/Qwen4 and
`e9f07a41e470c17429a6aceb3747c9cccb0b18e1ed5d7ff066a87540af218b48`
for Qwen9.

## 5. 数据与代码身份

| Artifact | Identity |
|---|---|
| SR-Agents source | commit `277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f` |
| Rule corpus | `16ee509ae5bea8c2e17167dffecd89100a7d8dfa31256c3742426758c7169b5e` |
| TheoremQA instances | `c969a7291e23361ba9f377e464be76093804deb628b964fb846c6eff6b28deeb` |
| LogicBench instances | `af5055caac041ea08cee47622f5d922b47b2ba0a8e3a60b87349599eeff1bdfe` |
| MedCalc-Bench instances | `814f6b082f56cf89dd9c50be7ab87b5bcb0e41633138951e42b38d6923c3244e` |
| CHAMP instances | `d61346716cede953afb352e739e170b96d2bfb98824edd91b8783dc3526c7cec` |
| K=2 retrieval input source | `b642012+bundle-efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f` |

K=2 imagination used `K_img=2`, temperature 0.7, the first two samples from
the frozen nested K=10 cache, the same MiniLM checkpoint recorded in each
K-ablation manifest, and ordered top-50 retrieval. Baseline reruns do not use
these imaginations, but they must use the same instances, corpus and evaluation
implementation.

## 6. 物理调度环境

| Slot | Hardware | K=2 model assignment |
|---|---|---|
| S1-0 | A100-SXM4 80GB | DeepSeek-7B → Yi-1.5-9B |
| S2-0 | A100 80GB PCIe | Qwen3.5-4B |
| S3-0 | A100-SXM4 40GB | GLM-4-9B → Mistral-7B |
| S3-1 | A100-SXM4 40GB | Llama-3.1-8B |
| N1-0 | RTX 4090，driver 报告 49,140 MiB | Qwen3.5-9B |

2026-07-23 inventory recorded:

- S1 client environment: Python 3.10.12, PyTorch 2.10.0+cu128,
  Transformers 5.13.1; service vLLM 0.19.1.
- S2 client environment: Python 3.11.13, PyTorch 2.8.0+cu128,
  Transformers 5.5.4; service environment vLLM 0.17.1,
  PyTorch 2.10.0+cu128, Transformers 4.57.6.
- S3 client environment: Python 3.10.12, PyTorch 2.11.0+cu126,
  Transformers 5.13.1; service environment vLLM 0.19.1,
  PyTorch 2.10.0+cu128.
- N1 在 staging 前只有 system Python 3.11.7；formal rows 能证明最终服务为
  vLLM 0.17.1，但最终 Python/PyTorch/Transformers 没有进入逐题身份。

这些节点是调度来源，不意味着当前仍保持相同状态。baseline 点火前必须重新做
read-only process/GPU/disk/worktree/checkpoint 检查。

## 7. 与 baseline 重跑直接相关的风险

K=2 Gate 的 S2 threshold 使用 calibration split 上的 Bare correctness。
保存的 28 个 routed gate 中有 12 个 `tau2` 非空；Qwen4 的 4 个 fixed gate
中另有 1 个 `tau2` 非空。因此旧 Bare runtime identity 未闭环不只影响
`Gated vs Bare` 比较，也可能影响 K=2 Gated 决策本身。baseline 重跑必须先
完成 fresh Bare，再重新计算所有 32 个 gate threshold 并逐题比较决策。

