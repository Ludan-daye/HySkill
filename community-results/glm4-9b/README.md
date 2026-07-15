# glm4-9b 数据档案

**模型**：ZhipuAI/glm-4-9b-chat（智谱，ModelScope 源）｜ 跑批机：A100-80G 服务器 A（端口 8002，与 llama31-8b 同卡分显存）｜ 特殊配置：无 no_think；重排臂已跑（长上下文）；vllm 0.19.1 + enforce-eager。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 全菜单汇总（检索/路由/门控/成本，格式同 qwen35-9b） | ⏳ 跑批中，ALL-DONE 后由巡检自动收录 |

## 留在跑批服务器不入库的原始件

位置 `/root/HySkill/results/multimodel/glm4-9b/`：检索榜单 ×25、routed/signals/taus/gated、做题 jsonl+eval、logs/。想象缓存在 `/root/HySkill/results/hyp_cache`。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「glm4-9b」列
- 运行备注：预热 0 空生成；曾因 corpus.json.zip 未解压导致检索段返工一次（预热资产无损，不影响数据有效性）
