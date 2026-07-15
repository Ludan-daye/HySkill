# yi15-9b 数据档案

**模型**：01ai/Yi-1.5-9B-Chat（零一万物，ModelScope 源）｜ 跑批机：2×A100-40G 服务器 B 的 GPU1（端口 8002）｜ 特殊配置：**上下文 4K → 重排臂跳过**（同 deepseek7b）；vllm 0.19.1 + enforce-eager；重跑段 CUDA_VISIBLE_DEVICES=1 绑卡 + WORKERS=16。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 菜单汇总（检索/路由/门控/成本；无重排块） | ⏳ 跑批中 |

## 留在跑批服务器不入库的原始件

位置 `/root/HySkill/results/multimodel/yi15-9b/`：检索榜单 ×25、routed/signals/taus/gated、做题 jsonl+eval、logs/。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「yi15-9b」列
- 运行备注：4K 长尾说明同 deepseek7b；嵌入进程绑 GPU1 是双模型同机的标准姿势（docs/08 坑表）
