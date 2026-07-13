# HySkill：查询侧假想技能生成用于技能检索与加载

> 把 HyDE（ACL 2023）的"假想文档嵌入"迁移到 LLM 智能体的技能（SKILL.md）检索上：
> **给定任务，先让模型生成一份"假想技能文档"，用它的嵌入去技能库检索真技能，并用假想↔真实的差距做加载门控。**

## 状态（2026-07-13）

- [x] 文献调研与新颖性排查完成（结论：查询侧 HyDE on skills 无人做，见 `docs/background.md` 第 5 节）
- [x] 研究方向确定：全面转向本 idea（由"复杂技能导致性能退化"方向转来）
- [ ] 方法方案待定：HySkill-R（纯检索）/ HySkill-RG（+门控，推荐）/ HySkill-RGF（+精炼），见 `docs/idea.md`
- [ ] 设计文档、实现计划、预实验

## 仓库结构

```
docs/
  background.md          研究背景：退化现象、加载方法版图、顶会接收状态、HyDE 谱系与新颖性排查
  idea.md                研究 idea：动机链条、方法三方案、新颖性防御、评测计划、风险
  hyde-method.md         HyDE 原文方法精读：相似度机制、公式链、消融、到 HySkill 的映射
  experiment-design.md   实验思路：SRA-Bench 的做法、我们借鉴什么、两个实验的设计与判据
  superpowers/           正式设计文档与实现计划
hyskill/                 方法实现（SR-Agents 插件：解析/生成/嵌入/融合/检索/门控）
scripts/                 smoke.sh 冒烟自检；run_phase0.sh 批跑；analyze.py 汇总表
tests/                   21 个单元测试 + 微型语料
```

## 关键结论速览

1. "技能会帮倒忙"已被证实（More Skills, Worse Agents：202 技能下降 21%），**主因是选错技能（skill shadowing），不是上下文变长**。
2. 选错源于"查询↔技能"的语义鸿沟（能力匹配 ≠ 语义相似）。HyDE 桥接这类鸿沟在文档（ACL 2023）和 API 工具（ToolDreamer，EACL 2026 口头）上均已验证。
3. 技能侧唯一沾边的 SkillDAG（arXiv 2606.03056）只做了**库侧**索引期生成（e_needs），推理时查询原样嵌入——与本 idea 方向正交，反而验证了可行性、可作消融基线。
4. 时效敏感：技能检索赛道 3 个月出了约 10 篇论文，SkillDAG v2 更新于 2026-07-02。**尽快预实验 + arXiv 占位。**
