---
name: kibble-case-organize
description: |
  将迁移/适配项目整理为 kibble 案例仓库规范结构，包括 case.toml 元数据、
  deploy 部署产物、devlog 开发日志、showcase 展示材料的完整组织流程，
  包含对话记录导出与摘要总结。
  当用户提到整理 kibble 案例、案例归档、项目沉淀、导出对话记录等场景时使用。
  Organize migration projects into kibble case repository structure.
triggers:
  - 整理 kibble 案例
  - 按 kibble 规范组织
  - 案例归档
  - 整理项目到 kibble
  - kibble case
  - 案例沉淀
  - 导出对话记录
  - chatlog 整理
  - organize kibble case
  - archive migration project
upstream: [pytorch-npu-migrate, hardware-comparison-report]
downstream: []
---

# 整理 kibble 案例

将已完成的迁移/适配项目按 kibble 案例仓库规范整理为可归档、可交付的标准结构。

## 前置条件

- kibble 仓库位于 `/data/kibble`
- 规范文档: `/data/kibble/README.md`、`/data/kibble/docs/CONTRIBUTING.md`、`/data/kibble/docs/taxonomy.md`
- 模板: `/data/kibble/templates/case.toml`
- 已有案例可参考: `/data/kibble/cases/*/`

## 目标结构

```
cases/<case-name>/
├── case.toml                  # 元数据（必填）
├── deploy/                    # 部署产物
│   ├── README.md              # 顶层索引
│   └── <project-name>/        # 完整可部署项目
│       ├── README.md          # 环境准备、运行命令、性能数据
│       ├── src/
│       ├── tests/
│       ├── scripts/
│       └── models/            # 含下载说明（不含实际权重）
├── devlog/
│   └── kernelcat-feedback.md  # 能力评估
└── showcase/
    ├── summary.md             # 案例总结（面向商务/市场）
    └── chatlog/               # 人机协作日志
        ├── <case-name>.txt
        └── <case-name>-summary.txt
```

**关键原则**:
- `deploy/` 根下只放 README 索引 + 项目子目录，不散落文件
- 清理 `.git`、`__pycache__`、`.egg-info` 等非交付物
- `models/` 不含实际权重，放下载命令

## 工作流程

### 阶段 0: 判断创建还是更新

检查 `/data/kibble/cases/` 下是否已有同名 case。若已存在则增量更新。

### 阶段 1: 分析源项目

阅读 README、迁移报告，盘点现有产物，确定 case.toml 分类字段值。

### 阶段 2: 创建目标结构

按上述目标结构组织文件。

### 阶段 3: 填写 case.toml

从 `/data/kibble/templates/case.toml` 复制模板，逐节填写：

1. 基础信息: name, summary, status, created, contacts
2. 分类: entry_layer, primary_layers, task, scope, tags
3. 技术栈: languages, tools, dev_mode
4. 来源与目标: source/target platform + hardware
5. 难度: level + factors
6. 工作量: kernelcat 耗时 vs 工程师耗时
7. KernelCAT 能力: version, completion, strengths, limitations
8. 环境要求: hardware, software
9. 产物: deploy, devlog, showcase 路径
10. 关联信息: source_repo, docs, paper

### 阶段 4: 编写 devlog/kernelcat-feedback.md

使用 `templates/kernelcat-feedback.md` 模板：
- 案例概述: 5 条 bullet
- 过程记录: 按阶段记录 KernelCAT 行为/问题/人工介入/结果
- 能力评估: 表格形式，含评分和说明
- 具体问题记录: 现象/原因/解决/建议
- 总结: 整体评价 + 工作量对比表 + 改进建议

### 阶段 5: 编写 showcase/summary.md

使用 `templates/summary.md` 模板，面向非技术人员：
- 一句话总结 + 案例亮点表 + 迁移效果 + 性能数据

### 阶段 6: 导出与摘要对话记录

#### 发现项目路径
```bash
python3 /tmp/history-viewer/view_chat_history.py \
  --cli-name kcat --list-projects /root/.kerminal/sessions/
```

#### 导出完整对话
```bash
git clone https://github.com/Zhiwen-Liu/claude-code-history-viewer /tmp/history-viewer
python3 /tmp/history-viewer/view_chat_history.py \
  --cli-name kcat \
  --project /data/models/<project-path> \
  --no-thinking \
  --export /data/kibble/cases/<case-name>/showcase/chatlog/<case-name>.txt \
  /root/.kerminal/sessions/
```

#### 编写对话摘要

生成 `<case-name>-summary.txt`：
1. 去噪: 移除无实质内容的消息
2. 分阶段组织: 迁移启动 → 功能测试 → 精度对比 → 项目整理 → 归档
3. 保留关键对话
4. 末尾附加关键成果汇总

### 阶段 7: 验证

- [ ] case.toml 必填字段完整，符合 taxonomy.md
- [ ] deploy/ 结构正确，无散落文件
- [ ] deploy/<project>/ 可独立部署
- [ ] models/ 含下载说明，无实际权重
- [ ] devlog/kernelcat-feedback.md 完整
- [ ] showcase/summary.md 面向非技术人员
- [ ] chatlog 目录存在（若有会话历史）
- [ ] 无 .git、__pycache__、.egg-info
- [ ] 无敏感信息

## 参考

- 完整案例: `references/bioemu-case.md`
- 更新已有案例: `references/boltz2-case.md`
- chatlog 导出工具: https://github.com/Zhiwen-Liu/claude-code-history-viewer
- devlog 模板: `templates/kernelcat-feedback.md`
- summary 模板: `templates/summary.md`
- deploy README 模板: `templates/deploy-readme.md`
- chatlog 摘要模板: `templates/chatlog-summary.txt`
