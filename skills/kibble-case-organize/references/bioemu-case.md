# BioEmu 案例整理记录

## 项目信息

| 项目 | 信息 |
|------|------|
| 名称 | BioEmu（微软蛋白质结构扩散模型） |
| 任务 | CUDA → 昇腾 NPU 迁移 |
| 难度 | easy |
| 位置 | `/data/kibble/cases/bioemu/` |

## 整理前的问题

| 问题 | 说明 |
|------|------|
| deploy 文件散落 | 项目文件直接放在 `deploy/` 根下，而非 `deploy/bioemu/` 子目录 |
| 缺少权重下载说明 | `deploy/models/checkpoints/` 无 README.md |
| case.toml 缺注释 | 分类字段无行内注释说明选择理由 |
| devlog 格式不符 | 使用了表格式而非规范的分阶段记录格式 |
| 缺少 showcase/summary.md | 无面向商务/市场的案例总结 |

## 整理步骤

### 1. 分析差异

```bash
# 将当前项目与 kibble 参考结构对比
diff <(cd /data/kibble/cases/bioemu && find . -type f | sort) \
     <(cd /data/models/bioemu-test && find . -type f | sort)
```

### 2. 重构 deploy/

关键操作: 把项目文件从 `deploy/` 根下移入 `deploy/bioemu/`

```bash
mkdir -p deploy/bioemu
mv deploy/{src,tests,scripts,models,...} deploy/bioemu/
# deploy/ 根下只保留 README.md + bioemu/
```

同时清理:
- `.git` 目录
- `__pycache__` 目录
- `.egg-info` 目录

### 3. 补充缺失文件

- `deploy/README.md`: 顶层索引，指向 `bioemu/`
- `deploy/bioemu/models/checkpoints/README.md`: 权重下载命令
- `showcase/summary.md`: 案例总结

### 4. 对齐 case.toml

给关键分类字段加行内注释:

```toml
entry_layer = "model"             # 用户入口是模型层推理/训练
primary_layers = ["model"]        # 主要工作在模型层设备适配
touched_layers = ["model", "op"]  # 涉及模型设备适配和 SO3 算子设备一致性修复
task = "adapt"                    # 少量适配，API 替换
dev_mode = "api"                  # 通过 torch_npu API 调用适配
```

### 5. 重写 devlog

从表格式改为标准分阶段记录格式（见模板）

## 最终结构

```
cases/bioemu/
├── case.toml
├── deploy/
│   ├── README.md              # 顶层索引
│   └── bioemu/                # 完整可部署项目
│       ├── README.md
│       ├── MIGRATION_REPORT.md
│       ├── pyproject.toml
│       ├── setup_env.sh
│       ├── src/bioemu/
│       ├── tests/
│       ├── scripts/
│       ├── models/checkpoints/
│       ├── notebooks/
│       └── docs/
├── devlog/
│   └── kernelcat-feedback.md
└── showcase/
    ├── summary.md
    └── chatlog/
```

## 经验教训

1. **deploy 所有散落问题最常见**: 迁移产物直接复制到 deploy/ 而非放入子目录
2. **showcase/summary.md 容易被遗漏**: 迁移流程关注技术，容易忘记商务材料
3. **devlog 格式有严格规范**: 不能自由发挥，必须按“阶段 → 行为/问题/介入/结果”结构
4. **模型权重不归档**: 只放下载说明，不提交实际文件
5. **先 diff 再动手**: 与参考案例对比后再调整，避免遗漏

## 相关 skills 案例

- 迁移方案: 见 `$pytorch-npu-migrate` 的 `references/bioemu-case.md`
- 验证报告: 见 `$hardware-comparison-report` 的 `references/bioemu-case.md`
