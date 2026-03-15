# Boltz CPU vs NPU 对比文档审查案例

## 项目背景

Boltz 是一个蛋白质结构预测模型，包含 Boltz1 和 Boltz2 两个版本。
项目将其从 CPU 迁移到华为昇腾 NPU (Ascend 910B2)，
产生了两份文档：`docs/CPU_vs_NPU.md`（主报告）和 `MIGRATION_REPORT.md`（迁移概述）。

## 审查过程

### 第一轮：文档-代码校验

1. **扫描代码中的数值断言**
   ```bash
   grep -rn 'allclose\|isclose\|atol' tests/ scripts/
   ```
   发现 3 个测试文件共 4 条 allclose + verify_precision.py 12 次 report()。

2. **核实 allclose 阈值**
   - 测试代码: `atol=1e-8`（OPM/TriAttn）、`atol=1e-5`（regression）
   - 文档: 3.1 节写 allclose=PASS，但 max_abs=3.12e-02，远超 atol
   - 实际靠 rtol=1e-5 在大值场景主导 → 表述易误导

3. **核实 randint 参数**
   - tests/: `randint(0,1)` → 全零 mask（trivial case）
   - verify_precision.py: `randint(0,2)` → 非零 mask（真实 case）
   - 文档覆盖表将 test 文件映射到 verify_precision 的数据 → 矛盾

4. **核实复现命令**
   - 文档条件: recycling=1, sampling=5
   - 复现命令: 未指定这两个参数
   - 代码默认: recycling=3, sampling=200
   - 结论: 复现命令缺参数，无法复现

### 第二轮：模块间交叉校验

1. **总结-细节数量匹配**
   - 精度总结: "4 项指标一致" vs 细节表只有 2 项 → 矛盾
   - 性能总结: 5 个组件 vs 细节表 6 个（缺 RelPos）→ 不一致

2. **RNG 逻辑自洽性**
   - 文档解释 Boltz2 差异来自 RNG 后端不同
   - 但 Boltz1 使用相同 RNG 机制却差异=0，未解释
   - 代码验证: 两个模型共享 diffusion.py 中的 torch.randn 采样路径
   - 结论: 自相矛盾

3. **交叉验证链维度混淆**
   - 链条: 底层(chunk一致性) → 中层(DiffCond) → 上层(CLI)
   - chunk 一致性是 NPU 同设备对比，不是 CPU vs NPU
   - 与跨设备精度链混在一起，逻辑不严密

4. **性能拆分自洽性**
   - 声称: 纯计算 34x (CPU 59s vs NPU 1.7s)，固定开销与设备无关
   - 反推: CPU固定开销=160s vs NPU固定开销=61.3s，相差近 100s
   - 结论: 前提不成立或拆分有误

5. **多文档统计口径**
   - MIGRATION_REPORT: "适配 11 个文件"
   - 实际 grep: 14 个 .py 文件引用 npu_utils
   - 差 3 个，口径未说明

## 发现汇总

| 等级 | 数量 | 典型问题 |
|------|:----:|----------|
| 🔴 事实错误 | 3 | 复现命令缺参数、覆盖表映射错、指标数不匹配 |
| 🔴 逻辑矛盾 | 3 | RNG 自相矛盾、性能拆分不自洽、验证链混维度 |
| 🟡 数据不一致 | 3 | 文件数不匹配、总结缺组件、输出遗漏 |
| ℹ️ 表述可改进 | 3 | allclose 误导、RNG 缺独立验证、上游 bug 缺追溯 |

## 关键经验

1. **randint(0,N) 是高频陷阱**: `randint(0,1)` 产生全零，`randint(0,2)` 才产生 0/1。
   文档和代码用了不同参数时，覆盖表映射就会出错。

2. **多模型共享采样路径时，RNG 差异叙事必须统一**: 不能 A 有差异 B 没有却不解释。

3. **总结表很容易和细节表脱节**: 写完总结后必须回头逐行核对细节数据。

4. **性能拆分要用算术验证**: 声称 "固定开销与设备无关" 就要用端到端数据反推检验。

5. **复现命令必须与代码默认值交叉核对**: 不能只看命令本身，要查未指定参数的默认值。

## 审查产出

审查意见已保存到项目 `docs/CPU_vs_NPU_review.md`，共 12 条问题。45 分钟完成。
