---
name: hardware-comparison-report
description: |
  硬件迁移项目的 CPU vs NPU/GPU 对比验证报告编写方法论。覆盖精度、性能、测试通过率三个维度。
  当用户需要编写 CPU 和 NPU 对比报告、精度验证报告、硬件迁移验证文档、对比文档、
  或精度性能报告时使用此 skill。
  Write comparison validation reports for hardware migration projects (CPU vs NPU/GPU).
triggers:
  - CPU 和 NPU 对比
  - 精度验证报告
  - 硬件迁移验证
  - 写对比文档
  - 精度性能报告
  - CPU vs NPU comparison report
  - write migration validation report
  - hardware benchmark report
upstream: [pytorch-npu-migrate, heterogeneous-validation]
downstream: [doc-cross-review, kibble-case-organize]
---

# 硬件对比验证报告编写

指导编写硬件迁移项目（CPU → NPU/GPU）的对比验证报告，覆盖精度、性能、测试通过率。

## 核心原则

### 1. 结论前置，细节后置

```
测试环境
├─ 第1 精度总结    ← 读者 30s 内获取结论
├─ 第2 性能总结    ← 读者 30s 内获取结论
├─ 第3 精度测试细节 ← 需追溯时查看
├─ 第4 性能测试细节
└─ 附录 (环境修复/复现命令)
```

总结表用 ✅/⚠️ 视觉标记，一眼可扫。

### 2. 以测试项为锚点组织精度数据

逐测试文件梳理所有数值断言，而非按“模块”分类：

```bash
grep -rl 'allclose\|isclose\|atol\|assert.*abs' tests/
```

报告开头放覆盖映射表：

```markdown
| 测试文件 | 数值断言数 | 对应章节 |
|---------|:---------:|:--------:|
| test_models.py | 1 | 第3.1 |
```

### 3. 区分确定性差异和随机性差异

凡涉及随机数的测试项，必须单独验证 RNG 一致性：

```python
torch.manual_seed(42)
cpu_r = torch.randn(5, device='cpu')
torch.manual_seed(42)
npu_r = torch.randn(5, device='npu:0')
print(f'max_diff={(cpu_r - npu_r.cpu()).abs().max()}')
```

报告中明确标注差异来源：
- ✅ 确定性计算差异 → float32 浮点误差
- ⚠️ 随机性差异 → RNG 实现不同，预期行为

### 4. 建立分层交叉验证链

```
底层: 基础算子 (matmul/softmax/LN) → 保证模型前向精度
中层: SDE/ODE 数值              → 保证采样器分布正确
上层: 单步精度 + 多步累积         → 保证端到端结果正确
```

报告中显式写出这条链，让读者知道各验证项的依赖关系。

### 5. 精度对比表统一 4 列标准格式

| 字段 | 含义 |
|------|------|
| max_abs_diff | 最坏情况 |
| mean_abs_diff | 整体水平 |
| max_rel_diff | 小值放大效应 |
| mean_rel_diff | 相对误差整体水平 |

**atol 设定规则** (必须在文档中明确记录):
- matmul: atol 随内积长度 N 放大，N=1024 时 atol ≈ 2e-3
- softmax/layernorm: 无大规模累加，atol=1e-5
- gelu/silu: 不同近似实现，atol 放宽至 5e-4
- bf16: ULP 分析确定 atol

**每个 PASS/DIFF 必须附带解释**。

通用 report() 函数见: `templates/comparison_utils.py`

### 6. 性能数据必须标注测量条件

每行性能数据必须说明：
- 数据规模 (batch_size, seq_len, 维度)
- 是否含 IO/后处理
- 是否含 NPU 同步 (`torch.npu.synchronize()`)

反直觉结果（NPU 更慢）必须给出原因分析。

### 7. 复现命令作为文档一部分

附录中放置可直接 copy-paste 执行的命令。

## 工作流程

```
1. 扫描 tests/ 全部数值断言 → 建立覆盖清单
2. 逐文件在 CPU 和 NPU 上运行，记录耗时 + 通过情况
3. 对每个数值断言提取 4 列精度数据
4. 编写脚本补充验证基础算子/SDE/多步累积
5. 验证 RNG 一致性，对随机性差异定性标注
6. 按模板填充: 总结表 → 细节 → 附录
```

## 参考案例

- BioEmu 项目: `references/bioemu-case.md`
- Boltz2 项目: `references/boltz2-case.md`
- 报告模板: `templates/report.md`
- 通用工具函数: `templates/comparison_utils.py`
