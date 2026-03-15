# BioEmu CPU vs NPU 验证案例

## 项目信息

- 项目: BioEmu (蛋白质结构扩散模型)
- 源仓库: microsoft/bioemu
- 迁移目标: Ascend 910B2 NPU
- 报告位置: `/data/models/bioemu-test/bioemu/docs/CPU_NPU_VALIDATION_REPORT.md`

## 数据规模

- 测试文件: 15 个，53 个用例
- 数值断言文件: 8 个，37 处断言
- 脚本验证项: 8 个独有项（与 tests/ 交叉验证）

## 精度结果摘要

- 确定性计算 max_abs_diff: ~1e-6 (tiny_model) ~ 1e-4 (生产 120M)
- 训练损失: diff = 0 (完全一致)
- NPU 确定性: bit-exact (diff = 0)
- 随机性差异: 已验证为 RNG 实现不同导致，非精度问题

## 性能结果摘要

- 密集计算: 10x ~ 28x 加速
- 端到端推理: 16.3x 加速
- 小规模计算: 无优势 (传输开销占主导)

## 关键经验

1. 扫描断言建覆盖清单，避免遗漏
2. RNG 验证避免误判梯度/累积差异
3. 基础算子 → 模型前向 → 采样分布的信任链很有说服力
4. 性能反直觉结果需解释（test_denoiser NPU 更慢因数据量太小）

## 验证脚本

```
scripts/precision_compare.py      # 精度 + 性能对比
scripts/deep_precision_test.py    # 深度精度 (5 维度)
scripts/precision_supplement.py   # 补充精度对比
scripts/compare_cpu_npu.py        # 基础算子 + SDE + Denoiser
```

## 相关 skills 案例

- 迁移方案: 见 `$pytorch-npu-migrate` 的 `references/bioemu-case.md`
- 案例归档: 见 `$kibble-case-organize` 的 `references/bioemu-case.md`
