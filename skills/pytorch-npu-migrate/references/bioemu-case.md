# BioEmu NPU 迁移案例

## 项目信息

| 项目 | 信息 |
|------|------|
| 源项目 | [microsoft/bioemu](https://github.com/microsoft/bioemu) |
| 模型类型 | 扩散模型 (蛋白质结构) |
| 参数量 | 120M |
| 迁移日期 | 2025-02 |
| 迁移方案 | **方案 B: 手动设备抽象层** |

## 迁移方案

未使用 `transfer_to_npu`，而是新建 `device.py` 抽象层，仅修改 3 个文件 ~20 行代码。

修改文件:
- `src/bioemu/device.py` (新增) - NPU/CUDA/CPU 设备检测
- `src/bioemu/sample.py` - 引入 get_device()
- `src/bioemu/so3_sde.py` - 修复 l_grid 设备一致性

## 测试结果

- 测试用例: 53/53 通过 (NPU)
- 精度: CPU vs NPU max_abs_diff ~1e-6 (tiny_model) ~ 1e-4 (生产模型)
- 性能: 端到端采样加速 16.3x

## 与 Enformer 案例的对比

| 维度 | Enformer | BioEmu |
|------|----------|--------|
| 迁移方案 | transfer_to_npu (零修改) | 手动 device.py (3 文件) |
| 模型类型 | Transformer | 扩散模型 + SO3 |
| 精度验证 | 余弦相似度 1.0 | 逐元素 4 列对比 |
| 验证深度 | 14 项测试 | 53 项测试 + 8 项脚本交叉验证 |

## 相关报告

- 验证报告: 见 `$hardware-comparison-report` 的 `references/bioemu-case.md`
- 项目位置: `/data/models/bioemu-test/bioemu/`
