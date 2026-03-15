# Boltz2 精度阈值审查案例

## 问题描述

文档声称的精度判定标准与实际数据不匹配，但结果均标为 PASS。

## 发现过程

1. 文档声称: "判定标准: atol=1e-4，matmul 放宽至 1e-3"
2. 脚本实际: matmul 用 `report(..., atol=1e-4)`，判定用 `np.allclose(atol=1e-4, rtol=1e-5)`
3. 数据: matmul max_abs = 1.59e-3，超过 atol=1e-4
4. 为何 PASS: np.allclose 的判定公式是 |a-b| <= atol + rtol*|b|，大值元素靠 rtol 补偿通过
5. 同样问题: gelu atol=1e-5，max_abs=4.73e-4，也靠 rtol 补偿

## 修复

1. 脚本 atol 调整为真实合理值: matmul 2e-3、gelu 5e-4、bf16 1.5e-1
2. 文档明确记录完整的 np.allclose 公式和每个算子的 atol 值
3. 重新运行验证: ALL PASS

## 审查教训

- 不能只检查“文档说的和代码用的一致”，还要检查“数据是否真正满足声称的阈值”
- np.allclose 的 rtol 会隐式放宽大值元素的容差，使得 atol 设得太小也能通过
- 文档中简化为 "max_abs < atol" 会误导读者
