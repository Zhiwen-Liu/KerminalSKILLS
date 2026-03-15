# Boltz2 CPU vs NPU 对比案例

## 项目信息

- 项目: Boltz2
- 报告位置: docs/CPU_vs_NPU.md
- 对比脚本: scripts/cpu_vs_npu_compare.py

## 关键经验

### 1. atol 必须按算子设定

不同算子的误差特性差异很大：
- matmul (N=1024): atol=2e-3 (浮点累加误差大)
- softmax: atol=1e-5 (无累加，精度极高)
- gelu: atol=5e-4 (近似实现差异)
- bf16: atol=1.5e-1 (基于 ULP 分析)

若用统一 atol=1e-4，matmul 和 gelu 会靠 rtol 补偿通过，判定标准名不副实。

### 2. bf16 用 ULP 分析

bf16 的 1 ULP 随元素绝对值变化: 对绝对值 ~16 的元素，1 ULP = 0.125。
max_abs=0.125 不是计算错误，是 bf16 格式固有极限。
52万元素中仅 6 个超过 0.1，mean_abs 仅 1.7e-6。

### 3. 每个结果必须有解释

只写 "PASS" 不够，读者需要知道:
- 为何在阈值内 (如 "内积长度短，累加误差小")
- max_rel_diff 异常大时为何仍可接受 ("小值放大效应")
- RNG 差异与计算精度的区分方法

## 性能数据

| 场景 | CPU | NPU | 加速比 |
|------|:---:|:---:|:------:|
| CLI 推理 | 48m30s | 41s | 71x |
| 纯推理 | 47m18s | 4.5s | 630x |
| matmul | 21.1ms | 0.051ms | 412x |
