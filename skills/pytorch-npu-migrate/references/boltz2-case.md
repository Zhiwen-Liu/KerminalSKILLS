# Boltz2 NPU 迁移案例

## 项目信息

- 项目: Boltz2 (蛋白质结构预测扩散模型)
- 源仓库: https://github.com/jwohlwend/boltz
- 迁移目标: Ascend 910B2 NPU
- 方案: transfer_to_npu + Lightning accelerator="gpu" (方案A 变体)

## 关键经验

### 1. transfer_to_npu 与 Lightning Trainer 配合

必须用 `accelerator="gpu"` 而非 `"cpu"`。Lightning 在 predict() 时会强制
调用 setup_device，若用 CPU accelerator 会把模型从 NPU 移回 CPU。

transfer_to_npu 会将 torch.cuda.* 重定向到 torch.npu.*，
Lightning 识别 Ascend910B2 为 "CUDA device"，自动管理设备分配。

### 2. npu_utils 延迟初始化

不能在模块顶层 `import torch_npu`，否则 CPU 模式也会初始化 NPU context。
使用 `@lru_cache` 延迟检测 + `BOLTZ_FORCE_CPU=1` 环境变量。

### 3. DataLoader segfault

`num_workers > 0` 在 NPU 环境下可能 segfault。解决: `--num_workers 0`。

### 4. predict_step 设备传输

Lightning CPU accelerator 不会自动移动 batch 到 NPU。
在 predict_step 开头添加 batch → model device 传输，返回值移回 CPU。

## 迁移成果

- CLI 推理加速: CPU 48m30s → NPU 41s = 71x
- 纯推理: 630x 加速
- 结构预测 + 亲和力预测均打通
