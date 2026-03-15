---
name: pytorch-npu-migrate
description: |
  PyTorch 模型迁移到华为昇腾 NPU 的完整流程。涵盖环境准备、代码适配（transfer_to_npu 零修改/手动设备抽象层）、
  测试验证、精度对齐和性能优化。当用户提到 PyTorch 模型迁移到 NPU、torch 适配昇腾、模型移植华为 NPU、
  NPU 推理或训练适配等场景时使用此 skill。
  Migrate PyTorch models to Huawei Ascend NPU. Covers environment setup, code adaptation
  (transfer_to_npu zero-modification / manual device abstraction), testing, precision alignment,
  and performance optimization.
triggers:
  - PyTorch 模型迁移到 NPU
  - torch 适配昇腾
  - 模型移植到华为 NPU
  - NPU 推理适配
  - NPU 训练适配
  - transfer_to_npu 用法
  - torch_npu 环境配置
  - enformer/transformers/CNN 迁移 NPU
  - migrate model to Ascend NPU
  - PyTorch NPU adaptation
upstream: []
downstream: [heterogeneous-validation, hardware-comparison-report, kibble-case-organize]
---

# PyTorch 模型迁移到昇腾 NPU

将 PyTorch 模型迁移到华为昇腾 NPU，基于 `torch_npu + transfer_to_npu` 零代码修改方案或手动设备抽象层方案。

## 迁移决策树

```
项目是否使用 cuda 硬编码？
├─ 是 → 是否需要同时支持 CPU/CUDA/NPU？
│       ├─ 否 → 方案 A: transfer_to_npu (零修改，推荐首试)
│       └─ 是 → 方案 B: 手动设备抽象层
└─ 否 → 方案 B: 手动设备抽象层
```

## 流程总览

```
环境准备 → 选择适配方案 → 代码适配 → 功能验证 → 精度验证 → 性能优化 → 交付
                                         │            │
                                         └─ 发现问题 ──┘ (迭代)
```

---

## 1. 环境准备

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<卡号>  # 可选，指定 NPU 卡号

npu-smi info
python -c "import torch_npu; print(torch_npu.__version__)"
```

> **卡号设置规范**: `ASCEND_RT_VISIBLE_DEVICES` 是可选配置。
> Python 代码中使用 `os.environ.setdefault()` 而非直接赋值，避免覆盖外部设置。

## 2. 代码适配

### 方案 A: transfer_to_npu 零代码修改 (推荐首试)

适用：代码中使用 `cuda` 硬编码设备，且不需要同时支持多后端。

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

model = Model().cuda()  # 实际分配到 NPU
```

**优点**: 改动量为零，快速验证。
**缺点**: 隐式转换，不支持多后端切换；禁用 `torch.jit.script`。

**Lightning Trainer 配合要点**:
- 必须设置 `accelerator="gpu"`（不能用 `"cpu"`）
- transfer_to_npu 将 CUDA 调用重定向到 NPU，Lightning 识别为 CUDA 设备后自动管理
- 若用 `accelerator="cpu"` + 手动 `model.to(npu)`，Lightning 会在 predict/fit 时强制移回 CPU

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu

trainer = Trainer(accelerator="gpu", devices=1)
trainer.predict(model, datamodule=dm)
```

参考案例: `references/enformer-case.md`, `references/boltz2-case.md`

### 方案 B: 手动设备抽象层

适用：项目需同时支持 CPU/CUDA/NPU，或原代码不依赖 cuda 硬编码。

```python
import os
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def _check_npu() -> bool:
    if os.environ.get("FORCE_CPU", "") == "1":
        return False
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False

def get_device(device=None):
    if device:
        return torch.device(device)
    if _check_npu():
        return torch.device("npu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
```

> **延迟导入原因**: `import torch_npu` 会立即初始化 NPU context 并占用显存。
> 使用 `@lru_cache` 确保只在首次调用时检测，`FORCE_CPU=1` 可完全跳过。

参考案例: `references/bioemu-case.md`

## 3. 测试验证矩阵

| 测试类型 | 验证内容 | 通过标准 |
|----------|----------|----------|
| 推理测试 | 前向传播、多头输出、embeddings | 无异常，shape 正确 |
| 训练测试 | 前向+反向、损失收敛、梯度有效性 | loss 下降趋势一致 |
| 精度测试 | CPU vs NPU 余弦相似度 | cosine_sim > 0.999 |
| 性能测试 | 推理延迟、吞吐量、显存占用 | 满足业务需求 |

## 4. 常见问题速查

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `aten::xxx not supported on NPU` | 算子未适配 | 通常自动回退 CPU，性能影响小 |
| `from_pretrained` 报错 `all_tied_weights_keys` | transformers 5.x 兼容性 | 手动加载：`hf_hub_download` + `load_state_dict` |
| DataLoader `Segmentation fault` | fork 子进程继承 NPU context | `num_workers=0` 或 `multiprocessing_context="spawn"` |
| JIT 编译失败 | transfer_to_npu 禁用 `torch.jit.script` | 需单独处理 JIT 模块 |
| tensor 在错误设备上 | 部分函数返回 CPU tensor | 手动 `.to(device)` |

## 5. 性能优化

### 优化器
```python
from torch_npu.optim import NpuFusedAdamW
optimizer = NpuFusedAdamW(model.parameters(), lr=1e-4)
```

### 混合精度
```python
from torch.npu.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(x, target=y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 显存优化
```python
config = ModelConfig(use_checkpointing=True)  # Gradient Checkpointing
```

## 6. 项目结构模板

```
model-npu/
├── model_npu/
│   ├── __init__.py
│   ├── npu_utils.py        # init_npu, get_device, load_pretrained
│   ├── inference.py
│   └── train.py
├── data/
├── test_npu.py
├── example.py
├── docs/API.md
├── README.md
└── MIGRATION_REPORT.md
```

模板代码见: `templates/npu_utils.py`, `templates/test_npu_template.py`

## 7. 迁移报告要点

1. **项目概述** — 源项目、版本、目标平台
2. **环境配置** — 硬件/软件环境、依赖包
3. **迁移方案** — 技术选型（方案 A/B）、代码修改清单
4. **功能验证** — 测试概览、详细结果
5. **精度验证** — CPU vs NPU 对比（建议使用 `heterogeneous-validation` skill）
6. **性能分析** — 推理/训练性能、显存占用
7. **已知问题** — 问题描述和解决方案

## 参考案例

- Enformer (方案 A: transfer_to_npu): `references/enformer-case.md`
- BioEmu (方案 B: 手动设备抽象层): `references/bioemu-case.md`
- Boltz2 (方案 A + Lightning): `references/boltz2-case.md`
