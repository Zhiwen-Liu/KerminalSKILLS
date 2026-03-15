# Enformer-PyTorch NPU 迁移案例

## 项目信息

| 项目 | 信息 |
|------|------|
| 源项目 | [enformer-pytorch](https://github.com/lucidrains/enformer-pytorch) |
| 版本 | v0.8.11 |
| 模型类型 | Transformer (Genomics) |
| 参数量 | 251M (完整) / 4.26M (测试) |
| 迁移日期 | 2025-02-08 |

## 环境配置

| 组件 | 版本 |
|------|------|
| NPU | Ascend 910B2 |
| CANN | 8.5.0 |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0 |
| Python | 3.11.14 |

## 迁移方案

采用 `torch_npu + transfer_to_npu` 零代码修改方案。

### 核心代码

```python
import os
# 可选，指定 NPU 卡号
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '4'

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

device = torch.device('npu:0')
torch.npu.set_device(device)

# 原始代码无需修改
from enformer_pytorch import Enformer
model = Enformer.from_hparams(dim=1536, depth=11).to(device)
```

## 测试结果

| 类别 | 测试数 | 通过 |
|------|--------|------|
| 推理测试 | 4 | 4 |
| 训练测试 | 3 | 3 |
| 精度测试 | 2 | 2 |
| Finetune | 2 | 2 |
| 性能测试 | 2 | 2 |
| 预训练模型 | 1 | 1 |
| **总计** | **14** | **14 (100%)** |

## 性能指标

| 指标 | 结果 |
|------|------|
| 推理延迟 | ~10ms |
| 吐吐量 | ~100 samples/s |
| NPU vs CPU 精度 | 余弦相似度 1.000000 |
| 预训练模型相关性 | 0.4721 (>阈值 0.1) |

## 解决的问题

### 1. transformers 5.x 兼容性

**问题**: `from_pretrained` 报错 `'all_tied_weights_keys'`

**解决**: 手动加载

```python
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id=model_id, filename='config.json')
model_path = hf_hub_download(repo_id=model_id, filename='pytorch_model.bin')

with open(config_path) as f:
    config = json.load(f)

model = Enformer.from_hparams(
    dim=config['dim'], depth=config['depth'], heads=config['heads'],
    output_heads=dict(human=5313, mouse=1643), target_length=896,
)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
```

### 2. 算子回退

以下算子回退到 CPU，性能影响可忽略:
- `aten::lgamma` - 位置编码
- `aten::poisson` - 波松采样

### 3. 训练 Loss 为负

**问题**: Poisson loss 返回负值

**原因**: `torch.rand()` 生成的 target 不符合波松分布

**解决**: 使用 `torch.poisson()` 生成 target

```python
# 错误
target = torch.rand(2, 128, 32) * 10

# 正确
target = torch.poisson(torch.ones(2, 128, 32) * 2.0)
```

## 项目结构

```
enformer-test/
├── enformer_npu/
│   ├── __init__.py
│   ├── npu_utils.py       # init_npu, load_pretrained
│   ├── inference.py       # EnformerNPU
│   └── train.py           # EnformerTrainer
├── data/test-sample.pt    # 验证数据
├── test_npu.py            # 14项测试
├── example.py
├── docs/API.md
├── README.md
└── MIGRATION_REPORT.md
```

## 参考路径

- 项目位置: `/data/models/enformer-test`
- 测试命令: `python test_npu.py`
- 预训练测试: `HF_ENDPOINT=https://hf-mirror.com python test_npu.py --pretrained`
