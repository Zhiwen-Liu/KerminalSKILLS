"""
NPU 工具函数模板

使用方法:
    from model_npu import init_npu, get_device, load_pretrained
    
    device = init_npu()
    model = load_pretrained(device=device)
"""
import os
import json
import torch
from typing import Optional

_npu_initialized = False


def init_npu(device_id: int = None) -> torch.device:
    """
    初始化 NPU 环境
    
    Args:
        device_id: NPU 设备 ID，默认从 ASCEND_RT_VISIBLE_DEVICES 读取
    
    Returns:
        torch.device: NPU 设备对象
    """
    global _npu_initialized
    
    if _npu_initialized:
        return get_device()
    
    if device_id is not None:
        # 仅在未设置时设置默认卡号，外部已设置则不覆盖
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', str(device_id))
    
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    
    if not torch.npu.is_available():
        raise RuntimeError("NPU 不可用，请检查驱动和 CANN 安装")
    
    torch.npu.set_device(0)
    _npu_initialized = True
    
    return torch.device('npu:0')


def get_device() -> torch.device:
    """获取当前设备"""
    if _npu_initialized:
        return torch.device('npu:0')
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device('npu:0')
    except ImportError:
        pass
    return torch.device('cpu')


def synchronize():
    """同步 NPU 操作"""
    try:
        torch.npu.synchronize()
    except:
        pass


def memory_info() -> dict:
    """获取 NPU 显存信息 (MB)"""
    try:
        return {
            'allocated_mb': torch.npu.memory_allocated() / 1024**2,
            'reserved_mb': torch.npu.memory_reserved() / 1024**2
        }
    except:
        return {'allocated_mb': 0, 'reserved_mb': 0}


def load_pretrained(
    model_id: str,
    model_class,
    config_key_mapping: dict = None,
    output_heads: dict = None,
    use_tf_gamma: bool = False,
    device: torch.device = None
) -> torch.nn.Module:
    """
    加载预训练模型 (解决 transformers 5.x 兼容性问题)
    
    Args:
        model_id: HuggingFace 模型 ID
        model_class: 模型类 (需有 from_hparams 方法)
        config_key_mapping: 配置键映射
        output_heads: 输出头配置
        use_tf_gamma: 是否使用 TF gamma
        device: 目标设备
    
    Returns:
        加载好的模型
    """
    from huggingface_hub import hf_hub_download
    
    # 下载配置和权重
    config_path = hf_hub_download(repo_id=model_id, filename='config.json')
    model_path = hf_hub_download(repo_id=model_id, filename='pytorch_model.bin')
    
    with open(config_path) as f:
        config = json.load(f)
    
    # 构建模型参数
    hparams = {}
    if config_key_mapping:
        for target_key, source_key in config_key_mapping.items():
            if source_key in config:
                hparams[target_key] = config[source_key]
    
    if output_heads:
        hparams['output_heads'] = output_heads
    if use_tf_gamma:
        hparams['use_tf_gamma'] = use_tf_gamma
    
    # 创建模型
    model = model_class.from_hparams(**hparams)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    
    if device is not None:
        model = model.to(device)
    
    return model
