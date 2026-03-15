"""
硬件对比通用工具函数

提供统一的精度对比、性能计时、结果汇总功能。
复制到项目的 scripts/ 目录即可使用。

用法:
    from comparison_utils import report, timer, Summary
"""
import time
import numpy as np
import torch
from typing import Optional


def report(label: str, cpu_val, npu_val, atol: float = 1e-5) -> dict:
    """
    对比 CPU 和 NPU 的计算结果，输出 4 列标准格式。

    Args:
        label: 对比项名称
        cpu_val: CPU 端结果 (Tensor 或 ndarray)
        npu_val: NPU 端结果 (Tensor 或 ndarray)
        atol: 绝对容差阈值

    Returns:
        dict: {label, max_abs, mean_abs, max_rel, mean_rel, ok}
    """
    c = _to_numpy(cpu_val)
    n = _to_numpy(npu_val)
    ad = np.abs(c - n)
    rd = ad / np.maximum(np.abs(c), 1e-12)
    ma, me = float(ad.max()), float(ad.mean())
    mr, mre = float(rd.max()), float(rd.mean())
    ok = bool(np.allclose(c, n, atol=atol, rtol=1e-5))
    tag = "PASS" if ok else "DIFF"
    print(f"  [{tag}] {label}")
    print(f"    max_abs={ma:.2e}  mean_abs={me:.2e}  max_rel={mr:.2e}  mean_rel={mre:.2e}")
    return {"label": label, "max_abs": ma, "mean_abs": me,
            "max_rel": mr, "mean_rel": mre, "ok": ok}


def _to_numpy(val) -> np.ndarray:
    """Tensor / ndarray / list 统一转为 float32 numpy."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().float().numpy().ravel()
    return np.asarray(val, dtype=np.float32).ravel()


class timer:
    """
    计时上下文管理器，自动处理 NPU 同步。

    用法:
        with timer("model forward", device) as t:
            output = model(input)
        print(t.elapsed)
    """
    def __init__(self, label: str = "", device: Optional[torch.device] = None):
        self.label = label
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device and self.device.type == 'npu':
            torch.npu.synchronize()
        self._start = time.time()
        return self

    def __exit__(self, *args):
        if self.device and self.device.type == 'npu':
            torch.npu.synchronize()
        self.elapsed = time.time() - self._start
        if self.label:
            print(f"  {self.label}: {self.elapsed:.3f}s")


class Summary:
    """
    累积多个 report() 结果，最后统一输出。

    用法:
        s = Summary()
        s.add(report("test1", cpu, npu))
        s.add(report("test2", cpu, npu))
        s.print_summary()
    """
    def __init__(self):
        self.results = []

    def add(self, result: dict):
        self.results.append(result)

    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.results)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in self.results:
            tag = "PASS \u2705" if r["ok"] else "DIFF \u26a0\ufe0f"
            print(f"  {tag}  {r['label']}  (max_abs={r['max_abs']:.2e})")
        print()
        if self.all_ok():
            print("ALL PASS \u2705")
        else:
            print("SOME NEED ATTENTION \u26a0\ufe0f")
