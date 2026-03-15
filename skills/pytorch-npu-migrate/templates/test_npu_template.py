#!/usr/bin/env python3
"""
NPU 迁移测试模板

使用方法:
    python test_npu.py              # 全部测试
    python test_npu.py --quick      # 快速测试
    python test_npu.py --inference  # 推理测试
    python test_npu.py --training   # 训练测试
    python test_npu.py --precision  # 精度测试
"""
import os
import sys
import time
import argparse

# 指定 NPU 卡号，可根据实际环境修改
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '4')

import torch
import torch.nn as nn


def init_npu():
    """NPU 初始化"""
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    return device


class NPUTestSuite:
    """测试套件基类"""
    
    def __init__(self, device):
        self.device = device
        self.results = {}
        self.model = None
    
    def setup(self):
        """初始化测试模型 - 子类实现"""
        raise NotImplementedError
    
    def _record(self, name, passed, detail=""):
        """记录测试结果"""
        self.results[name] = {'passed': passed, 'detail': detail}
        status = "✅" if passed else "❌"
        print(f"  {status} {name}" + (f": {detail}" if detail else ""))
        return passed
    
    def test_inference(self):
        """推理测试"""
        print("\n[推理测试]")
        self.model.eval()
        
        # TODO: 根据具体模型实现
        test_input = self._create_test_input()
        
        try:
            with torch.no_grad():
                output = self.model(test_input)
            self._record("基本推理", True, f"{output.shape}")
        except Exception as e:
            self._record("基本推理", False, str(e))
    
    def test_training(self):
        """训练测试"""
        print("\n[训练测试]")
        self.model.train()
        
        try:
            from torch_npu.optim import NpuFusedAdamW
            optimizer = NpuFusedAdamW(self.model.parameters(), lr=1e-4)
        except:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # TODO: 根据具体模型实现
        test_input = self._create_test_input()
        test_target = self._create_test_target()
        
        try:
            optimizer.zero_grad()
            loss = self._compute_loss(test_input, test_target)
            loss.backward()
            optimizer.step()
            self._record("前向+反向传播", True, f"loss={loss.item():.4f}")
        except Exception as e:
            self._record("前向+反向传播", False, str(e))
    
    def test_precision(self):
        """精度测试 - CPU vs NPU"""
        print("\n[精度测试]")
        
        # TODO: 根据具体模型实现
        test_input = self._create_test_input()
        
        try:
            self.model.eval()
            with torch.no_grad():
                npu_out = self.model(test_input).cpu()
            
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                npu_out.flatten().unsqueeze(0),
                npu_out.flatten().unsqueeze(0)  # TODO: 与 CPU 结果对比
            ).item()
            
            self._record("余弦相似度", cos_sim > 0.999, f"{cos_sim:.6f}")
        except Exception as e:
            self._record("精度测试", False, str(e))
    
    def test_performance(self):
        """性能测试"""
        print("\n[性能测试]")
        self.model.eval()
        test_input = self._create_test_input()
        
        # Warmup
        with torch.no_grad():
            _ = self.model(test_input)
        torch.npu.synchronize()
        
        # 测时
        times = []
        for _ in range(10):
            torch.npu.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = self.model(test_input)
            torch.npu.synchronize()
            times.append(time.time() - start)
        
        avg_ms = sum(times) / len(times) * 1000
        self._record("推理性能", avg_ms < 100, f"{avg_ms:.2f}ms")
    
    def _create_test_input(self):
        """创建测试输入 - 子类实现"""
        raise NotImplementedError
    
    def _create_test_target(self):
        """创建测试目标 - 子类实现"""
        raise NotImplementedError
    
    def _compute_loss(self, inputs, targets):
        """计算损失 - 子类实现"""
        raise NotImplementedError
    
    def summary(self):
        """输出测试汇总"""
        print("\n" + "=" * 50)
        print("测试汇总")
        print("=" * 50)
        
        passed = sum(1 for r in self.results.values() if r['passed'])
        total = len(self.results)
        
        print(f"\n通过: {passed}/{total} ({100*passed/total:.0f}%)")
        
        failed = [k for k, v in self.results.items() if not v['passed']]
        if failed:
            print(f"\n失败项: {', '.join(failed)}")
        
        print("\n" + ("✅ 所有测试通过!" if passed == total else f"⚠️ {total-passed} 项测试失败"))
        
        return passed == total


def main():
    parser = argparse.ArgumentParser(description='NPU 迁移测试')
    parser.add_argument('--quick', action='store_true', help='快速测试')
    parser.add_argument('--inference', action='store_true', help='仅推理测试')
    parser.add_argument('--training', action='store_true', help='仅训练测试')
    parser.add_argument('--precision', action='store_true', help='精度对比测试')
    args = parser.parse_args()
    
    print("=" * 50)
    print("NPU 迁移测试")
    print("=" * 50)
    
    device = init_npu()
    print(f"\n设备: {device}")
    
    # TODO: 替换为具体测试类
    # tester = MyModelTestSuite(device).setup()
    
    # run_all = not any([args.inference, args.training, args.precision])
    # if run_all or args.inference: tester.test_inference()
    # if run_all or args.training: tester.test_training()
    # if run_all or args.precision: tester.test_precision()
    # if run_all and not args.quick: tester.test_performance()
    # 
    # success = tester.summary()
    # sys.exit(0 if success else 1)
    
    print("\n请实现具体测试类")


if __name__ == '__main__':
    main()
