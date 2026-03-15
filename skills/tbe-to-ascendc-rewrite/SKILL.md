---
name: tbe-to-ascendc-rewrite
description: |
  将华为 TBE 算子迁移改写为 AscendC 原生 kernel。核心理念「先正确后快」：先实现泛化模板验证精度，
  再通过多模板架构做性能优化。当用户提到 TBE 算子转 AscendC、AscendC kernel 开发/调试/优化、
  NPU 算子精度/性能问题、多模板 kernel 架构设计时使用此 skill。
  Rewrite Huawei TBE operators as AscendC native kernels for Ascend NPU.
triggers:
  - TBE 算子转 AscendC
  - AscendC kernel 开发
  - AscendC 算子调试
  - AscendC 性能优化
  - NPU 算子精度问题
  - 多模板 kernel 架构
  - TBE to AscendC rewrite
  - AscendC operator development
  - Ascend kernel optimization
upstream: []
downstream: []
---

# TBE 算子 AscendC 改写

将 TBE 算子迁移为 AscendC 原生 kernel。核心理念「先正确后快」：先实现泛化模板验证精度，再通过多模板架构做性能优化。

## 改写流程

```
分析 TBE 原始实现 → 查官方参考 → 泛化模板实现 → 精度验证 → 多模板优化 → 性能验证 → 交付
```

每次优化后必须重跑精度，因为同步原语和流水线的微小调整很容易引入数据竞争。

### 官方参考查找

官方算子库 https://gitcode.com/cann/ops-nn 是最重要的参考源：
- `op_host/` 中 `CalTilingKey()` — 理解如何根据 shape 选择模板
- `op_kernel/` 同一算子的多个 `.h` 文件 — 每个对应一个模板分支
- 命名规律：`_float.h` / `_cast.h`（按数据类型），`_nc_large_*.h`（按 UB 容量）

## 工程规范

```
OpName/
  op_host/     # Tiling 结构体(.h) + tiling 计算(.cpp)
  op_kernel/   # AscendC kernel
  torch_ext/   # torch C++ extension
  test/        # 测试脚本
  results/     # 测试结果
```

**关键约束**:
- Tiling 结构体只在 `op_host/` 定义一次，kernel 通过 `#include` 引用
- Tiling 结构体只保留 kernel 实际引用的字段

## 多模板架构设计

标准开发模式：**先做泛化模板支持所有场景，再增加性能优化模板，在 tiling 中判断走哪个模板**。

### 分支维度一：UB 容量（几乎所有算子都需要）

```
perCalcSize = 所需 buffer 数 × ncAlign × elemSize
if perCalcSize ≤ ubSize → SMALL 模板（整体装入，可批量处理）
else                    → LARGE 模板（切片处理）
```

- SMALL: `yNumPerCalc = ubSize / perCalcSize` 批量处理多个空间位置
- LARGE: NC 维度切片，task 数膨胀为 `空间位置数 × NC切片数`

### 分支维度二：数据类型（涉及 fp16/bf16 时）

fp16/bf16 通常需 cast 到 fp32 再计算，UB 需额外 cast buffer。
官方用 `tilingKey = dtype编码 * 10 + UB容量编码` 组合两个维度，kernel 通过 `TILING_KEY_IS(n)` 分发。

### NC 切片通用策略

```
ncSliceNum  = ceil(perCalcSize / ubSize)
ncSliceLen  = ncAlign / ncSliceNum / ALIGN_NUM * ALIGN_NUM
ncSliceNum  = ceil(ncNum / ncSliceLen)       // 重算实际片数
ncSliceTail = ncNum - ncSliceLen * (ncSliceNum - 1)
taskNum     = spatialPositions * ncSliceNum   // task 数膨胀
```

kernel 中 task 拆解：
```
n = taskIdx % ncSliceNum             // NC 第几片
spatialIdx = taskIdx / ncSliceNum    // 空间位置
ncMoveNum = (n == ncSliceNum-1) ? ncSliceTail : ncSliceLen
```

## 同步原语选择

### PipeBarrier（粗粒度，调试用）

`PipeBarrier<PIPE_ALL>` 阻塞所有 pipe，最安全但消除流水线并行性。仅用于调试确认逻辑正确。

### HardEvent（细粒度，生产用）

用 `SetFlag/WaitFlag` 控制特定 pipe 间依赖，允许无关 pipe 并行，通常带来 20-50% 性能提升。

NPU 三级流水线 MTE2(读) → V(计算) → MTE3(写)：

| Event | 含义 | 典型用法 |
|-------|------|----------|
| MTE2_V | 读完成，V 可计算 | DataCopy 后 Set，Muls 前 Wait |
| V_MTE3 | 计算完成，可写出 | Muls 后 Set，scatter 前 Wait |
| V_MTE2 | V 释放 input buffer | Muls 后 Set，下轮 DataCopy 前 Wait |
| MTE3_V | 写出完成，可复用 output buffer | scatter 后 Set，下轮 Muls 前 Wait |

**关键规则**:
- Init 中 Alloc，Release 中 Release，必须成对
- 外层循环开始前先 SetFlag 一次作为初始信号
- Process 结束前 WaitFlag 所有未完成的 event
- 精度问题时先全换 PipeBarrier 定位，确认正确后再替换回 HardEvent

### 流水线模式

SMALL 模板——批次间流水：
```
外层循环(按 yNumPerCalc 分批):
  WaitFlag<V_MTE2>; WaitFlag<MTE3_V>;
  内层处理每个 task...
  SetFlag<V_MTE2>; SetFlag<MTE3_V>;
```

LARGE 模板——task 间流水：
```
每个 task:
  WaitFlag<V_MTE2>;       DataCopy(in←GM);   SetFlag<MTE2_V>;
  WaitFlag<MTE2_V,MTE3_V>; Muls(out,in,fac);  SetFlag<V_MTE2,V_MTE3>;
  WaitFlag<V_MTE3>;       scatter(GM←out);   SetFlag<MTE3_V>;
```

## 优化手段（按优先级）

1. **HardEvent 替代 PipeBarrier** — 最大单项性能提升
2. **分支跳过不必要开销** — 可整除时跳过 atomic 和 clear
3. **批量处理** — 小数据量通过 yNumPerCalc 合并多次计算
4. **NC 切片而非 gather** — 大 NC 场景 scatter 模式更稳定
5. **合并 kernel 调用** — ClearOutput+Process 在同一 kernel 中顺序执行
6. **Tiling 结构体精简** — 删除不用的字段减少 host→device 传输量

## 内存与对齐

### DataCopy 对齐

所有 DataCopy 操作对齐到 32 字节（fp32 对齐 8 元素，fp16 对齐 16 元素）。NC 不对齐时：
- **host 侧 padding**: NC pad 到 ncAlign，输出后裁剪
- **DataCopyPad**: 高版本 SDK API，直接处理非对齐数据

### UB 容量保护

UB 总容量约 192-256KB（平台相关），需预留系统开销（通常减去 6KB）。NC 维度随输入变化时必须在 tiling 中动态计算。

## 散射与原子操作

多核 scatter 写入冲突的判断与处理：

| 场景 | 处理 |
|------|------|
| 可整除（无冲突） | 跳过 atomic 和 clear |
| 不可整除 | SetAtomicAdd + ClearOutput + SyncAll 全核同步 |
| fp16 原子累加精度不足 | 用 fp32 workspace 累加后 cast 回 |

## 精度验证规范

以 TBE 算子为对比基准，容差 fp32 atol=1e-5, fp16 atol=1e-2。

测试用例必须覆盖：

| 维度 | 目的 | 示例 |
|------|------|------|
| 模板分支 | 确保每个 tiling 分支都被测到 | 小 NC + 大 NC |
| atomic 路径 | 整除和非整除都要覆盖 | 56/7=整除, 7/3=非整除 |
| 规模范围 | 小/中/大 | scalar, 典型模型, 极端大 NC |
| 边界值 | 对齐边界、特殊尺寸 | NC=1, identity(in=out), 素数 HW |
| 数据类型 | 每个场景双类型 | fp32 + fp16 |

## 性能验证规范

- 随机生成 1000 个用例，要求 ≥95% 不劣于 TBE
- **两边必须对等**: 都通过 torch dispatch 调用
- **预分配缓冲区**: 计时循环内不应包含 tensor 分配
- **报告包含分布统计**: 加速比的最差/P5/中位/P95/最佳，及每个模板胜率

## 踩坑速查

| 现象 | 排查方向 |
|------|----------|
| 崩溃/挂死 | UB 溢出； DataCopy 越界 |
| 部分输出全零 | 多核任务分配错误； ClearOutput 后 SyncAll 缺失 |
| 精度偏差 | 分支不对齐； fp16 累加； in-place Muls |
| HardEvent 后精度错 | SetFlag/WaitFlag 配对遗漏，先全换 PipeBarrier 定位 |
| 性能数据失真 | 调用层级不对等；循环内分配 |
| 编译后行为未变 | 编译缓存，clean 重建； torch ext 未重编 |

## 交付检查清单

- [ ] Tiling 结构体无废弃字段，与 kernel 引用完全一致
- [ ] 精度测试覆盖所有模板分支、atomic/非 atomic、边界值、双数据类型
- [ ] 性能 1000 随机用例 ≥95%，报告含分模板胜率和加速比分布
- [ ] 性能测试两边均通过 torch dispatch
- [ ] README 与代码一致，无旧数据、无残留中间产物
