# Boltz2 Kibble 案例组织案例

## 项目信息

- 案例: boltz2
- 位置: /data/kibble/cases/boltz2/
- 特点: 已有案例的增量更新（非创建）

## 关键经验

### 1. 更新已有案例的流程

1. diff 所有关键文件找出不同步的部分
2. 同步变更文件到 deploy/
3. case.toml 只更新变化的字段（updated/scope/strengths/limitations）
4. devlog 追加新阶段，不重写旧内容
5. chatlog 摘要追加新轮工作

### 2. 权重文件处理

权重不归档到项目中，仅提供下载指南:
- models/README.md 含 HuggingFace 官方源 + 镜像站双通道
- 文件清单含大小和用途
- 自动下载和自定义缓存目录说明

### 3. 多余文件清理

需要清理的典型文件:
- 测试输出目录 (npu_prot/, cpu_prot/)
- NPU 编译缓存 (kernel_meta/, fusion_result.json)
- Python 缓存 (__pycache__/)
- 旧审查文件 (CPU_vs_NPU_review*.md)
- 与 NPU 迁移无关的上游文档 (evaluation.md, training.md 等)
