# <项目名> 部署产物

本目录包含 <项目名> 昇腾 NPU 适配版的完整可部署项目。

## 目录结构

```
deploy/
├── README.md              # 本文件
└── <project-name>/        # 完整项目目录（可直接部署运行）
    ├── README.md          # 环境准备、运行命令、性能数据
    ├── src/               # 源代码
    ├── tests/             # 测试用例
    ├── scripts/           # 验证脚本
    ├── models/            # 模型权重（需下载）
    └── ...                # 其他项目文件
```

## 快速开始

```bash
cd <project-name>/
cat README.md              # 查看完整部署指南
```

详细的环境搭建、模型下载、运行命令和性能测试说明见 [`<project-name>/README.md`](<project-name>/README.md)。
