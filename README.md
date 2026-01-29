# 2025 数学建模美赛代码仓库

包含模板和多个图表可视化工具等。

## 项目架构

```text
MCM_ICM2025
├── mcm_kit/                   # 模板源代码主包
│   ├── evaluation/            # 评估工具：提供多模型数据评估方案
│   ├── optimization           # 优化工具：暂无
│   ├── prediction/            # 预测工具：暂无
│   ├── visualization/         # 可视化工具：暂无
│   └── __init__.py
├── tests/                     # 单元测试 / 使用例目录
│   ├── test_evaluation.py
│   └── ...
├── pyproject.toml                   # 安装配置文件
└── README.md
```

## 模板

### 快速安装

使用 python 安装项目内的模板库。

在项目根目录执行以下代码：
```bash
pip install -e .
```

### 使用

安装完毕后，直接在代码里引入即可

```py
from mcm_kit import *
# ...
```