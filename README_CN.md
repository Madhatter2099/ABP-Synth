# abp-synth

> **合成运动员生物护照（ABP）时序数据生成器 — 面向反兴奋剂研究**

[![Python](https://img.shields.io/badge/Python-≥3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 项目简介

**abp-synth** 能够生成逼真的合成纵向血液指标时序数据（血红蛋白 HGB 与网织红细胞百分比 RET），用于反兴奋剂研究。本工具从真实的 [ABPS](https://cran.r-project.org/package=ABPS) 数据集提取统计基线，并合成带有可选 EPO 违规特征注入的时间序列。

### 为什么需要？

真实运动员血液护照数据几乎不可获得（隐私保护、标签极度稀缺）。**abp-synth** 提供：

- 基于 Sottas et al. (2008) 文献与 ABPS 包的**生理学基线**
- 带均值回归的多元随机游走（AR(1)）生成**时间自相关序列**
- 可配置的**EPO 两阶段异常特征**（HGB 阶跃上升 + RET 骤降）注入
- 开箱即用的**出版级可视化**

---

## 安装

```bash
pip install -e .
```

## 快速上手

### Python API（5 行搞定）

```python
from abp_synth import generate_dataset

dataset = generate_dataset(n_normal=1000, n_doping=100, seed=42)
print(dataset.summary())
dataset.save("./output")
```

### 命令行

```bash
# 生成完整数据集
abp-synth generate --n-normal 10000 --n-doping 1000 --output ./output

# 仅提取基线
abp-synth baseline --output ./output

# 生成可视化图表
abp-synth visualize --input ./output --output ./figures
```

---

## 核心概念

### 基线提取

从 ABPS 数据集中提取 4 个核心血液指标的统计参数（均值向量 $\mu$ 与协方差矩阵 $\Sigma$）。

### 合成时序生成

正常运动员使用带均值回归的随机游走：

$$\mathbf{x}_t = \mathbf{x}_{t-1} + \boldsymbol{\epsilon}_t + \alpha(\boldsymbol{\mu} - \mathbf{x}_{t-1})$$

### EPO 异常注入

- **阶段 A（注射期）**：HGB 线性上升 +1.5 ~ +3.0 g/dL
- **阶段 B（停药期）**：RET 正弦骤降 −0.25% ~ −0.50%

---

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 参考文献

1. Sottas, P.E. et al. (2008). *Biostatistics*, 9(2), 285-296.
2. Sharpe, K. et al. (2006). *Haematologica*, 91(12), 1603-1610.
3. WADA (2019). *ABP Operating Guidelines*.

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
