[![PyPI version](https://badge.fury.io/py/FLAML.svg)](https://badge.fury.io/py/FLAML)
![Conda version](https://img.shields.io/conda/vn/conda-forge/flaml)
[![Build](https://github.com/microsoft/FLAML/actions/workflows/python-package.yml/badge.svg)](https://github.com/microsoft/FLAML/actions/workflows/python-package.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/FLAML)](https://pypi.org/project/FLAML/)
[![Downloads](https://pepy.tech/badge/flaml)](https://pepy.tech/project/flaml)
[![](https://img.shields.io/discord/1025786666260111483?logo=discord&style=flat)](https://discord.gg/Cppx2vSPVP)

<!-- [![Join the chat at https://gitter.im/FLAMLer/community](https://badges.gitter.im/FLAMLer/community.svg)](https://gitter.im/FLAMLer/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) -->

# 快速高效的自动化机器学习与超参数调优算法库

<p align="center">
    <img src="https://github.com/microsoft/FLAML/blob/main/website/static/img/flaml.svg"  width=200>
    <br>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

:fire: FLAML 全面支持 [Microsoft Fabric Data Science](https://learn.microsoft.com/zh-cn/fabric/data-science/automated-machine-learning-fabric) 中的 AutoML 与超参数调优。此外，得益于 Microsoft Fabric 产品团队的贡献，我们引入了对 Python 3.11+ 的支持、一系列全新的算法评估器（Estimators），并深度集成了 MLflow。

:fire: **重要提醒**：[AutoGen](https://microsoft.github.io/autogen/) 已迁移至专属的[独立 GitHub 仓库](https://github.com/microsoft/autogen)。FLAML 代码库中已不再内置 `autogen` 模块——请直接使用 AutoGen 独立包。

## 什么是 FLAML

FLAML 是一个轻量级 Python 算法库，专注于机器学习（ML）与人工智能操作（AI Operations）的高效自动化。它能够自动化编排基于大语言模型、经典机器学习模型等工作流，并极致优化其性能表现。

- **低算力消耗的经济型自动化**：FLAML 支持在严格的计算资源与时间约束下，自动化完成模型选择与超参数优化。
- **高性价比的常见任务求解**：对于分类、回归等常见机器学习任务，FLAML 能够在极低算力开销下迅速为用户提供的高质量数据筛选出最优模型。它易于定制与扩展，用户可以在平滑的灵活性区间内随心设定所需的自定义程度。
- **快速经济的通用自动化调优**：支持各类复杂场景下的自动调优（例如：基座大模型的推理超参数、MLOps/LMOps 工作流配置、流水线、数学与统计模型、特定算法、计算实验参数、底层软件系统配置等），能够从容应对具有异构评估开销（Heterogeneous Evaluation Cost）、复杂约束条件、先验引导以及早停机制的超大规模搜索空间。

FLAML 凝聚了微软研究院（Microsoft Research）以及宾夕法尼亚州立大学、斯蒂文斯理工学院、华盛顿大学和滑铁卢大学等合作科研机构的[系列前沿学术研究成果](https://microsoft.github.io/FLAML/docs/Research/)。

此外，FLAML 在微软官方跨平台开源机器学习框架 [ML.NET](http://dot.net/ml) 中提供了成熟的 .NET 原生实现。

## 安装指南

FLAML 最新版本要求 **Python >= 3.10 且 < 3.14**。虽然其他 Python 版本可能支持核心组件运行，但无法保证全部模型的完整兼容性。可通过 `pip` 直接安装：

```bash
pip install flaml
```

默认情况下仅安装最基础的轻量依赖。您可以根据所需的功能按需安装扩展选项。例如，若需使用 [`automl`](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML) 模块所需的全部依赖，请执行：

```bash
pip install "flaml[automl]"
```

查阅更多安装选项请见[安装文档 (Installation)](https://microsoft.github.io/FLAML/docs/Installation)。各类[示例 Notebook](https://github.com/microsoft/FLAML/tree/main/notebook) 可能需要安装对应的特定扩展组件。

## 快速上手

- **三行代码即可运行**：作为 [scikit-learn 风格评估器](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML) 体验高效、经济的 AutoML 引擎：

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification")
```

- **限定基学习器**：将 FLAML 作为 XGBoost、LightGBM、随机森林（Random Forest）等模型的极速调参工具，或配合[自定义评估器](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#estimator-and-search-space)使用：

```python
automl.fit(X_train, y_train, task="classification", estimator_list=["lgbm"])
```

- **通用自定义函数调优**：针对任意[用户自定义函数 (UDF)](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function) 运行通用超参数调优：

```python
from flaml import tune

tune.run(
    evaluation_function, config={…}, low_cost_partial_config={…}, time_budget_s=3600
)
```

- **零样本 AutoML (Zero-shot AutoML)**：直接沿用 lightgbm、xgboost 等现有原生训练 API，同时享受 AutoML 在各具体任务上自动精选的高性能超参数配置：

```python
from flaml.default import LGBMRegressor

# 像平时使用 lightgbm.LGBMRegressor 一样直接使用 LGBMRegressor
estimator = LGBMRegressor()
# 超参数将根据输入的训练数据特征自动进行最优配置
estimator.fit(X_train, y_train)
```

## 文档指引

查阅 FLAML 的详尽官方文档请前往 [此处官方站点](https://microsoft.github.io/FLAML/)。

此外，您还可以了解：

- 围绕 FLAML 的[学术论文 (Research)](https://microsoft.github.io/FLAML/docs/Research) 与[技术博客 (Blogposts)](https://microsoft.github.io/FLAML/blog)。

- 加入官方 [Discord 交流社区](https://discord.gg/Cppx2vSPVP)。

- 查阅[开发者贡献指南 (Contributing Guide)](https://microsoft.github.io/FLAML/docs/Contribute)。

- ML.NET 相关文档与教程：[Model Builder 模型生成器](https://learn.microsoft.com/zh-cn/dotnet/machine-learning/tutorials/predict-prices-with-model-builder)、[ML.NET CLI 命令行](https://learn.microsoft.com/zh-cn/dotnet/machine-learning/tutorials/sentiment-analysis-cli) 以及 [AutoML API 接口指南](https://learn.microsoft.com/zh-cn/dotnet/machine-learning/how-to-guides/how-to-use-the-automl-api)。

## 参与贡献

本项目非常欢迎开源社区的贡献与建议。绝大多数贡献都需要您签署贡献者许可协议（CLA），声明您有权且确实授予我们使用您贡献内容的权利。详情请访问：<https://cla.opensource.microsoft.com>。

如果您刚接触 GitHub，[此处](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/)提供了关于参与 GitHub 开源协同开发的详尽指南。

当您提交 Pull Request 时，CLA 机器人将自动核验您是否需要签署 CLA，并在 PR 中进行相应的状态标识（如状态检查与评论提示）。只需根据机器人的提示完成确认即可。在所有采用微软 CLA 的代码仓库中，此步骤仅需签署一次。

本项目遵循 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。欲了解更多信息，请查阅[行为准则常见问题解答 (FAQ)](https://opensource.microsoft.com/codeofconduct/faq/)，如有其他疑问或建议，亦可联系 [opencode@microsoft.com](mailto:opencode@microsoft.com)。

## 贡献者墙

<a href="https://github.com/microsoft/flaml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/flaml&max=204" />
</a>

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
