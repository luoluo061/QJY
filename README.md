# 基于机器学习的电商平台用户流失预警模型构建与优化

本项目用于支撑本科论文《基于机器学习的电商平台用户流失预警模型构建与优化》。

它不是一个泛化的软件产品，而是一套面向论文实验的可复现实验包。核心目标是把淘宝用户行为日志转化为用户级流失预警建模流程，并导出可直接写入论文的表格、图和结果说明。

## 快速认识这个仓库

如果你第一次打开这个仓库，可以先看这一段。

### 项目目标

本项目想解决的问题是：

> 能不能根据用户过去一段时间的电商行为，提前识别出未来可能流失的用户。

这里的“流失”定义为：

> 用户在预测期内不再产生任何行为。

### 数据概况

当前真实实验文件审计结果如下：

- 原始数据文件：`archive/.csv`
- 时间范围：`2014-11-18 00:00:00` 到 `2014-12-18 23:00:00`
- 原始记录数：`12,256,906`
- 原始唯一用户数：`10,000`
- 观察期建模用户数：`9,904`
- 流失样本占比：`4.81%`

### 系统流程

整个系统按下面顺序运行：

1. 读取真实 CSV 并做字段审计
2. 划分观察期和预测期
3. 按观察期用户全集构造标签
4. 生成用户级特征
5. 训练 baseline 模型
6. 做特征筛选、参数调优和优化建模
7. 导出论文图表和结果表

### 当前核心结果

在当前实验口径下：

- Baseline 按 `ROC_AUC` 最优模型：`LR`
- Optimized 按 `ROC_AUC` 最优模型：`XGBoost`，`ROC_AUC=0.8971`
- Optimized 按 `F1` 最优模型：`RF`，`F1=0.3844`
- 关键特征主要集中在：
  - `active_day_ratio`
  - `active_days`
  - `last_3d_actions`
  - `last_7d_actions`
  - `recency_days`

### 结果在哪里看

最重要的产出都在 `outputs/` 下：

- 基础结果表：`outputs/tables`
- 基础图：`outputs/figures`
- 论文图表：`outputs/figures/thesis`
- 论文结果表：`outputs/tables/thesis`

### 初学者建议阅读顺序

1. [BEGINNER_GUIDE.md](E:\QJY\BEGINNER_GUIDE.md)
2. [DELIVERY_GUIDE.md](E:\QJY\DELIVERY_GUIDE.md)
3. [THESIS_INTEGRATION_GUIDE.md](E:\QJY\THESIS_INTEGRATION_GUIDE.md)
4. [RESULTS_INTERPRETATION.md](E:\QJY\RESULTS_INTERPRETATION.md)
5. [figure_table_catalog.md](E:\QJY\outputs\tables\thesis\figure_table_catalog.md)

## 1. 这个项目是什么

项目围绕一个明确问题展开：

- 输入：淘宝用户行为日志 CSV
- 处理：数据审计、时间窗口划分、标签构建、用户级特征工程、模型训练、模型优化、结果可视化
- 输出：建模数据表、实验结果表、论文图表、图表目录说明

项目当前已经完成两条主线：

1. `pipeline`
   负责从真实原始数据生成建模数据、baseline 与优化模型结果。
2. `thesis_visualization`
   负责基于已有建模结果进一步生成论文可直接使用的图表和结果目录。

## 2. 系统是怎么进行的

整个系统按如下顺序运行：

1. 读取真实 CSV，不预设字段结构。
2. 自动审计字段名、字段类型、时间范围、缺失值与行为编码分布。
3. 根据真实时间范围设计观察期与预测期。
4. 以“观察期用户全集”为建模对象构造用户级样本。
5. 用“预测期是否还有行为”定义流失标签。
6. 构造行为频次、活跃度、近期活跃、类目广度、时间跨度等用户级特征。
7. 训练 baseline 模型：`LR`、`RF`、`XGBoost`、`LightGBM`。
8. 在样本不平衡处理、特征筛选、参数调优后训练 optimized 模型。
9. 导出论文图表，包括 ROC、PR、混淆矩阵、特征重要性、PDP、训练过程曲线等。
10. 生成图表目录说明，标注每张图适合放在哪一章、代表什么意义。

## 3. 当前实验口径

当前真实数据实验口径如下：

- 原始数据文件：`archive/.csv`
- 原始数据时间范围：`2014-11-18 00:00:00` 到 `2014-12-18 23:00:00`
- 观察期：`2014-11-18 00:00:00` 到 `2014-12-11 23:00:00`
- 预测期：`2014-12-12 00:00:00` 到 `2014-12-18 23:00:00`
- 原始唯一用户数：`10000`
- 观察期建模用户数：`9904`
- 流失标签定义：预测期无任何行为记为流失

说明：

- `10000` 是全量原始数据的唯一用户数。
- `9904` 是进入建模阶段的观察期用户数。
- 有 `96` 个用户只在预测期出现、不在观察期出现，因此无法构造观察期特征，不纳入建模样本。

## 4. 目录结构

推荐先按这个顺序看目录：

```text
E:\QJY
├─ README.md
├─ DELIVERY_GUIDE.md
├─ RESULTS_INTERPRETATION.md
├─ run_pipeline.py
├─ run_thesis_visualization.py
├─ archive
│  └─ .csv
├─ src
│  └─ churn_pipeline
│     ├─ pipeline.py
│     └─ thesis_visualization.py
└─ outputs
   ├─ tables
   │  ├─ *.csv
   │  └─ thesis
   │     ├─ *.csv
   │     └─ figure_table_catalog.md
   └─ figures
      ├─ *.png
      └─ thesis
         └─ *.png
```

## 5. 入口脚本

### 5.1 生成基础实验结果

```bash
py run_pipeline.py
```

作用：

- 读取真实 CSV
- 生成数据审计表
- 生成时间窗口表
- 生成用户级建模数据
- 训练 baseline 与 optimized 模型
- 输出基础结果图表

### 5.2 生成论文图表

```bash
py run_thesis_visualization.py
```

作用：

- 复用 `pipeline` 产出
- 生成实验过程图
- 生成论文模型结果图
- 生成特征解释图
- 生成训练过程图
- 生成图表目录说明

## 6. 核心产出在哪里

### 6.1 基础结果

位置：`outputs/tables` 与 `outputs/figures`

主要包括：

- 数据审计表
- 标签分布表
- 特征说明表
- 特征描述统计表
- baseline 模型对比表
- optimized 模型对比表
- 优化前后对比表
- 基础 ROC、混淆矩阵、相关性图、特征重要性图

### 6.2 论文图表

位置：`outputs/tables/thesis` 与 `outputs/figures/thesis`

主要包括：

- 实验过程图
- 论文级模型结果图
- 解释性图表
- 按模型区分的训练过程图
- 图表目录说明文件

## 7. 最重要的三个说明文件

如果要把项目打包给别人看，建议对方按下面顺序阅读：

1. [README.md](E:\QJY\README.md)
   先理解项目是什么、流程怎么走、目录怎么看。
2. [BEGINNER_GUIDE.md](E:\QJY\BEGINNER_GUIDE.md)
   面向初学者解释什么是用户流失预警、这些数据和图表分别代表什么、应该怎么理解这些实验结果。
3. [DELIVERY_GUIDE.md](E:\QJY\DELIVERY_GUIDE.md)
   查看交付包里每一类文件分别是什么、该先看哪几个。
4. [THESIS_INTEGRATION_GUIDE.md](E:\QJY\THESIS_INTEGRATION_GUIDE.md)
   查看当前实验结果如何嵌入论文初稿，哪些旧描述必须改，哪些图表适合放正文、哪些适合放附录。
5. [RESULTS_INTERPRETATION.md](E:\QJY\RESULTS_INTERPRETATION.md)
   查看每张关键图表在论文中该放哪里、代表什么含义、可以怎么解释。

## 8. 当前结论摘要

基于当前实验结果，可以先得到这些核心结论：

- 淘宝用户行为日志可以支撑用户流失预警研究。
- 基于观察期行为构造的用户级特征对流失识别具有较强区分能力。
- baseline 阶段 `LR` 的 `ROC_AUC` 最优。
- 优化后 `XGBoost` 的 `ROC_AUC` 最优，`RF` 的 `F1` 最优。
- 关键特征集中在用户近期活跃程度、活跃天数与行为新近性上。

## 9. 打包建议

如果你要把这个项目完整交付给导师、答辩老师或其他评阅人，建议保留：

- `README.md`
- `DELIVERY_GUIDE.md`
- `RESULTS_INTERPRETATION.md`
- `run_pipeline.py`
- `run_thesis_visualization.py`
- `src/`
- `outputs/`
- `archive/.csv`

如果只需要“论文结果交付包”，可以重点保留：

- `README.md`
- `DELIVERY_GUIDE.md`
- `RESULTS_INTERPRETATION.md`
- `outputs/tables/`
- `outputs/figures/`

## 9.1 关于原始数据文件

仓库默认不提交原始数据大文件：

- `archive/.csv`
- `archive.zip`

原因：

- 这两个文件体积过大，超过 GitHub 普通 Git 提交的单文件大小限制。
- 当前仓库定位为“代码 + 结果 + 文档”的论文交付仓库，而不是原始数据托管仓库。

如果需要复现实验，请将原始 CSV 按当前目录结构放回：

- `archive/.csv`

## 10. 一句话理解这个项目

这是一个把淘宝用户行为日志转化为“可复现实验流程 + 可直接写进论文的结果图表”的本科论文实验包。
