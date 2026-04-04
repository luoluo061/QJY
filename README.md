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

## 数据、结果与图表怎么理解

如果你不想来回翻多个说明文件，只看这一节也可以理解本项目的大部分内容。

### 1. 原始数据是什么

本项目的原始输入是淘宝用户行为日志。

它不是“每个用户一行”的表，而是“每次行为一行”的流水数据。也就是说，一名用户看商品、加购、收藏、购买，都可能分别对应多条记录。

当前真实实验文件包含这些字段：

- `user_id`：用户编号
- `item_id`：商品编号
- `behavior_type`：行为类型编码
- `user_geohash`：地理位置编码
- `item_category`：商品类目编号
- `time`：行为发生时间，精确到小时

这类数据的特点是：

- 行为记录很多，但每条记录都很细碎
- 不能直接拿原始日志喂给分类模型
- 必须先把行为流水整理成“每个用户一行”的用户级建模数据

### 2. 什么叫观察期、预测期、流失标签

用户流失预测和普通分类不一样，它必须遵守时间顺序。

本项目先把时间切成两段：

- 观察期：用来观察用户过去做了什么
- 预测期：用来判断用户后面还来不来

当前实验口径是：

- 观察期：`2014-11-18 00:00:00` 到 `2014-12-11 23:00:00`
- 预测期：`2014-12-12 00:00:00` 到 `2014-12-18 23:00:00`

流失标签定义为：

- 预测期内仍有行为：未流失
- 预测期内完全无行为：流失

这个定义的意义是：

- 让“流失”变成可计算、可复现的二分类问题
- 保证特征来自过去，标签来自未来，避免数据泄漏

### 3. 系统是怎么把原始日志变成模型输入的

系统的核心思想是给每个用户做一份“行为画像”。

它会从观察期行为中构造一系列用户级特征，例如：

- 活跃度特征
  - 总行为数
  - 活跃天数
  - 活跃小时数
- 近期行为特征
  - 最近 7 天行为数
  - 最近 3 天行为数
  - 最近一次行为距离观察期结束的间隔天数
- 行为结构特征
  - 不同行为类型的次数和占比
- 兴趣广度特征
  - 去重商品数
  - 去重类目数

这样，模型最终看到的就不是一条条零散日志，而是每个用户的一组特征值。

### 4. 为什么要比较四种模型

本项目比较了四种常见分类模型：

- `LR`：逻辑回归
- `RF`：随机森林
- `XGBoost`
- `LightGBM`

这样做的原因不是为了“凑数量”，而是为了回答论文中的一个核心问题：

> 不同机器学习模型在电商用户流失预警任务上的效果差异如何。

不同模型的特点大致可以这样理解：

- 逻辑回归更容易解释，适合看基础线性规律
- 随机森林适合挖掘非线性关系，鲁棒性较好
- XGBoost 和 LightGBM 属于提升树模型，通常在复杂表格数据上表现更强

### 5. 当前实验结果怎么看

当前结果不能只看一个数字，而要分层理解。

#### 第一层：看 baseline

文件：

- [baseline_model_comparison.csv](E:\QJY\outputs\tables\baseline_model_comparison.csv)
- [baseline_roc_curve.png](E:\QJY\outputs\figures\thesis\baseline_roc_curve.png)

它回答的是：

- 在不做复杂优化的前提下，四种模型的基础表现如何

当前结果显示：

- Baseline 按 `ROC_AUC` 最优的是 `LR`

这说明：

- 仅凭基础用户行为特征，模型已经能较好区分流失与未流失用户

#### 第二层：看优化后结果

文件：

- [optimized_model_comparison.csv](E:\QJY\outputs\tables\optimized_model_comparison.csv)
- [optimized_roc_curve.png](E:\QJY\outputs\figures\thesis\optimized_roc_curve.png)
- [optimized_pr_curve.png](E:\QJY\outputs\figures\thesis\optimized_pr_curve.png)
- [optimization_before_after_comparison.csv](E:\QJY\outputs\tables\optimization_before_after_comparison.csv)

它回答的是：

- 做了特征筛选和参数调优以后，模型有没有提升

当前结果显示：

- 按 `ROC_AUC`，优化后最优模型是 `XGBoost`
- 按 `F1`，优化后最优模型是 `RF`

这说明：

- 如果更关注整体区分能力，可以优先看 XGBoost
- 如果更关注少数类流失用户识别的平衡性，可以重点看 RF 的 F1 表现

#### 第三层：看最优模型是否真的“有用”

文件：

- [best_model_confusion_matrix_count.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_count.png)
- [best_model_confusion_matrix_normalized.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_normalized.png)
- [threshold_precision_recall_f1_curve.png](E:\QJY\outputs\figures\thesis\threshold_precision_recall_f1_curve.png)
- [best_model_calibration_curve.png](E:\QJY\outputs\figures\thesis\best_model_calibration_curve.png)

这些图回答的是：

- 模型到底识别出了多少流失用户
- 是否误伤了太多未流失用户
- 换一个阈值会不会更合适
- 模型输出的风险概率是否可信

### 6. 每类图到底在讲什么

下面用尽量直白的方式解释。

#### 数据时间范围图

文件：

- [data_time_range.png](E:\QJY\outputs\figures\thesis\data_time_range.png)

代表什么：

- 它展示原始行为记录在时间上的分布情况

意义：

- 说明数据覆盖时间连续，适合做时间窗口实验

#### 时间窗口示意图

文件：

- [time_window_schema.png](E:\QJY\outputs\figures\thesis\time_window_schema.png)

代表什么：

- 它展示观察期和预测期是怎么切分的

意义：

- 说明实验是“先看过去，再预测未来”

#### 标签分布图

文件：

- [label_distribution.png](E:\QJY\outputs\figures\thesis\label_distribution.png)

代表什么：

- 它展示流失样本和未流失样本的数量差异

意义：

- 说明任务是否存在类别不平衡

#### 行为类型分布图

文件：

- [behavior_type_distribution.png](E:\QJY\outputs\figures\thesis\behavior_type_distribution.png)

代表什么：

- 它展示不同类型行为的数量差异

意义：

- 说明原始日志不是均匀分布的，行为结构本身就是一个重要信号

#### 缺失值概况图

文件：

- [missing_value_overview.png](E:\QJY\outputs\figures\thesis\missing_value_overview.png)

代表什么：

- 它展示每个字段缺失的比例

意义：

- 说明哪些字段可以直接用，哪些字段不能过度依赖

#### ROC 曲线

文件：

- [baseline_roc_curve.png](E:\QJY\outputs\figures\thesis\baseline_roc_curve.png)
- [optimized_roc_curve.png](E:\QJY\outputs\figures\thesis\optimized_roc_curve.png)

代表什么：

- 比较模型整体区分能力

意义：

- 曲线越靠左上，说明模型越能把流失用户和未流失用户分开

#### PR 曲线

文件：

- [optimized_pr_curve.png](E:\QJY\outputs\figures\thesis\optimized_pr_curve.png)

代表什么：

- 比较模型在少数类流失用户上的识别质量

意义：

- 对不平衡分类问题特别重要

#### 混淆矩阵

文件：

- [best_model_confusion_matrix_count.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_count.png)
- [best_model_confusion_matrix_normalized.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_normalized.png)

代表什么：

- 模型分对了多少，分错了多少

意义：

- 最直观地说明漏判与误判

#### 特征重要性图

文件：

- [best_model_feature_importance.png](E:\QJY\outputs\figures\thesis\best_model_feature_importance.png)
- [all_models_permutation_importance.png](E:\QJY\outputs\figures\thesis\all_models_permutation_importance.png)

代表什么：

- 模型最依赖哪些特征来判断用户会不会流失

意义：

- 它回答“为什么模型会这样预测”

#### 箱线图

文件：

- [boxplot_recency_days.png](E:\QJY\outputs\figures\thesis\boxplot_recency_days.png)
- [boxplot_last_7d_actions.png](E:\QJY\outputs\figures\thesis\boxplot_last_7d_actions.png)
- [boxplot_active_days.png](E:\QJY\outputs\figures\thesis\boxplot_active_days.png)

代表什么：

- 比较流失用户和未流失用户在某个特征上的分布差异

意义：

- 帮助把模型结论翻译成用户行为差异

#### PDP 图

文件：

- [pdp_recency_days.png](E:\QJY\outputs\figures\thesis\pdp_recency_days.png)
- [pdp_last_7d_actions.png](E:\QJY\outputs\figures\thesis\pdp_last_7d_actions.png)
- [pdp_active_days.png](E:\QJY\outputs\figures\thesis\pdp_active_days.png)

代表什么：

- 某个特征变化时，模型预测流失风险会怎样变化

意义：

- 帮助理解“特征影响方向”

### 7. 这些结果最终说明了什么

从当前实验结果看，可以得到几个核心认识：

1. 用户行为日志确实可以支持流失预警研究。
2. 用户最近是否还活跃，是最重要的流失信号之一。
3. 活跃天数、近期行为次数、最近行为间隔，是模型最重视的变量。
4. 优化后的模型比 baseline 更适合这个任务。
5. 模型不仅能输出预测结果，还可以通过重要性图、箱线图和 PDP 图给出行为层面的解释。

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

## 9. 关于原始数据文件

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
