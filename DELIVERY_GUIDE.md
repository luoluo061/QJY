# 项目交付清单与阅读顺序

本文档用于回答三个问题：

1. 这个交付包里到底有什么。
2. 第一次接触这个项目的人应该先看什么。
3. 哪些文件是代码，哪些文件是结果，哪些文件是论文直接可用材料。

## 1. 推荐阅读顺序

如果你是第一次看这个项目，建议按下面顺序阅读：

1. `README.md`
2. `outputs/tables/stage_summary.csv`
3. `outputs/tables/time_window_design.csv`
4. `outputs/tables/baseline_model_comparison.csv`
5. `outputs/tables/optimized_model_comparison.csv`
6. `outputs/tables/thesis/figure_table_catalog.md`
7. `RESULTS_INTERPRETATION.md`

这样可以先快速知道：

- 研究对象是什么
- 系统流程怎么走
- 当前实验结果怎样
- 图表该怎么放到论文里

## 2. 代码文件说明

### `run_pipeline.py`

基础实验主入口。

用途：

- 从原始 CSV 重新生成建模数据与基础实验结果。

### `run_thesis_visualization.py`

论文图表生成入口。

用途：

- 基于已有 pipeline 结果生成论文级图表。

### `src/churn_pipeline/pipeline.py`

核心实验流水线模块。

负责：

- 数据审计
- 时间窗口划分
- 标签构建
- 特征工程
- baseline 建模
- 优化建模
- 基础结果导出

### `src/churn_pipeline/thesis_visualization.py`

论文图表模块。

负责：

- 实验过程图
- 模型结果图
- 模型解释图
- 训练过程图
- 图表目录说明

## 3. 数据文件说明

### `archive/.csv`

原始实验数据。

注意：

- 这是整个实验流程唯一的数据输入源。
- 所有建模数据、标签、图表与结果表都由它推导而来。

## 4. 结果文件说明

### 4.1 `outputs/tables`

这是基础结果表目录，适合看“系统做了什么、结果如何”。

优先看这些文件：

- `data_audit_overview.csv`
- `time_window_design.csv`
- `label_distribution.csv`
- `feature_description_table.csv`
- `feature_descriptive_statistics.csv`
- `baseline_model_comparison.csv`
- `optimized_model_comparison.csv`
- `optimization_before_after_comparison.csv`
- `stage_summary.csv`

### 4.2 `outputs/figures`

这是基础图目录，适合看“数据和模型大致表现如何”。

优先看这些文件：

- `daily_activity_volume.png`
- `correlation_heatmap.png`
- `baseline_roc_curve.png`
- `optimized_roc_curve.png`
- `feature_importance.png`

### 4.3 `outputs/tables/thesis`

这是论文级表格目录，适合直接服务论文写作。

重点文件：

- `baseline_model_comparison.csv`
- `optimized_model_comparison.csv`
- `optimization_before_after_comparison.csv`
- `best_model_feature_importance.csv`
- `all_models_permutation_importance_top10.csv`
- `figure_table_catalog.md`

### 4.4 `outputs/figures/thesis`

这是论文级图表目录，适合直接插入论文或答辩 PPT。

内容覆盖：

- 实验过程图
- ROC / PR / 混淆矩阵 / 校准曲线
- 特征重要性与箱线图
- PDP 图
- 训练过程曲线

## 5. 如果只想快速看懂结果

建议只看下面这组文件：

### 第一步：看实验设计

- `outputs/tables/time_window_design.csv`
- `outputs/figures/thesis/time_window_schema.png`
- `outputs/figures/thesis/label_distribution.png`

### 第二步：看模型效果

- `outputs/tables/optimized_model_comparison.csv`
- `outputs/figures/thesis/optimized_roc_curve.png`
- `outputs/figures/thesis/optimized_pr_curve.png`
- `outputs/figures/thesis/best_model_confusion_matrix_normalized.png`

### 第三步：看为什么模型会这样判断

- `outputs/figures/thesis/best_model_feature_importance.png`
- `outputs/figures/thesis/boxplot_recency_days.png`
- `outputs/figures/thesis/boxplot_last_7d_actions.png`
- `outputs/figures/thesis/pdp_recency_days.png`

## 6. 如果要用于论文写作

直接使用下面两个文件即可定位图表：

- `outputs/tables/thesis/figure_table_catalog.md`
- `RESULTS_INTERPRETATION.md`

它们分别解决：

- 每张图表文件名、标题、推荐章节、结论摘要
- 每张关键图到底要怎么解释、放在哪里最合适

## 7. 一页式理解

这个交付包本质上包含三层内容：

1. 原始数据层
   即 `archive/.csv`
2. 实验系统层
   即 `run_pipeline.py`、`run_thesis_visualization.py` 和 `src/`
3. 论文结果层
   即 `outputs/tables`、`outputs/figures` 及其 `thesis` 子目录

所以别人拿到这个包以后，不仅能看到“最后的图”，还能追溯“这些图是怎么来的”。
