# 项目速览

这份文档用于帮助第一次接触这个仓库的人，在几分钟内看懂项目的目标、输入输出、核心流程和结果文件。

## 这个项目在做什么

项目解决的是一个标准的用户流失预测问题：

- 输入：用户在一段时间内的行为日志
- 输出：用户是否会在后续一段时间内停止活跃

这里的关键不是“直接拿日志训练”，而是先把行为流水整理成用户级画像，再交给分类模型学习。

## 原始数据长什么样

原始数据位于 `archive/.csv`，是按行为记录组织的日志表，不是按用户汇总后的宽表。

核心字段包括：

- `user_id`
- `item_id`
- `behavior_type`
- `user_geohash`
- `item_category`
- `time`

一名用户可能对应很多条记录，所以必须先做聚合，才能进入建模阶段。

## 项目的核心流程

### 1. 数据审计

先检查：

- 数据量
- 时间范围
- 缺失情况
- 行为分布

对应输出：

- `outputs/tables/data_audit_overview.csv`
- `outputs/tables/data_columns.csv`
- `outputs/tables/behavior_distribution.csv`

### 2. 时间窗划分

项目采用固定时间窗：

- 观察期：用于提取用户历史行为特征
- 预测期：用于定义用户是否流失

对应输出：

- `outputs/tables/time_window_design.csv`

### 3. 标签构建

标签逻辑很直接：

- 预测期仍有行为：未流失
- 预测期没有行为：流失

对应输出：

- `outputs/tables/label_distribution.csv`

### 4. 特征工程

项目会把观察期日志转换成用户级特征，主要包含：

- 活跃度特征
- 近期行为特征
- 行为结构特征
- 兴趣广度特征

对应输出：

- `outputs/tables/user_modeling_dataset.csv`
- `outputs/tables/feature_description_table.csv`
- `outputs/tables/feature_descriptive_statistics.csv`

### 5. 模型训练与优化

当前实现比较了四类模型：

- `LR`
- `RF`
- `XGBoost`
- `LightGBM`

先做 baseline，再做特征筛选和参数优化。

对应输出：

- `outputs/tables/baseline_model_comparison.csv`
- `outputs/tables/optimized_model_comparison.csv`
- `outputs/tables/optimization_before_after_comparison.csv`

### 6. 结果可视化

为了方便分析模型效果，项目还会生成一组图表：

- ROC 曲线
- 混淆矩阵
- 特征重要性图
- 标签分布图
- 扩展分析图

## 建议阅读顺序

如果想快速理解项目，建议按这个顺序看：

1. `README.md`
2. `outputs/tables/stage_summary.csv`
3. `outputs/tables/time_window_design.csv`
4. `outputs/tables/feature_description_table.csv`
5. `outputs/tables/optimized_model_comparison.csv`
6. `outputs/figures/optimized_roc_curve.png`
7. `outputs/figures/feature_importance.png`

## 两个入口脚本分别做什么

### `run_pipeline.py`

负责从原始数据出发，生成基础建模结果。

### `run_thesis_visualization.py`

负责在已有结果基础上生成更完整的扩展图表。文件名保留历史命名，但现在可以把它理解成“扩展可视化入口”。

## 看结果时重点关注什么

### 如果看整体区分能力

优先看：

- `optimized_model_comparison.csv`
- `optimized_roc_curve.png`

### 如果看少数类识别效果

优先看：

- `optimized_model_comparison.csv` 里的 `precision`、`recall`、`f1`
- `optimized_pr_curve.png`

### 如果看模型为什么会这样判断

优先看：

- `feature_importance.csv`
- `feature_importance.png`
- `outputs/figures/thesis/` 下的扩展解释图

## 这个仓库应该如何理解

这个仓库不是“只有结果截图”的展示仓库，而是完整保留了：

- 原始输入
- 特征构建逻辑
- 模型训练逻辑
- 结果表
- 可视化结果

因此老师或其他读者既能看最后结论，也能顺着代码和结果回溯整个实现过程。
