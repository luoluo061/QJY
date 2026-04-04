# 论文初稿与实验结果整合指南

本文档用于把当前代码仓库中的真实实验结果，准确嵌入到你的论文初稿中。

重点解决四个问题：

1. 论文初稿当前章节结构是什么。
2. 当前实验结果应该插入到哪些位置。
3. 每张图和每张表具体说明什么。
4. 初稿中哪些描述与当前真实实验结果不一致，必须修改。

## 1. 从初稿中识别出的当前论文结构

从 [论文初稿.doc](E:\QJY\论文初稿.doc) 抽取到的结构看，当前论文大致是：

1. 第一章 绪论
2. 第二章 相关理论与技术基础
3. 第三章 用户流失预警模型实验设计
4. 第四章 模型实现与实验过程
5. 第五章 模型优化与结果分析
6. 第六章 结论与展望

这套结构和当前仓库输出是匹配的，说明不需要重写目录，只需要把实验口径和图表内容嵌进去。

## 2. 必须先改的几处不一致

这些地方如果不改，论文文字会和当前代码、图表、结果表明显冲突。

### 2.1 用户规模描述不一致

初稿中抽取到的描述：

- 数据集记录了约 100 万用户的真实行为日志。

当前真实实验结果：

- 原始 CSV 实际唯一用户数为 `10000`。
- 进入观察期建模的用户数为 `9904`。

建议修改为：

- 本研究使用的实际实验文件包含 10000 个唯一用户的行为记录，其中 9904 个用户在观察期内有行为并进入建模样本。

### 2.2 时间窗口描述不一致

初稿中抽取到的窗口痕迹：

- `2014-11-18至2014-12-08`
- `2014-12-09至2014-12-15`
- `2014-12-16至2014-12-18`

当前真实代码口径：

- 观察期：`2014-11-18 00:00:00` 到 `2014-12-11 23:00:00`
- 预测期：`2014-12-12 00:00:00` 到 `2014-12-18 23:00:00`

建议修改为：

- 本文采用前 24 天作为观察期，最后 7 天作为预测期。观察期用于构造用户级特征，预测期用于定义流失标签。

### 2.3 特征数量描述不一致

初稿中抽取到的描述：

- 从四个维度构造了 13 个特征。

当前代码实际特征：

- 当前建模表中实际使用的是一组扩展后的用户级特征，约 29 个核心特征字段，不是 13 个。

建议修改为：

- 本文在活跃度、近期行为、行为结构、类目广度与时间新近性等维度上构建了多项用户级特征，并结合特征筛选保留关键变量进入模型。

### 2.4 最优模型描述不一致

初稿中抽取到的描述偏向：

- 重点优化 LightGBM
- LightGBM AUC 从 `0.848` 提升到 `0.851`

当前真实实验结果：

- Baseline 按 `ROC_AUC` 最优的是 `LR`
- Optimized 按 `ROC_AUC` 最优的是 `XGBoost`，`ROC_AUC=0.8971`
- Optimized 按 `F1` 最优的是 `RF`，`F1=0.3844`

建议修改为：

- 若以 ROC_AUC 为主要指标，优化后 XGBoost 综合表现最佳。
- 若以 F1 为重点指标，优化后随机森林在流失类识别平衡性上表现更优。

## 3. 论文里这些图表到底怎么用

下面按章节说明。

## 4. 第三章：实验设计部分怎么用结果

第三章要证明的是实验设计合理，而不是模型多强。

### 建议插入内容

#### 图：`outputs/figures/thesis/data_time_range.png`

放置位置：

- 第三章 `3.1 数据来源与说明`

用途：

- 说明原始行为数据覆盖的时间范围是连续的。
- 为后续时间窗口划分提供依据。

推荐写法：

- 图中显示，原始行为日志覆盖 2014 年 11 月 18 日至 2014 年 12 月 18 日，共连续 31 天，说明数据具备按时间窗口构建流失预警任务的基础。

#### 图：`outputs/figures/thesis/time_window_schema.png`

放置位置：

- 第三章 `3.2.1 时间窗口划分`

用途：

- 直观展示观察期与预测期。
- 明确避免数据泄漏。

推荐写法：

- 本文将前 24 天划为观察期，最后 7 天划为预测期。观察期用于提取用户画像，预测期用于判断用户是否流失，从而保证特征构造与标签定义在时间上严格分离。

#### 表：`outputs/tables/time_window_design.csv`

放置位置：

- 紧接时间窗口图之后

用途：

- 给出精确时间范围与天数。

#### 图：`outputs/figures/thesis/label_distribution.png`

放置位置：

- 第三章 `3.2.3 流失标签定义`

用途：

- 说明流失样本占比较低，存在类别不平衡。

推荐写法：

- 由标签分布可见，流失用户占比约为 4.81%，说明该任务属于典型的不平衡二分类问题，后续模型优化中需要关注少数类识别能力。

#### 图：`outputs/figures/thesis/missing_value_overview.png`

放置位置：

- 第三章 `3.2.2 数据清洗`

用途：

- 说明缺失值情况。
- 解释 `user_geohash` 为什么不能作为核心特征直接使用。

#### 图：`outputs/figures/thesis/behavior_type_distribution.png`

放置位置：

- 第三章 `3.1 数据来源与说明` 或 `3.2 数据预处理`

用途：

- 说明行为类型分布高度不均衡。
- 解释原始行为日志为何需要上卷到用户级特征。

## 5. 第四章：模型实现与实验过程怎么用结果

第四章的重点是“系统如何做”，因此建议结合代码逻辑和中间结果表。

### 建议使用的表

#### 表：`outputs/tables/data_audit_overview.csv`

放置位置：

- 第四章 `4.2 数据预处理实现`

用途：

- 直接说明审计后的数据量、用户数、类目数、时间范围、缺失情况。

#### 表：`outputs/tables/feature_description_table.csv`

放置位置：

- 第四章 `4.3 特征工程实现`

用途：

- 用来解释每个特征是什么。
- 这是“系统到底提取了什么”的核心表。

#### 表：`outputs/tables/feature_descriptive_statistics.csv`

放置位置：

- 第四章 `4.3 特征工程实现`

用途：

- 展示特征分布概况。

#### 图：`outputs/figures/correlation_heatmap.png`

放置位置：

- 第四章 `4.3 特征工程实现`

用途：

- 用来说明特征与标签、特征与特征之间的相关关系。
- 适合引出后面的特征筛选。

### 第四章建议强调什么

这一章建议用文字强调：

- 系统先做了真实 CSV 审计，而不是假设字段结构。
- 标签是基于观察期用户全集构造，避免遗漏预测期无行为用户。
- 特征工程围绕活跃度、近期行为、行为结构、时间新近性等维度展开。

## 6. 第五章：模型优化与结果分析怎么用结果

第五章是图表使用最密集的章节。

### 6.1 Baseline 结果

#### 表：`outputs/tables/thesis/baseline_model_comparison.csv`

放置位置：

- 第五章 `5.1 Baseline 模型比较`

用途：

- 给出四模型初始指标对比。

#### 图：`outputs/figures/thesis/baseline_roc_curve.png`

放置位置：

- 紧跟 baseline 对比表

用途：

- 用图形化方式展示 baseline 模型区分能力。

建议写法：

- 在 baseline 阶段，四种模型均具备一定区分能力，其中逻辑回归在 ROC_AUC 指标上表现最好，说明该任务中线性可分信号已经较强。

### 6.2 优化后结果

#### 表：`outputs/tables/thesis/optimized_model_comparison.csv`

放置位置：

- 第五章 `5.2 模型优化结果`

用途：

- 展示优化后四种模型的结果。

#### 表：`outputs/tables/thesis/optimization_before_after_comparison.csv`

放置位置：

- 第五章 `5.2 模型优化结果`

用途：

- 展示各模型优化前后变化。

#### 图：`outputs/figures/thesis/optimized_roc_curve.png`

放置位置：

- 第五章 `5.2 模型优化结果`

用途：

- 比较优化后模型 ROC 表现。

#### 图：`outputs/figures/thesis/optimized_pr_curve.png`

放置位置：

- 紧跟 optimized ROC 图之后

用途：

- 强调在不平衡分类下对流失类的识别能力。

建议写法：

- 虽然多个模型在 ROC_AUC 上接近，但从 PR 曲线看，优化后模型对少数类流失用户的识别能力存在更明显差异，因此仅用 AUC 判断模型优劣是不够的。

### 6.3 最优模型评估

如果你准备把 `XGBoost` 作为正文最优模型，建议使用下面这组图。

#### 图：`outputs/figures/thesis/best_model_confusion_matrix_count.png`

用途：

- 展示绝对分类数量。

#### 图：`outputs/figures/thesis/best_model_confusion_matrix_normalized.png`

用途：

- 展示分类结构，尤其是流失类召回率。

#### 图：`outputs/figures/thesis/threshold_precision_recall_f1_curve.png`

用途：

- 说明阈值变化对 Precision、Recall、F1 的影响。
- 用来支撑“为什么当前阈值可接受”。

#### 图：`outputs/figures/thesis/best_model_calibration_curve.png`

用途：

- 说明预测概率是否可靠。

### 6.4 模型解释与业务含义

这一部分是你的论文从“会分类”走向“能解释”的关键。

#### 图：`outputs/figures/thesis/all_models_permutation_importance.png`

用途：

- 对比不同模型对关键特征的依赖程度。
- 用于说明稳定重要特征。

#### 图：`outputs/figures/thesis/best_model_feature_importance.png`

用途：

- 展示最优模型内部的重要特征排序。

建议写法：

- 结果表明，模型最依赖的并不是单一点击总量，而是用户近期是否仍保持持续活跃，这与电商流失形成机制一致。

#### 图：`outputs/figures/thesis/boxplot_recency_days.png`

用途：

- 展示流失组和未流失组在最近行为间隔上的差异。

#### 图：`outputs/figures/thesis/boxplot_last_7d_actions.png`

用途：

- 展示近期活跃程度差异。

#### 图：`outputs/figures/thesis/boxplot_active_days.png`

用途：

- 展示整体活跃稳定性差异。

#### 图：`outputs/figures/thesis/pdp_recency_days.png`

用途：

- 展示 `recency_days` 增大时，预测流失风险如何变化。

#### 图：`outputs/figures/thesis/pdp_last_7d_actions.png`

用途：

- 展示近期行为次数提升是否显著降低流失风险。

#### 图：`outputs/figures/thesis/pdp_active_days.png`

用途：

- 展示用户活跃天数与流失风险的边际关系。

## 7. 哪些图适合正文，哪些图适合附录

### 建议正文保留

优先放这些：

1. `time_window_schema.png`
2. `label_distribution.png`
3. `baseline_model_comparison.csv`
4. `optimized_model_comparison.csv`
5. `optimized_roc_curve.png`
6. `optimized_pr_curve.png`
7. `best_model_confusion_matrix_normalized.png`
8. `best_model_feature_importance.png`
9. `boxplot_recency_days.png`
10. `boxplot_last_7d_actions.png`
11. `pdp_recency_days.png`

### 建议附录保留

这些更偏训练过程或补充证明：

1. `lr_learning_curve.png`
2. `lr_validation_curve_c.png`
3. `rf_oob_error_vs_estimators.png`
4. `rf_validation_curve_max_depth.png`
5. `xgboost_train_valid_metrics.png`
6. `lightgbm_metric_vs_iterations.png`
7. `lightgbm_plot_importance.png`
8. `all_models_permutation_importance.png`

## 8. 你现在最应该怎么改初稿

建议按下面顺序改：

1. 先统一实验口径
   把用户数、时间窗口、特征数量、最优模型描述改成与当前真实结果一致。
2. 再改第三章
   把实验设计部分和 `time_window_schema.png`、`label_distribution.png` 对齐。
3. 再改第五章
   以 `optimized_model_comparison.csv`、`optimized_roc_curve.png`、`optimized_pr_curve.png` 作为主结果。
4. 最后补解释部分
   用 feature importance、箱线图、PDP 图回答“为什么会流失”。

## 9. 一句话结论

当前仓库里的图和表已经足够支撑一篇完整的本科论文实验部分，但前提是你要先把初稿里与真实实验结果不一致的旧描述替换掉，再按本指南把图表放回对应章节。
