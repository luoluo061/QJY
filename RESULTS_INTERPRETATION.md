# 结果图表位置与含义说明

本文档的目标是让阅读者快速明白：

- 每类图表在论文中应该放在哪一章
- 它解决的是什么问题
- 图里看到什么，应该怎样解释

## 1. 第4章：数据审计与实验设计

这一章的任务不是证明模型多强，而是证明实验设计合理、数据可用。

### `outputs/figures/thesis/data_time_range.png`

建议标题：

- 淘宝用户行为数据时间范围分布图

建议放置位置：

- 第4章“数据来源与时间范围”

代表意义：

- 说明数据覆盖时间是连续的，不是零散采样。
- 说明样本具备做时间窗口划分的基础。

推荐表述：

- 原始行为日志覆盖 31 天，时间连续且无明显断层，因此可以按照固定观察期与预测期构建流失预警实验。

### `outputs/figures/thesis/time_window_schema.png`

建议标题：

- 观察期与预测期时间窗口示意图

建议放置位置：

- 第4章“实验窗口设计”

代表意义：

- 明确告诉读者哪些数据用于构造特征，哪些数据用于定义标签。
- 体现“先观察、后预测”的时间因果顺序。

推荐表述：

- 本文采用前 24 天作为观察期、后 7 天作为预测期，先利用观察期行为构建用户画像，再根据预测期是否仍有行为定义流失标签。

### `outputs/figures/thesis/label_distribution.png`

建议标题：

- 流失标签分布图

建议放置位置：

- 第4章“标签构建结果”

代表意义：

- 说明样本不平衡是否严重。
- 为后续引出 class weight、特征筛选、模型调优提供依据。

推荐表述：

- 流失样本占比较低，说明该任务属于典型类别不平衡分类问题，后续模型优化需要重点考虑召回率与 F1 的提升。

### `outputs/figures/thesis/behavior_type_distribution.png`

建议标题：

- 行为类型分布图

建议放置位置：

- 第4章“原始数据概况”

代表意义：

- 说明原始行为数据并不均匀，存在主行为占多数的特征。
- 解释为什么需要把行为日志聚合成更稳定的用户级特征。

### `outputs/figures/thesis/missing_value_overview.png`

建议标题：

- 字段缺失值概况图

建议放置位置：

- 第4章“数据质量分析”

代表意义：

- 说明哪些字段缺失严重。
- 解释为何 `user_geohash` 不能直接作为核心建模字段。

## 2. 第5章：模型效果对比

这一章要回答的问题是：模型效果如何，优化有没有意义。

### `outputs/tables/baseline_model_comparison.csv`

建议放置位置：

- 第5章“Baseline 模型结果”

代表意义：

- 展示四类模型在初始条件下的整体表现。
- 用于说明最初哪类模型更适合这个任务。

### `outputs/figures/thesis/baseline_roc_curve.png`

建议放置位置：

- 第5章“Baseline 模型结果”

代表意义：

- 展示 baseline 条件下四种模型的区分能力。

### `outputs/tables/optimized_model_comparison.csv`

建议放置位置：

- 第5章“模型优化结果”

代表意义：

- 展示加入类别不平衡处理、特征筛选和参数调优后的模型表现。

### `outputs/figures/thesis/optimized_roc_curve.png`

建议放置位置：

- 第5章“模型优化结果”

代表意义：

- 展示优化后四种模型的 ROC 对比。
- 适合说明谁的整体区分能力最强。

### `outputs/figures/thesis/optimized_pr_curve.png`

建议放置位置：

- 第5章“模型优化结果”

代表意义：

- 在类别不平衡背景下，PR 曲线比 ROC 更能说明对流失类的识别质量。

推荐表述：

- 由于流失样本占比较低，仅比较 ROC_AUC 不足以全面反映模型对少数类的识别能力，因此本文进一步使用 PR 曲线进行补充分析。

### `outputs/tables/optimization_before_after_comparison.csv`

建议放置位置：

- 第5章“优化前后对比”

代表意义：

- 直接比较每个模型在优化前后的 AUC 与 F1 变化。
- 用于说明优化策略是否有效。

## 3. 第5章：最优模型评估

这一部分不是比较模型，而是深入解释最优模型表现。

### `outputs/figures/thesis/best_model_confusion_matrix_count.png`

代表意义：

- 展示绝对分类数量。
- 适合说明识别出多少流失用户、误判了多少未流失用户。

### `outputs/figures/thesis/best_model_confusion_matrix_normalized.png`

代表意义：

- 展示分类比例结构。
- 适合解释流失类召回率和未流失类误判情况。

### `outputs/figures/thesis/threshold_precision_recall_f1_curve.png`

代表意义：

- 展示阈值变化时精确率、召回率和 F1 的变化。
- 适合解释为什么论文采用当前阈值，或为什么业务场景可以调整阈值。

### `outputs/figures/thesis/best_model_calibration_curve.png`

代表意义：

- 展示模型输出概率是否可信。
- 如果曲线接近对角线，说明预测概率更接近真实概率。

## 4. 第5章：模型解释与关键特征分析

这一部分用来回答“模型为什么这么判断”。

### `outputs/figures/thesis/all_models_permutation_importance.png`

代表意义：

- 比较不同模型对特征重要性的共同关注点。
- 适合说明哪些特征在多模型下都稳定重要。

### `outputs/figures/thesis/best_model_feature_importance.png`

代表意义：

- 用于展示最优模型内部最重要的特征排序。

推荐表述：

- 最优模型对近期活跃程度、活跃天数和行为新近性相关特征最为敏感，说明用户是否持续保持近期活跃是流失预警中的核心信号。

### `outputs/figures/thesis/boxplot_*.png`

重点关注：

- `boxplot_recency_days.png`
- `boxplot_last_7d_actions.png`
- `boxplot_active_days.png`

代表意义：

- 通过流失组与未流失组的分布差异，展示关键特征的群体差别。

推荐表述：

- 与未流失用户相比，流失用户通常具有更长的最近行为间隔、更少的近期行为次数和更低的活跃天数，说明活跃衰减是流失形成的重要前兆。

### `outputs/figures/thesis/pdp_recency_days.png`

代表意义：

- 说明随着 `recency_days` 增大，预测流失概率如何变化。

### `outputs/figures/thesis/pdp_last_7d_actions.png`

代表意义：

- 说明近期行为次数增加是否会显著降低流失概率。

### `outputs/figures/thesis/pdp_active_days.png`

代表意义：

- 说明用户活跃天数提升是否会抑制流失风险。

## 5. 附录或模型优化小节：训练过程图

这类图的价值在于证明参数和训练轮次不是拍脑袋定的。

### LR

- `lr_learning_curve.png`
- `lr_validation_curve_c.png`

作用：

- 说明逻辑回归的样本规模敏感性与正则化参数影响。

### RF

- `rf_oob_error_vs_estimators.png`
- `rf_validation_curve_max_depth.png`

作用：

- 说明随机森林树数量和树深的选择依据。

### XGBoost

- `xgboost_train_valid_metrics.png`

作用：

- 观察 boosting rounds 增加时训练集和验证集指标是否分离。
- 用于判断是否出现过拟合。

### LightGBM

- `lightgbm_metric_vs_iterations.png`
- `lightgbm_plot_importance.png`

作用：

- 展示 LightGBM 的迭代收敛过程与树模型内部的重要特征。

## 6. 如果正文篇幅有限，优先保留哪些图

建议正文优先保留：

1. `time_window_schema.png`
2. `label_distribution.png`
3. `optimized_roc_curve.png`
4. `optimized_pr_curve.png`
5. `best_model_confusion_matrix_normalized.png`
6. `best_model_feature_importance.png`
7. `boxplot_recency_days.png`
8. `pdp_recency_days.png`

其余训练过程图可以放附录。

## 7. 一句话总结这些图表的作用

这些图表共同完成四件事：

1. 证明数据和实验设计是合理的。
2. 证明模型优化确实带来了效果提升。
3. 解释模型到底依赖哪些用户行为信号做出判断。
4. 帮助论文读者从“数据输入”一路看到“建模机制”和“结果含义”。
