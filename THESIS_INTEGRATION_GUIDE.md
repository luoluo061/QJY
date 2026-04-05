# 扩展图表与汇报素材说明

文件名保留历史命名，方便兼容现有目录结构。当前这份文档的作用是说明扩展输出目录中的图表和表格分别适合展示什么内容。

## 一、这份文档适合什么时候看

当你需要：

- 准备老师查看项目时的演示材料
- 整理更完整的结果展示页面
- 快速定位 `outputs/tables/thesis/` 和 `outputs/figures/thesis/` 中的内容

就看这份文档。

## 二、扩展目录里主要有什么

### `outputs/figures/thesis/`

这里放的是扩展图表，通常比基础图更多，适合做完整展示。

典型内容包括：

- 时间范围图
- 时间窗示意图
- 标签分布图
- ROC / PR 曲线
- 混淆矩阵
- 校准曲线
- 特征重要性图
- 箱线图
- PDP 图
- 训练过程曲线

### `outputs/tables/thesis/`

这里放的是扩展结果表，适合展示对比关系和补充说明。

典型内容包括：

- baseline 模型对比表
- 优化模型对比表
- 优化前后对比表
- 最优模型特征重要性表
- 多模型 permutation importance 对比表
- 图表目录表

## 三、推荐展示顺序

如果老师要快速看项目，建议按这个顺序展示：

1. `data_time_range.png`
2. `time_window_schema.png`
3. `label_distribution.png`
4. `optimized_model_comparison.csv`
5. `optimized_roc_curve.png`
6. `optimized_pr_curve.png`
7. `best_model_confusion_matrix_normalized.png`
8. `best_model_feature_importance.png`

这样能形成一条完整叙述链：

- 数据覆盖是否连续
- 时间窗如何设计
- 标签是否平衡
- 模型整体效果如何
- 少数类识别效果如何
- 最终模型主要依赖什么特征

## 四、各类扩展图表的作用

### 1. 实验设计类

适合展示：

- `data_time_range.png`
- `time_window_schema.png`
- `missing_value_overview.png`
- `behavior_type_distribution.png`

核心作用：

- 说明数据可用
- 说明时间窗设计合理
- 说明原始日志为什么要做用户级聚合

### 2. 模型效果类

适合展示：

- `baseline_roc_curve.png`
- `optimized_roc_curve.png`
- `optimized_pr_curve.png`
- `best_model_confusion_matrix_count.png`
- `best_model_confusion_matrix_normalized.png`

核心作用：

- 说明 baseline 与优化模型之间的差异
- 说明整体区分能力和少数类识别能力
- 说明误判和漏判结构

### 3. 模型解释类

适合展示：

- `all_models_permutation_importance.png`
- `best_model_feature_importance.png`
- `boxplot_*.png`
- `pdp_*.png`
- `best_model_calibration_curve.png`

核心作用：

- 说明模型依赖哪些行为信号
- 说明关键特征变化对预测结果的影响
- 说明模型输出概率是否稳定可信

## 五、建议重点讲的结论

基于当前扩展输出，汇报时建议聚焦这几条：

1. 项目不是简单堆模型，而是先完成时间窗建模链路，再做模型比较。
2. 优化后树模型对复杂行为模式的利用更充分。
3. 活跃天数、近期行为次数、最近活跃间隔等特征是主要判断依据。
4. 在类别不平衡任务里，不能只看 AUC，需要同时看 F1、PR 曲线和混淆矩阵。

## 六、如果只展示最少内容

最小展示集建议为：

- `outputs/tables/optimized_model_comparison.csv`
- `outputs/figures/thesis/optimized_roc_curve.png`
- `outputs/figures/thesis/optimized_pr_curve.png`
- `outputs/figures/thesis/best_model_feature_importance.png`

这组内容已经足够支撑一次比较完整的项目讲解。
