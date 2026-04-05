# 结果解读说明

这份文档用于回答三个问题：

1. 每类结果文件在说明什么。
2. 看不同图表时应该关注什么指标。
3. 如何把结果讲清楚，而不是只报一个分数。

## 一、先看哪几类结果

### 1. 数据与实验设计

建议先看：

- `outputs/tables/data_audit_overview.csv`
- `outputs/tables/time_window_design.csv`
- `outputs/tables/label_distribution.csv`

这部分回答：

- 数据是否可靠
- 时间窗是否合理
- 标签是否存在不平衡

### 2. 模型对比

建议再看：

- `outputs/tables/baseline_model_comparison.csv`
- `outputs/tables/optimized_model_comparison.csv`
- `outputs/tables/optimization_before_after_comparison.csv`

这部分回答：

- 不同模型谁更强
- 优化前后有没有提升
- 哪个指标上的提升最明显

### 3. 模型解释

建议最后看：

- `outputs/tables/feature_importance.csv`
- `outputs/figures/feature_importance.png`
- `outputs/figures/thesis/` 下的扩展解释图

这部分回答：

- 模型主要依赖哪些特征
- 不同特征与流失风险之间是什么关系

## 二、几类关键图表怎么理解

### `data_time_range.png`

看什么：

- 数据是否覆盖完整时间段
- 是否存在明显断层

说明什么：

- 数据具备按时间窗构建流失预测任务的基础

### `time_window_schema.png`

看什么：

- 观察期与预测期的切分是否清楚

说明什么：

- 特征来自过去，标签来自未来，流程没有时间泄漏

### `label_distribution.png`

看什么：

- 流失类占比是否过低

说明什么：

- 任务是否属于类别不平衡问题

### `baseline_roc_curve.png` / `optimized_roc_curve.png`

看什么：

- 曲线谁更靠近左上角
- AUC 是否提升

说明什么：

- 模型整体区分能力强不强

### `optimized_pr_curve.png`

看什么：

- 在召回率变化时，精确率下降速度如何

说明什么：

- 在少数类识别场景下，模型是否真的有效

### `best_model_confusion_matrix_*`

看什么：

- 漏判多少
- 误判多少

说明什么：

- 模型在业务上是否可用，代价主要落在哪一侧

### `feature_importance.png`

看什么：

- 排名前几的特征是谁
- 这些特征是否集中在某些行为维度

说明什么：

- 模型在决策时最依赖哪些用户行为信号

## 三、看表格时重点关注哪些字段

### `baseline_model_comparison.csv` / `optimized_model_comparison.csv`

重点看：

- `roc_auc`
- `f1`
- `precision`
- `recall`

解读方式：

- `roc_auc` 更适合看整体区分能力
- `f1` 更适合看不平衡场景下的综合效果
- `precision` 和 `recall` 用来判断误报与漏报的平衡

### `optimization_before_after_comparison.csv`

重点看：

- `roc_auc_gain`
- `f1_gain`

解读方式：

- 如果两个增益都为正，说明优化是有效的
- 如果只提升 `roc_auc` 但 `f1` 没提升，需要看是否牺牲了少数类识别

### `feature_selection_scores.csv`

重点看：

- 哪些特征的互信息得分更高

解读方式：

- 得分高说明该特征与标签关联更强，通常更值得保留进入优化建模

## 四、当前项目结果应该怎么讲

基于现有输出，可以按下面逻辑概括结果：

1. 原始行为日志经过时间窗切分和用户级聚合后，可以形成稳定的建模样本。
2. 基础模型已经能区分部分流失用户，说明行为数据中存在可学习信号。
3. 经过特征筛选、类别不平衡处理和参数搜索后，树模型优势更加明显。
4. 关键特征主要集中在活跃度、近期活跃和行为强度上，说明“是否持续活跃”是预测流失的核心信息来源。

## 五、汇报时建议避免的表达

不建议只说：

- “AUC 提高了”
- “模型效果不错”
- “XGBoost 最好”

更好的表达方式是：

- 模型在整体区分能力上提升了多少
- 模型对少数类流失用户的识别是否同步提升
- 哪些特征主导了模型判断
- 当前结果更适合用在哪类业务目标下

## 六、如果老师重点看代码和结果

建议重点展示下面几项：

1. `pipeline.py` 中时间窗切分与标签构建逻辑
2. `user_modeling_dataset.csv` 中的特征结构
3. `optimized_model_comparison.csv` 中的最终模型结果
4. `feature_importance.png` 中的关键特征

这样能把“数据怎么进来、模型怎么做、结果怎么出、为什么这么判断”串成一条完整链路。
