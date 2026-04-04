# 初学者阅读指南：这个项目到底在做什么

如果你刚接触机器学习、用户流失预测或者论文实验项目，这份文档就是给你看的。

它会尽量用通俗的话解释：

1. 这个项目研究的到底是什么问题。
2. 数据从哪里来，长什么样。
3. 系统一步一步做了什么。
4. 为什么会有这么多图和表。
5. 每类图表到底代表什么含义。
6. 看完这个仓库以后，你应该能得到什么结论。

## 1. 先用一句人话说清楚这个项目

这个项目研究的是：

> 能不能根据电商平台用户过去一段时间的行为，提前判断哪些用户接下来可能会“流失”。

这里的“流失”，不是指账户注销，而是指：

> 在未来一段时间里，这个用户不再产生任何行为。

所以这个项目的核心逻辑其实很像：

- 先看用户之前做了什么
- 再看他后面还来不来
- 然后让机器学习模型从中找规律

## 2. 什么叫“用户流失预警”

“预警”两个字很重要。

它不是等用户已经离开以后再统计，而是希望：

- 在用户还没有彻底流失之前
- 根据他最近行为变少、活跃下降、兴趣收缩等信号
- 提前识别出风险用户

如果这个模型真的可用，那么电商平台就可以：

- 提前发优惠券
- 做精准召回
- 做运营干预
- 减少老用户流失

所以这类研究有明确业务价值。

## 3. 数据从哪里来

本项目使用的是淘宝用户行为日志数据。

在当前实验文件中，真实读取到的数据字段包括：

- `user_id`：用户编号
- `item_id`：商品编号
- `behavior_type`：行为类型编码
- `user_geohash`：用户地理位置编码
- `item_category`：商品类目编号
- `time`：行为发生时间，精确到小时

你可以在这里看到数据审计结果：

- [data_audit_overview.csv](E:\QJY\outputs\tables\data_audit_overview.csv)
- [data_columns.csv](E:\QJY\outputs\tables\data_columns.csv)

## 4. 这些原始数据有什么特点

原始数据不是“每行一个用户”，而是：

> 每一行代表某个用户在某个时间点，对某个商品发生了一次行为。

所以它更像“行为流水”。

例如一个用户今天点了商品、明天加购、后天购买，那么这会对应多行记录。

因此机器学习不能直接拿原始日志一行一行去训练，而要先把它们整理成：

> 每个用户一行的用户级特征表

这就是特征工程要做的事情。

## 5. 系统到底做了哪些步骤

这个项目的完整流程可以概括成下面 6 步。

### 第一步：读取真实 CSV 并做数据审计

系统不会先假设字段结构，而是先检查：

- 列名是什么
- 每列是什么类型
- 时间字段格式对不对
- 行为编码有哪些值
- 缺失值多不多
- 数据时间范围有多长

这一步的意义是：

- 避免“想当然”
- 保证实验建立在真实数据上
- 防止后面标签构建和特征工程做错

### 第二步：划分观察期和预测期

这是流失预测里最重要的设计之一。

当前项目的做法是：

- 观察期：`2014-11-18` 到 `2014-12-11`
- 预测期：`2014-12-12` 到 `2014-12-18`

为什么要这样分？

因为我们想模拟真实业务场景：

- 先看到用户“过去”的行为
- 再预测他“未来”会不会流失

如果把未来数据也混进特征里，就会发生“数据泄漏”，实验结果会虚高，不可信。

### 第三步：定义流失标签

标签就是模型最终要预测的答案。

在本项目中：

- 如果一个用户在预测期内还有任何行为，记为未流失
- 如果一个用户在预测期内完全没有行为，记为流失

这样就把“流失”这个抽象概念，变成了可以计算的二分类标签。

### 第四步：构造用户级特征

这一部分可以理解成“给每个用户做一份行为画像”。

系统构造的特征大致包括这些类型：

- 活跃度特征
  - 总行为数
  - 活跃天数
  - 活跃小时数
- 近期行为特征
  - 最近 7 天行为数
  - 最近 3 天行为数
  - 最近一次行为距离观察期结束有多少天
- 行为结构特征
  - 各行为类型出现次数和占比
- 兴趣广度特征
  - 去重商品数
  - 去重类目数
- 稳定性特征
  - 活跃跨度
  - 平均每日行为数

你可以直接看这个表：

- [feature_description_table.csv](E:\QJY\outputs\tables\feature_description_table.csv)

它相当于“特征词典”。

### 第五步：训练模型

本项目先训练四个 baseline 模型：

- LR：逻辑回归
- RF：随机森林
- XGBoost
- LightGBM

然后再做优化，包括：

- 样本不平衡处理思路
- 特征筛选
- 参数调优

为什么要比较多个模型？

因为不同模型擅长的模式不同：

- 逻辑回归适合比较清晰、线性趋势较强的数据
- 随机森林适合处理非线性关系
- XGBoost 和 LightGBM 适合复杂模式挖掘，通常效果更强

### 第六步：导出论文图表

模型训练完以后，项目不会只给一个数字结果，而是会输出很多图和表。

这是因为论文不只是要“报一个分数”，还要回答：

- 数据是什么样的
- 模型怎么设计的
- 优化有没有效果
- 为什么模型会这样判断
- 这些结果意味着什么

## 6. 仓库里为什么会有这么多图和表

因为一个完整的论文实验，至少要回答四类问题：

1. 数据是否可靠
2. 实验设计是否合理
3. 模型效果是否好
4. 模型结论是否可解释

不同的图表就是用来回答不同问题的。

## 7. 这些“表”到底是干什么的

很多初学者会觉得表格太多，不知道看哪个。

其实可以这样理解。

### `data_audit_overview.csv`

这是“数据体检报告”。

它告诉你：

- 一共有多少行数据
- 有多少用户
- 有多少商品和类目
- 时间范围是什么
- 哪些字段缺失严重

意义：

- 证明数据是真实可用的
- 也是后续实验设计的基础

### `time_window_design.csv`

这是“实验时间切分说明书”。

它告诉你：

- 哪段时间是观察期
- 哪段时间是预测期
- 每段有多少天

意义：

- 解释模型到底是用什么时间范围做预测的

### `label_distribution.csv`

这是“流失标签统计表”。

它告诉你：

- 流失和未流失各有多少样本
- 流失占比是多少

意义：

- 判断是否存在类别不平衡

### `feature_description_table.csv`

这是“特征说明表”。

意义：

- 解释每个变量到底表示什么
- 是写论文方法部分时最重要的表之一

### `baseline_model_comparison.csv`

这是“基础模型成绩单”。

意义：

- 比较四个模型在未做复杂优化时的表现

### `optimized_model_comparison.csv`

这是“优化后模型成绩单”。

意义：

- 看优化之后模型有没有提升

### `optimization_before_after_comparison.csv`

这是“前后对照表”。

意义：

- 最适合写论文里“优化有效性分析”

## 8. 这些“图”到底怎么看

这是最关键的部分。

下面尽量用教学式的方式解释。

## 9. 实验过程图：它们不是在说模型强，而是在说实验设计合理

### 9.1 数据时间范围图

文件：

- [data_time_range.png](E:\QJY\outputs\figures\thesis\data_time_range.png)

它是什么：

- 一张展示不同日期行为记录量的时间分布图

你看它时要关注什么：

- 时间是不是连续的
- 中间有没有突然断掉
- 是否有明显异常日期

它的意义：

- 说明这个数据集不是零散抽样，而是连续时间日志
- 证明它适合做“观察期-预测期”的时序预测实验

对小白的理解方式：

- 如果数据只是一堆乱七八糟的时间点，就很难谈“先观察再预测”
- 只有时间连续，流失预警才有意义

### 9.2 观察期/预测期时间窗口图

文件：

- [time_window_schema.png](E:\QJY\outputs\figures\thesis\time_window_schema.png)

它是什么：

- 一张把时间切成两段的示意图

你要怎么理解：

- 左边那段：给模型看的历史行为
- 右边那段：用来判断用户有没有流失

它的意义：

- 避免未来信息泄漏到过去
- 保证实验像真实预测，而不是“事后诸葛亮”

### 9.3 标签分布图

文件：

- [label_distribution.png](E:\QJY\outputs\figures\thesis\label_distribution.png)

它是什么：

- 一张展示流失用户和未流失用户数量的柱状图

你要看什么：

- 两类是不是差很多

它的意义：

- 如果流失用户明显更少，模型就可能偏向“全部预测成未流失”
- 所以后续不能只看准确率，还要看召回率、F1、PR 曲线

### 9.4 行为类型分布图

文件：

- [behavior_type_distribution.png](E:\QJY\outputs\figures\thesis\behavior_type_distribution.png)

它是什么：

- 展示不同 `behavior_type` 记录数的分布图

意义：

- 告诉你用户行为不是均匀分布的
- 某些行为大量出现，某些行为很少

为什么重要：

- 这会影响特征构造方式
- 也解释了为什么不能直接拿原始日志做建模

### 9.5 缺失值概况图

文件：

- [missing_value_overview.png](E:\QJY\outputs\figures\thesis\missing_value_overview.png)

它是什么：

- 每个字段缺失率的图

你要看什么：

- 哪些字段缺失特别多

意义：

- 缺失严重的字段不能简单拿来当核心特征
- 比如 `user_geohash` 缺失高，就不能过度依赖它

## 10. 模型结果图：它们回答“模型到底准不准”

### 10.1 ROC 曲线

文件：

- [baseline_roc_curve.png](E:\QJY\outputs\figures\thesis\baseline_roc_curve.png)
- [optimized_roc_curve.png](E:\QJY\outputs\figures\thesis\optimized_roc_curve.png)

它是什么：

- 比较模型区分正负样本能力的曲线

给小白的理解：

- 你可以把它理解为“模型把流失用户排在前面的能力”
- 曲线越靠左上越好
- AUC 越大越好

它的意义：

- 适合看模型整体区分能力
- 不依赖固定阈值

但要注意：

- 在类别不平衡问题里，ROC 不能说明全部问题
- 所以还要看 PR 曲线

### 10.2 PR 曲线

文件：

- [optimized_pr_curve.png](E:\QJY\outputs\figures\thesis\optimized_pr_curve.png)

它是什么：

- 展示 Precision 和 Recall 之间关系的曲线

给小白的理解：

- Precision：你说某人会流失，结果他说对的比例有多高
- Recall：真正会流失的人里，你抓到了多少

为什么重要：

- 流失用户通常是少数类
- PR 曲线比 ROC 更适合看少数类识别能力

### 10.3 混淆矩阵

文件：

- [best_model_confusion_matrix_count.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_count.png)
- [best_model_confusion_matrix_normalized.png](E:\QJY\outputs\figures\thesis\best_model_confusion_matrix_normalized.png)

它是什么：

- 一张四格表，统计模型“猜对了多少、猜错了多少”

小白理解法：

- 左上：没流失，模型也说没流失
- 右上：没流失，模型却说流失了
- 左下：流失了，模型却没发现
- 右下：流失了，模型也成功识别到了

它的意义：

- 比单一指标更直观
- 特别适合解释“漏判”和“误判”

### 10.4 阈值-Precision/Recall/F1 曲线

文件：

- [threshold_precision_recall_f1_curve.png](E:\QJY\outputs\figures\thesis\threshold_precision_recall_f1_curve.png)

它是什么：

- 展示阈值变化时三个指标怎么变化

为什么需要它：

- 模型输出的往往不是直接的“流失/未流失”，而是一个概率
- 最后要把概率变成类别，就要设阈值，比如 `0.5`

它的意义：

- 说明为什么选当前阈值
- 如果业务更重视召回，就可以把阈值往下调
- 如果业务更重视精确率，就可以把阈值往上调

### 10.5 概率校准曲线

文件：

- [best_model_calibration_curve.png](E:\QJY\outputs\figures\thesis\best_model_calibration_curve.png)

它是什么：

- 检查模型给出的“概率”到底靠不靠谱

小白理解法：

- 如果模型说“这批用户有 80% 的流失概率”
- 那么真实情况里最好也接近 80% 真会流失

它的意义：

- 如果以后模型要给运营系统输出风险分值，这张图很重要

## 11. 模型解释图：它们回答“模型为什么这样判断”

### 11.1 全模型 permutation importance 对比

文件：

- [all_models_permutation_importance.png](E:\QJY\outputs\figures\thesis\all_models_permutation_importance.png)

它是什么：

- 比较不同模型各自最依赖哪些特征

小白理解法：

- 如果把某个特征打乱以后，模型性能明显下降
- 说明这个特征很重要

意义：

- 不是只看一个模型
- 而是看多个模型是否都重视同一批特征

### 11.2 最优模型特征重要性图

文件：

- [best_model_feature_importance.png](E:\QJY\outputs\figures\thesis\best_model_feature_importance.png)

它是什么：

- 最优模型内部的特征重要性排序图

意义：

- 直接告诉你模型最看重什么

在本项目里它通常说明：

- 近期是否活跃
- 最近一次行为离现在多久
- 活跃天数多不多

这些是判断流失风险的核心信号

### 11.3 箱线图

文件示例：

- [boxplot_recency_days.png](E:\QJY\outputs\figures\thesis\boxplot_recency_days.png)
- [boxplot_last_7d_actions.png](E:\QJY\outputs\figures\thesis\boxplot_last_7d_actions.png)
- [boxplot_active_days.png](E:\QJY\outputs\figures\thesis\boxplot_active_days.png)

它是什么：

- 比较“流失用户”和“未流失用户”在某个特征上的分布差异

小白理解法：

- 这不是看模型，而是在看两类用户本身有什么不同

意义：

- 它能帮助你从业务角度解释结果
- 比如流失用户是不是最近更久没来
- 是不是最近 7 天活动明显更少

### 11.4 PDP 图

文件：

- [pdp_recency_days.png](E:\QJY\outputs\figures\thesis\pdp_recency_days.png)
- [pdp_last_7d_actions.png](E:\QJY\outputs\figures\thesis\pdp_last_7d_actions.png)
- [pdp_active_days.png](E:\QJY\outputs\figures\thesis\pdp_active_days.png)

它是什么：

- PDP 的全称是 Partial Dependence Plot，偏依赖图

小白理解法：

- 它想回答的问题是：
  - 如果只改变这一个特征
  - 模型预测的流失概率会怎么变化

比如：

- `recency_days` 越大，是不是流失概率越高
- `last_7d_actions` 越多，是不是流失概率越低

它的意义：

- 帮助理解“特征影响方向”
- 不只是知道“重要”，还知道“怎么影响”

## 12. 训练过程图：它们回答“为什么这样调参数”

这些图更适合放附录，但对学习也很有用。

### LR 学习曲线和验证曲线

文件：

- [lr_learning_curve.png](E:\QJY\outputs\figures\thesis\lr_learning_curve.png)
- [lr_validation_curve_c.png](E:\QJY\outputs\figures\thesis\lr_validation_curve_c.png)

意义：

- 看样本量变化后模型效果是否稳定
- 看正则化参数 `C` 对模型表现的影响

### RF OOB error 和 max_depth 验证曲线

文件：

- [rf_oob_error_vs_estimators.png](E:\QJY\outputs\figures\thesis\rf_oob_error_vs_estimators.png)
- [rf_validation_curve_max_depth.png](E:\QJY\outputs\figures\thesis\rf_validation_curve_max_depth.png)

意义：

- 看树数量增加后误差是否收敛
- 看树深太浅或太深会不会影响泛化

### XGBoost 与 LightGBM 训练过程图

文件：

- [xgboost_train_valid_metrics.png](E:\QJY\outputs\figures\thesis\xgboost_train_valid_metrics.png)
- [lightgbm_metric_vs_iterations.png](E:\QJY\outputs\figures\thesis\lightgbm_metric_vs_iterations.png)

意义：

- 看迭代轮次增加后训练集和验证集指标怎么变化
- 判断是否有过拟合风险

## 13. 这些结果最后说明了什么

从当前仓库结果看，整体可以说明：

1. 淘宝用户行为日志可以支撑流失预警研究。
2. 用户近期活跃程度、活跃天数、最近行为间隔等特征很关键。
3. baseline 模型已经有较强区分能力。
4. 优化后模型表现进一步提升。
5. 模型不只是“能预测”，还可以解释为什么某些用户更可能流失。

## 14. 如果你是第一次看，建议按这个顺序浏览

### 第一步：先看项目是什么

- [README.md](E:\QJY\README.md)

### 第二步：再看交付结构

- [DELIVERY_GUIDE.md](E:\QJY\DELIVERY_GUIDE.md)

### 第三步：看论文怎么嵌入这些结果

- [THESIS_INTEGRATION_GUIDE.md](E:\QJY\THESIS_INTEGRATION_GUIDE.md)

### 第四步：看图表索引

- [figure_table_catalog.md](E:\QJY\outputs\tables\thesis\figure_table_catalog.md)

### 第五步：看图表详细解释

- [RESULTS_INTERPRETATION.md](E:\QJY\RESULTS_INTERPRETATION.md)

## 15. 一句话总结

这个仓库不是只给出“一个模型分数”，而是完整展示了：

- 数据从哪里来
- 标签怎么定义
- 特征怎么构造
- 模型怎么训练
- 为什么这个模型有效
- 结果应该怎样放进论文并进行解释
