# Figure And Table Catalog

| 类型 | 文件名 | 标题建议 | 推荐章节 | 可直接引用的结论摘要 |
| --- | --- | --- | --- | --- |
| figure | data_time_range.png | 淘宝用户行为数据时间范围分布图 | 第4章 数据审计与实验设计 | 样本时间连续覆盖31天，适合进行固定观察期与预测期的流失预警研究。 |
| figure | time_window_schema.png | 观察期与预测期时间窗口示意图 | 第4章 数据审计与实验设计 | 将前24天作为观察期、后7天作为预测期，符合先观测后预测的实验逻辑。 |
| figure | label_distribution.png | 流失标签分布图 | 第4章 标签构建 | 流失样本占比约4.81%，属于明显类别不平衡问题。 |
| figure | behavior_type_distribution.png | 行为类型分布图 | 第4章 数据概况 | 行为编码1占绝对多数，说明原始行为存在明显长尾与稀疏性。 |
| figure | missing_value_overview.png | 字段缺失值概况图 | 第4章 数据审计 | 只有 user_geohash 存在大比例缺失，其余核心字段完整，可支撑稳定建模。 |
| figure | baseline_roc_curve.png | Baseline 四模型 ROC 曲线 | 第5章 Baseline 模型结果 | Baseline 阶段四种模型均具备较高区分能力，其中 LR 的 ROC_AUC 最优。 |
| figure | optimized_roc_curve.png | Optimized 四模型 ROC 曲线 | 第5章 模型优化结果 | 优化后 XGBoost 的 ROC 曲线整体包络更靠左上，综合区分能力最佳。 |
| figure | optimized_pr_curve.png | Optimized 四模型 PR 曲线 | 第5章 模型优化结果 | 在不平衡样本下，PR 曲线更能体现模型对流失类的识别质量。 |
| figure | best_model_confusion_matrix_count.png | 最优模型混淆矩阵（计数版） | 第5章 最优模型评估 | 计数版用于展示绝对分类结果，归一化版用于展示类别召回与误判结构。 |
| figure | best_model_confusion_matrix_normalized.png | 最优模型混淆矩阵（归一化版） | 第5章 最优模型评估 | 计数版用于展示绝对分类结果，归一化版用于展示类别召回与误判结构。 |
| figure | threshold_precision_recall_f1_curve.png | 阈值-Precision/Recall/F1 曲线 | 第5章 最优模型阈值分析 | 该图可用于说明不同阈值下精确率、召回率与 F1 的权衡关系。 |
| figure | best_model_calibration_curve.png | 最优模型概率校准曲线 | 第5章 最优模型评估 | 用于检验模型输出概率是否与真实流失概率保持一致。 |
| figure | all_models_permutation_importance.png | 全模型 Permutation Importance 对比 | 第5章 模型解释 | 比较不同模型对关键特征的依赖程度，可识别稳定的重要行为指标。 |
| figure | best_model_feature_importance.png | 最优模型 Feature Importance 图 | 第5章 模型解释 | 展示最优模型中最能区分流失与未流失用户的特征。 |
| figure | boxplot_active_day_ratio.png | active_day_ratio 的流失/未流失箱线图 | 第5章 关键特征分析 | active_day_ratio 在流失组与未流失组之间存在分布差异，可支撑行为解释。 |
| figure | boxplot_active_days.png | active_days 的流失/未流失箱线图 | 第5章 关键特征分析 | active_days 在流失组与未流失组之间存在分布差异，可支撑行为解释。 |
| figure | boxplot_last_3d_actions.png | last_3d_actions 的流失/未流失箱线图 | 第5章 关键特征分析 | last_3d_actions 在流失组与未流失组之间存在分布差异，可支撑行为解释。 |
| figure | boxplot_last_7d_actions.png | last_7d_actions 的流失/未流失箱线图 | 第5章 关键特征分析 | last_7d_actions 在流失组与未流失组之间存在分布差异，可支撑行为解释。 |
| figure | boxplot_recency_days.png | recency_days 的流失/未流失箱线图 | 第5章 关键特征分析 | recency_days 在流失组与未流失组之间存在分布差异，可支撑行为解释。 |
| figure | pdp_recency_days.png | recency_days 的 PDP 图 | 第5章 关键特征分析 | PDP 展示 recency_days 对预测流失概率的边际影响趋势。 |
| figure | pdp_last_7d_actions.png | last_7d_actions 的 PDP 图 | 第5章 关键特征分析 | PDP 展示 last_7d_actions 对预测流失概率的边际影响趋势。 |
| figure | pdp_active_days.png | active_days 的 PDP 图 | 第5章 关键特征分析 | PDP 展示 active_days 对预测流失概率的边际影响趋势。 |
| figure | lr_learning_curve.png | LR 学习曲线 | 第5章 模型训练过程 | 展示逻辑回归在训练集规模扩大时的拟合与泛化表现。 |
| figure | lr_validation_curve_c.png | LR 参数 C 验证曲线 | 第5章 模型训练过程 | 用于说明正则化强度变化对 LR 泛化能力的影响。 |
| figure | rf_oob_error_vs_estimators.png | RF OOB Error 与 n_estimators 关系图 | 第5章 模型训练过程 | 用于展示随机森林树数量增加后袋外误差的收敛趋势。 |
| figure | rf_validation_curve_max_depth.png | RF 参数 max_depth 验证曲线 | 第5章 模型训练过程 | 用于观察树深变化对随机森林拟合程度与泛化表现的影响。 |
| figure | xgboost_train_valid_metrics.png | XGBoost 训练/验证集 AUC 与 Logloss 曲线 | 第5章 模型训练过程 | 可用于分析 XGBoost 在 boosting rounds 增加时是否出现过拟合。 |
| figure | lightgbm_metric_vs_iterations.png | LightGBM 训练/验证指标迭代曲线 | 第5章 模型训练过程 | 用于观察 LightGBM 在迭代过程中的收敛速度与泛化稳定性。 |
| figure | lightgbm_plot_importance.png | LightGBM Feature Importance 图 | 第5章 模型训练过程 | 补充展示 LightGBM 在树模型框架下最常使用的特征。 |
| table | baseline_model_comparison.csv | Baseline 模型对比表 | 第5章 Baseline 模型结果 | 用于展示四种基础模型的准确率、精确率、召回率、F1 与 ROC_AUC。 |
| table | optimized_model_comparison.csv | 优化后模型对比表 | 第5章 模型优化结果 | 用于展示不平衡处理、特征筛选与参数调优后的模型表现。 |
| table | optimization_before_after_comparison.csv | 优化前后结果对比表 | 第5章 模型优化结果 | 用于直接说明各模型在 ROC_AUC 与 F1 上的提升幅度。 |

## 使用说明

1. 第4章优先引用实验过程图，用于说明数据时间范围、窗口设计、标签构建合理性以及原始数据质量。
2. 第5章先放模型对比表和 ROC/PR 图，再放混淆矩阵、阈值曲线和校准曲线，用于说明模型效果与业务取舍。
3. 模型解释图建议放在结果分析小节，用于回答“哪些行为特征最重要、它们如何影响流失预测”。
4. 训练过程图建议放在模型优化或附录，用于证明参数调整和训练轮次选择不是主观拍脑袋，而是有曲线依据。
5. 如果正文篇幅有限，可把训练过程图放附录，正文保留时间窗口图、标签分布图、ROC/PR、混淆矩阵、特征重要性和 top5 箱线图。