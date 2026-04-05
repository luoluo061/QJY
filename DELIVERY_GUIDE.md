# 仓库导航

这份文档只做一件事：帮助阅读者快速定位仓库里各类文件分别是什么，以及先看哪些内容最有效。

## 推荐查看顺序

第一次进入仓库时，建议按下面顺序浏览：

1. `README.md`
2. `src/churn_pipeline/pipeline.py`
3. `outputs/tables/stage_summary.csv`
4. `outputs/tables/optimized_model_comparison.csv`
5. `outputs/figures/optimized_roc_curve.png`
6. `outputs/figures/feature_importance.png`

如果想继续看更完整的图表分析，再进入：

- `outputs/figures/thesis/`
- `outputs/tables/thesis/`

这里的 `thesis` 是历史目录名，目前可以理解为扩展分析输出目录。

## 代码目录

### `run_pipeline.py`

主入口脚本。用于重新执行完整建模流程。

### `run_thesis_visualization.py`

扩展可视化入口。基于已有结果继续输出更多展示图。

### `src/churn_pipeline/pipeline.py`

核心流水线模块，负责：

- 数据审计
- 时间窗划分
- 标签构建
- 特征工程
- baseline 建模
- 优化建模
- 基础结果输出

### `src/churn_pipeline/thesis_visualization.py`

扩展可视化模块，负责：

- 过程图生成
- 模型结果图生成
- 模型解释图生成
- 扩展目录与结果整理

## 数据目录

### `archive/`

存放原始输入数据。

当前主数据文件为：

- `archive/.csv`

## 结果目录

### `outputs/tables/`

主要是结构化结果表，建议重点看：

- `data_audit_overview.csv`
- `time_window_design.csv`
- `label_distribution.csv`
- `feature_description_table.csv`
- `baseline_model_comparison.csv`
- `optimized_model_comparison.csv`
- `optimization_before_after_comparison.csv`
- `stage_summary.csv`

### `outputs/figures/`

主要是基础图表，建议重点看：

- `daily_activity_volume.png`
- `correlation_heatmap.png`
- `baseline_roc_curve.png`
- `optimized_roc_curve.png`
- `feature_importance.png`

### `outputs/tables/thesis/`

扩展结果表目录，适合做更细粒度的展示和对比。

### `outputs/figures/thesis/`

扩展图表目录，包含更多适合汇报与答辩展示的图。

## 如果只想快速判断项目完成度

建议直接看这几项：

1. `src/churn_pipeline/pipeline.py` 是否完整覆盖建模链路
2. `outputs/tables/stage_summary.csv` 是否给出阶段性结果
3. `outputs/tables/optimized_model_comparison.csv` 是否给出最终模型对比
4. `outputs/figures/optimized_roc_curve.png` 是否给出可视化结果
5. `outputs/figures/feature_importance.png` 是否给出解释性输出

## 这份仓库的结构逻辑

可以把仓库拆成三层理解：

1. 数据层：`archive/`
2. 实现层：`run_pipeline.py`、`run_thesis_visualization.py`、`src/`
3. 结果层：`outputs/tables/`、`outputs/figures/`

这样阅读会比较清楚：先看实现，再看结果，再回到细节表格核对。
