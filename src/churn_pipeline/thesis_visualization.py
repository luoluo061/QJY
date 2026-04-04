from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, plot_importance as lgb_plot_importance
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


RANDOM_STATE = 42


@dataclass(frozen=True)
class ThesisPaths:
    root: Path
    tables_dir: Path
    figures_dir: Path
    thesis_tables_dir: Path
    thesis_figures_dir: Path


def get_paths() -> ThesisPaths:
    root = Path(__file__).resolve().parents[2]
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    thesis_tables_dir = tables_dir / "thesis"
    thesis_figures_dir = figures_dir / "thesis"
    thesis_tables_dir.mkdir(parents=True, exist_ok=True)
    thesis_figures_dir.mkdir(parents=True, exist_ok=True)
    return ThesisPaths(root, tables_dir, figures_dir, thesis_tables_dir, thesis_figures_dir)


def load_existing_outputs(paths: ThesisPaths) -> dict[str, pd.DataFrame]:
    names = [
        "data_audit_overview.csv",
        "data_columns.csv",
        "daily_activity_distribution.csv",
        "time_window_design.csv",
        "label_distribution.csv",
        "behavior_distribution.csv",
        "user_modeling_dataset.csv",
        "feature_selection_scores.csv",
        "optimized_best_params.csv",
        "baseline_model_comparison.csv",
        "optimized_model_comparison.csv",
        "feature_importance.csv",
        "optimization_before_after_comparison.csv",
    ]
    loaded = {}
    for name in names:
        loaded[name] = pd.read_csv(paths.tables_dir / name)
    return loaded


def prepare_training_data(outputs: dict[str, pd.DataFrame]):
    data = outputs["user_modeling_dataset.csv"].copy()
    feature_cols = [c for c in data.columns if c not in {"user_id", "label_churn"}]
    X = data[feature_cols]
    y = data["label_churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    selected_features = outputs["feature_selection_scores.csv"]["feature_name"].head(20).tolist()
    return data, feature_cols, selected_features, X_train, X_test, y_train, y_test


def _parse_best_params(outputs: dict[str, pd.DataFrame], model_name: str) -> dict:
    row = outputs["optimized_best_params.csv"].loc[outputs["optimized_best_params.csv"]["model"] == model_name].iloc[0]
    return json.loads(row["best_params"])


def fit_models(outputs: dict[str, pd.DataFrame], X_train, X_test, y_train, y_test, selected_features):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    baseline_models = {
        "LR": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    optimized_models = {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "RF": RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
        "LightGBM": LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    baseline_payload = {}
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        baseline_payload[name] = {"model": model, "prob": prob, "pred": (prob >= 0.5).astype(int)}

    optimized_payload = {}
    for name, model in optimized_models.items():
        params = _parse_best_params(outputs, name)
        model.set_params(**params)
        model.fit(X_train[selected_features], y_train)
        prob = model.predict_proba(X_test[selected_features])[:, 1]
        optimized_payload[name] = {"model": model, "prob": prob, "pred": (prob >= 0.5).astype(int)}

    return baseline_payload, optimized_payload


def save_experiment_process_figures(paths: ThesisPaths, outputs: dict[str, pd.DataFrame]) -> list[dict]:
    catalog = []

    daily = outputs["daily_activity_distribution.csv"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["row_count"], color="#1f77b4", linewidth=2)
    ax.fill_between(daily["date"], daily["row_count"], alpha=0.15, color="#1f77b4")
    ax.set_title("淘宝用户行为数据时间范围分布图")
    ax.set_xlabel("日期")
    ax.set_ylabel("行为记录数")
    fig.tight_layout()
    file_name = "data_time_range.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "淘宝用户行为数据时间范围分布图", "chapter": "第4章 数据审计与实验设计", "summary": "样本时间连续覆盖31天，适合进行固定观察期与预测期的流失预警研究。"})

    window = outputs["time_window_design.csv"].copy()
    fig, ax = plt.subplots(figsize=(12, 2.8))
    y = [1, 1]
    starts = pd.to_datetime(window["start"])
    ends = pd.to_datetime(window["end"])
    colors = ["#4E79A7", "#E15759"]
    for idx, row in window.iterrows():
        start = pd.to_datetime(row["start"])
        end = pd.to_datetime(row["end"])
        ax.barh([0], [(end - start).total_seconds() / 86400 + 1], left=start, height=0.45, color=colors[idx], label=row["window_type"])
        ax.text(start + (end - start) / 2, 0, f"{row['window_type']} ({row['days']} days)", ha="center", va="center", color="white", fontsize=10)
    ax.set_yticks([])
    ax.set_title("观察期与预测期时间窗口示意图")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    file_name = "time_window_schema.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "观察期与预测期时间窗口示意图", "chapter": "第4章 数据审计与实验设计", "summary": "将前24天作为观察期、后7天作为预测期，符合先观测后预测的实验逻辑。"})

    label = outputs["label_distribution.csv"].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label["label_churn"].astype(str), label["sample_count"], color=["#59A14F", "#E15759"])
    for idx, row in label.iterrows():
        ax.text(idx, row["sample_count"], f"{row['sample_count']} ({row['sample_ratio']:.2%})", ha="center", va="bottom")
    ax.set_title("流失标签分布图")
    ax.set_xlabel("标签")
    ax.set_ylabel("样本数")
    fig.tight_layout()
    file_name = "label_distribution.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "流失标签分布图", "chapter": "第4章 标签构建", "summary": "流失样本占比约4.81%，属于明显类别不平衡问题。"})

    behavior = outputs["behavior_distribution.csv"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(behavior["behavior_type"].astype(str), behavior["count"], color="#F28E2B")
    for idx, row in behavior.iterrows():
        ax.text(idx, row["count"], f"{row['ratio']:.2%}", ha="center", va="bottom")
    ax.set_title("行为类型分布图")
    ax.set_xlabel("行为类型编码")
    ax.set_ylabel("记录数")
    fig.tight_layout()
    file_name = "behavior_type_distribution.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "行为类型分布图", "chapter": "第4章 数据概况", "summary": "行为编码1占绝对多数，说明原始行为存在明显长尾与稀疏性。"})

    cols = outputs["data_columns.csv"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cols["column_name"], cols["missing_rate"], color="#B07AA1")
    ax.set_title("字段缺失值概况图")
    ax.set_xlabel("字段")
    ax.set_ylabel("缺失率")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    file_name = "missing_value_overview.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "字段缺失值概况图", "chapter": "第4章 数据审计", "summary": "只有 user_geohash 存在大比例缺失，其余核心字段完整，可支撑稳定建模。"})

    return catalog


def save_model_result_figures(paths: ThesisPaths, baseline_payload, optimized_payload, y_test) -> tuple[list[dict], str]:
    catalog = []

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, payload in baseline_payload.items():
        fpr, tpr, _ = roc_curve(y_test, payload["prob"])
        ax.plot(fpr, tpr, linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Baseline 四模型 ROC 曲线")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    file_name = "baseline_roc_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "Baseline 四模型 ROC 曲线", "chapter": "第5章 Baseline 模型结果", "summary": "Baseline 阶段四种模型均具备较高区分能力，其中 LR 的 ROC_AUC 最优。"})

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, payload in optimized_payload.items():
        fpr, tpr, _ = roc_curve(y_test, payload["prob"])
        ax.plot(fpr, tpr, linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Optimized 四模型 ROC 曲线")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    file_name = "optimized_roc_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "Optimized 四模型 ROC 曲线", "chapter": "第5章 模型优化结果", "summary": "优化后 XGBoost 的 ROC 曲线整体包络更靠左上，综合区分能力最佳。"})

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, payload in optimized_payload.items():
        precision, recall, _ = precision_recall_curve(y_test, payload["prob"])
        ax.plot(recall, precision, linewidth=2, label=name)
    ax.set_title("Optimized 四模型 PR 曲线")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    file_name = "optimized_pr_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "Optimized 四模型 PR 曲线", "chapter": "第5章 模型优化结果", "summary": "在不平衡样本下，PR 曲线更能体现模型对流失类的识别质量。"})

    best_model_name = "XGBoost"
    best = optimized_payload[best_model_name]
    cm = confusion_matrix(y_test, best["pred"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    for matrix, suffix, title in [
        (cm, "count", "最优模型混淆矩阵（计数版）"),
        (cm_norm, "normalized", "最优模型混淆矩阵（归一化版）"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = f"{matrix[i, j]:.3f}" if suffix == "normalized" else f"{int(matrix[i, j])}"
                ax.text(j, i, value, ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["预测未流失", "预测流失"])
        ax.set_yticklabels(["真实未流失", "真实流失"])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        file_name = f"best_model_confusion_matrix_{suffix}.png"
        fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
        plt.close(fig)
        catalog.append({"type": "figure", "file_name": file_name, "title": title, "chapter": "第5章 最优模型评估", "summary": "计数版用于展示绝对分类结果，归一化版用于展示类别召回与误判结构。"})

    precision, recall, thresholds = precision_recall_curve(y_test, best["prob"])
    thresholds = np.append(thresholds, 1.0)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.plot(thresholds, f1, label="F1")
    ax.set_title("阈值-Precision/Recall/F1 曲线")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    file_name = "threshold_precision_recall_f1_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "阈值-Precision/Recall/F1 曲线", "chapter": "第5章 最优模型阈值分析", "summary": "该图可用于说明不同阈值下精确率、召回率与 F1 的权衡关系。"})

    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_test, best["prob"], n_bins=10, strategy="uniform", ax=ax)
    ax.set_title("最优模型概率校准曲线")
    fig.tight_layout()
    file_name = "best_model_calibration_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "最优模型概率校准曲线", "chapter": "第5章 最优模型评估", "summary": "用于检验模型输出概率是否与真实流失概率保持一致。"})

    return catalog, best_model_name


def save_interpretability_figures(paths: ThesisPaths, optimized_payload, best_model_name, selected_features, X_test_sel, y_test, data) -> list[dict]:
    catalog = []
    X_test_sel = X_test_sel.astype(float)

    rows = []
    for name, payload in optimized_payload.items():
        result = permutation_importance(payload["model"], X_test_sel, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
        scores = pd.DataFrame({"feature_name": selected_features, "importance": result.importances_mean}).sort_values("importance", ascending=False).head(10)
        for _, row in scores.iterrows():
            rows.append({"model": name, "feature_name": row["feature_name"], "importance": row["importance"]})
    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(paths.thesis_tables_dir / "all_models_permutation_importance_top10.csv", index=False, encoding="utf-8-sig")

    pivot = perm_df.pivot_table(index="feature_name", columns="model", values="importance", aggfunc="mean").fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(pivot))
    for model in pivot.columns:
        ax.barh(pivot.index, pivot[model], left=bottom, label=model)
        bottom += pivot[model].values
    ax.set_title("全模型 Permutation Importance 对比")
    ax.set_xlabel("Importance")
    ax.legend()
    fig.tight_layout()
    file_name = "all_models_permutation_importance.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "全模型 Permutation Importance 对比", "chapter": "第5章 模型解释", "summary": "比较不同模型对关键特征的依赖程度，可识别稳定的重要行为指标。"})

    best_model = optimized_payload[best_model_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame({"feature_name": selected_features, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
    else:
        result = permutation_importance(best_model, X_test_sel, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
        importance = pd.DataFrame({"feature_name": selected_features, "importance": result.importances_mean}).sort_values("importance", ascending=False)
    importance.to_csv(paths.thesis_tables_dir / "best_model_feature_importance.csv", index=False, encoding="utf-8-sig")

    top15 = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature_name"], top15["importance"], color="#F28E2B")
    ax.set_title("最优模型 Feature Importance 图")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    file_name = "best_model_feature_importance.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "最优模型 Feature Importance 图", "chapter": "第5章 模型解释", "summary": "展示最优模型中最能区分流失与未流失用户的特征。"})

    top5_features = importance.head(5)["feature_name"].tolist()
    for feature in top5_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn = data.loc[data["label_churn"] == 1, feature]
        retain = data.loc[data["label_churn"] == 0, feature]
        ax.boxplot([retain, churn], labels=["retain", "churn"], showfliers=False)
        ax.set_title(f"{feature} 的流失/未流失箱线图")
        ax.set_ylabel(feature)
        fig.tight_layout()
        file_name = f"boxplot_{feature}.png"
        fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
        plt.close(fig)
        catalog.append({"type": "figure", "file_name": file_name, "title": f"{feature} 的流失/未流失箱线图", "chapter": "第5章 关键特征分析", "summary": f"{feature} 在流失组与未流失组之间存在分布差异，可支撑行为解释。"})

    for feature in ["recency_days", "last_7d_actions", "active_days"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(best_model, X_test_sel, [feature], ax=ax)
        ax.set_title(f"{feature} 的 PDP 图")
        fig.tight_layout()
        file_name = f"pdp_{feature}.png"
        fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
        plt.close(fig)
        catalog.append({"type": "figure", "file_name": file_name, "title": f"{feature} 的 PDP 图", "chapter": "第5章 关键特征分析", "summary": f"PDP 展示 {feature} 对预测流失概率的边际影响趋势。"})

    return catalog


def save_training_process_figures(paths: ThesisPaths, feature_cols, selected_features, X_train, y_train, X_test, y_test) -> list[dict]:
    catalog = []

    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))])
    train_sizes, train_scores, valid_scores = learning_curve(
        lr_pipe, X_train[selected_features], y_train, cv=3, scoring="roc_auc", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train AUC")
    ax.plot(train_sizes, valid_scores.mean(axis=1), marker="s", label="Validation AUC")
    ax.set_title("LR Learning Curve")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("ROC_AUC")
    ax.legend()
    fig.tight_layout()
    file_name = "lr_learning_curve.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "LR 学习曲线", "chapter": "第5章 模型训练过程", "summary": "展示逻辑回归在训练集规模扩大时的拟合与泛化表现。"})

    c_values = np.logspace(-2, 1, 8)
    train_scores, valid_scores = validation_curve(
        lr_pipe, X_train[selected_features], y_train, param_name="clf__C", param_range=c_values, cv=3, scoring="roc_auc", n_jobs=1
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(c_values, train_scores.mean(axis=1), marker="o", label="Train AUC")
    ax.semilogx(c_values, valid_scores.mean(axis=1), marker="s", label="Validation AUC")
    ax.set_title("LR Validation Curve (C)")
    ax.set_xlabel("C")
    ax.set_ylabel("ROC_AUC")
    ax.legend()
    fig.tight_layout()
    file_name = "lr_validation_curve_c.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "LR 参数 C 验证曲线", "chapter": "第5章 模型训练过程", "summary": "用于说明正则化强度变化对 LR 泛化能力的影响。"})

    estimators = [50, 100, 150, 200, 250, 300, 400, 500]
    oob_errors = []
    for n in estimators:
        rf = RandomForestClassifier(
            n_estimators=n, oob_score=True, bootstrap=True, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1
        )
        rf.fit(X_train[selected_features], y_train)
        oob_errors.append(1 - rf.oob_score_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(estimators, oob_errors, marker="o")
    ax.set_title("RF OOB Error vs n_estimators")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("OOB Error")
    fig.tight_layout()
    file_name = "rf_oob_error_vs_estimators.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "RF OOB Error 与 n_estimators 关系图", "chapter": "第5章 模型训练过程", "summary": "用于展示随机森林树数量增加后袋外误差的收敛趋势。"})

    max_depths = [3, 5, 7, 10, 15, 20, None]
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1
    )
    train_scores, valid_scores = validation_curve(
        rf, X_train[selected_features], y_train, param_name="max_depth", param_range=max_depths, cv=3, scoring="roc_auc", n_jobs=1
    )
    depth_labels = [30 if d is None else d for d in max_depths]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(depth_labels, train_scores.mean(axis=1), marker="o", label="Train AUC")
    ax.plot(depth_labels, valid_scores.mean(axis=1), marker="s", label="Validation AUC")
    ax.set_title("RF Validation Curve (max_depth)")
    ax.set_xlabel("max_depth (30 means None)")
    ax.set_ylabel("ROC_AUC")
    ax.legend()
    fig.tight_layout()
    file_name = "rf_validation_curve_max_depth.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "RF 参数 max_depth 验证曲线", "chapter": "第5章 模型训练过程", "summary": "用于观察树深变化对随机森林拟合程度与泛化表现的影响。"})

    X_tr = X_train[selected_features]
    X_va = X_test[selected_features]
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb.fit(X_tr, y_train, eval_set=[(X_tr, y_train), (X_va, y_test)], verbose=False)
    results = xgb.evals_result()
    rounds = np.arange(1, len(results["validation_0"]["auc"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, results["validation_0"]["auc"], label="Train AUC")
    ax.plot(rounds, results["validation_1"]["auc"], label="Valid AUC")
    ax.plot(rounds, results["validation_0"]["logloss"], label="Train Logloss")
    ax.plot(rounds, results["validation_1"]["logloss"], label="Valid Logloss")
    ax.set_title("XGBoost Train/Valid AUC and Logloss")
    ax.set_xlabel("Boosting Rounds")
    ax.set_ylabel("Metric Value")
    ax.legend()
    fig.tight_layout()
    file_name = "xgboost_train_valid_metrics.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "XGBoost 训练/验证集 AUC 与 Logloss 曲线", "chapter": "第5章 模型训练过程", "summary": "可用于分析 XGBoost 在 boosting rounds 增加时是否出现过拟合。"})

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(X_tr, y_train, eval_set=[(X_tr, y_train), (X_va, y_test)], eval_metric=["auc", "binary_logloss"])
    lgbm_results = lgbm.evals_result_
    rounds = np.arange(1, len(lgbm_results["training"]["auc"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, lgbm_results["training"]["auc"], label="Train AUC")
    ax.plot(rounds, lgbm_results["valid_1"]["auc"], label="Valid AUC")
    ax.plot(rounds, lgbm_results["training"]["binary_logloss"], label="Train Logloss")
    ax.plot(rounds, lgbm_results["valid_1"]["binary_logloss"], label="Valid Logloss")
    ax.set_title("LightGBM Metric vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Metric Value")
    ax.legend()
    fig.tight_layout()
    file_name = "lightgbm_metric_vs_iterations.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "LightGBM 训练/验证指标迭代曲线", "chapter": "第5章 模型训练过程", "summary": "用于观察 LightGBM 在迭代过程中的收敛速度与泛化稳定性。"})

    fig, ax = plt.subplots(figsize=(8, 6))
    lgb_plot_importance(lgbm, ax=ax, max_num_features=15, importance_type="split")
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()
    file_name = "lightgbm_plot_importance.png"
    fig.savefig(paths.thesis_figures_dir / file_name, dpi=220)
    plt.close(fig)
    catalog.append({"type": "figure", "file_name": file_name, "title": "LightGBM Feature Importance 图", "chapter": "第5章 模型训练过程", "summary": "补充展示 LightGBM 在树模型框架下最常使用的特征。"})

    return catalog


def save_result_tables(paths: ThesisPaths, outputs: dict[str, pd.DataFrame], best_model_name: str) -> list[dict]:
    catalog = []
    outputs["baseline_model_comparison.csv"].to_csv(paths.thesis_tables_dir / "baseline_model_comparison.csv", index=False, encoding="utf-8-sig")
    outputs["optimized_model_comparison.csv"].to_csv(paths.thesis_tables_dir / "optimized_model_comparison.csv", index=False, encoding="utf-8-sig")
    outputs["optimization_before_after_comparison.csv"].to_csv(paths.thesis_tables_dir / "optimization_before_after_comparison.csv", index=False, encoding="utf-8-sig")
    catalog.extend([
        {"type": "table", "file_name": "baseline_model_comparison.csv", "title": "Baseline 模型对比表", "chapter": "第5章 Baseline 模型结果", "summary": "用于展示四种基础模型的准确率、精确率、召回率、F1 与 ROC_AUC。"},
        {"type": "table", "file_name": "optimized_model_comparison.csv", "title": "优化后模型对比表", "chapter": "第5章 模型优化结果", "summary": "用于展示不平衡处理、特征筛选与参数调优后的模型表现。"},
        {"type": "table", "file_name": "optimization_before_after_comparison.csv", "title": "优化前后结果对比表", "chapter": "第5章 模型优化结果", "summary": "用于直接说明各模型在 ROC_AUC 与 F1 上的提升幅度。"},
    ])
    return catalog


def write_catalog(paths: ThesisPaths, entries: list[dict]) -> None:
    lines = ["# Figure And Table Catalog", "", "| 类型 | 文件名 | 标题建议 | 推荐章节 | 可直接引用的结论摘要 |", "| --- | --- | --- | --- | --- |"]
    for entry in entries:
        lines.append(f"| {entry['type']} | {entry['file_name']} | {entry['title']} | {entry['chapter']} | {entry['summary']} |")
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "1. 第4章优先引用实验过程图，用于说明数据时间范围、窗口设计、标签构建合理性以及原始数据质量。",
            "2. 第5章先放模型对比表和 ROC/PR 图，再放混淆矩阵、阈值曲线和校准曲线，用于说明模型效果与业务取舍。",
            "3. 模型解释图建议放在结果分析小节，用于回答“哪些行为特征最重要、它们如何影响流失预测”。",
            "4. 训练过程图建议放在模型优化或附录，用于证明参数调整和训练轮次选择不是主观拍脑袋，而是有曲线依据。",
            "5. 如果正文篇幅有限，可把训练过程图放附录，正文保留时间窗口图、标签分布图、ROC/PR、混淆矩阵、特征重要性和 top5 箱线图。",
        ]
    )
    (paths.thesis_tables_dir / "figure_table_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = get_paths()
    outputs = load_existing_outputs(paths)
    data, feature_cols, selected_features, X_train, X_test, y_train, y_test = prepare_training_data(outputs)
    baseline_payload, optimized_payload = fit_models(outputs, X_train, X_test, y_train, y_test, selected_features)

    entries = []
    entries.extend(save_experiment_process_figures(paths, outputs))
    model_entries, best_model_name = save_model_result_figures(paths, baseline_payload, optimized_payload, y_test)
    entries.extend(model_entries)
    entries.extend(save_interpretability_figures(paths, optimized_payload, best_model_name, selected_features, X_test[selected_features], y_test, data))
    entries.extend(save_training_process_figures(paths, feature_cols, selected_features, X_train, y_train, X_test, y_test))
    entries.extend(save_result_tables(paths, outputs, best_model_name))
    write_catalog(paths, entries)
    print("Thesis visualization completed.")
    print(f"Figures: {paths.thesis_figures_dir}")
    print(f"Tables: {paths.thesis_tables_dir}")
