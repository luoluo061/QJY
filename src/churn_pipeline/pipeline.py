from __future__ import annotations

"""用户流失预测主实验流水线。

这个模块把原始电商行为日志加工为用户级建模样本，并在统一的数据口径下完成：
1. 数据审计
2. 时间窗口划分
3. 标签构建
4. 特征工程
5. baseline 模型训练
6. 特征筛选与参数优化
7. 结果与解释性输出

实现上刻意采用分阶段函数拆分，便于说明每一步实验设计的目的和产物。
"""

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
CHUNKSIZE = 1_000_000


@dataclass(frozen=True)
class Paths:
    root: Path
    data_file: Path
    output_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class WindowConfig:
    observation_start: pd.Timestamp
    observation_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp

    @property
    def observation_days(self) -> int:
        return int((self.observation_end.normalize() - self.observation_start.normalize()).days + 1)

    @property
    def prediction_days(self) -> int:
        return int((self.prediction_end.normalize() - self.prediction_start.normalize()).days + 1)


def get_paths() -> Paths:
    """集中管理输入输出路径，保证所有实验产物落到固定目录。"""
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for path in (output_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return Paths(
        root=root,
        data_file=root / "archive" / ".csv",
        output_dir=output_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )


def _update_numeric_store(store: defaultdict[int, float], series: pd.Series) -> None:
    for key, value in series.items():
        store[int(key)] += float(value)


def _update_min_store(store: dict[int, pd.Timestamp], series: pd.Series) -> None:
    for key, value in series.items():
        key = int(key)
        if key not in store or value < store[key]:
            store[key] = value


def _update_max_store(store: dict[int, pd.Timestamp], series: pd.Series) -> None:
    for key, value in series.items():
        key = int(key)
        if key not in store or value > store[key]:
            store[key] = value


def _update_set_store(store: defaultdict[int, set], grouped: pd.Series) -> None:
    for key, values in grouped.items():
        store[int(key)].update(pd.Series(values).dropna().tolist())


def audit_dataset(paths: Paths) -> tuple[pd.DataFrame, pd.Series]:
    """审计原始日志数据质量，并输出后续实验依赖的基础统计结果。"""
    null_counts = Counter()
    behavior_counts = Counter()
    daily_counts = Counter()
    hourly_counts = Counter()
    min_time = None
    max_time = None
    rows = 0
    sample_bad_times: list[str] = []
    unique_users: set[int] = set()
    unique_items: set[int] = set()
    unique_categories: set[int] = set()

    for chunk in pd.read_csv(paths.data_file, chunksize=CHUNKSIZE):
        rows += len(chunk)
        null_counts.update(chunk.isna().sum().to_dict())
        behavior_counts.update(chunk["behavior_type"].value_counts(dropna=False).to_dict())
        unique_users.update(chunk["user_id"].unique().tolist())
        unique_items.update(chunk["item_id"].unique().tolist())
        unique_categories.update(chunk["item_category"].unique().tolist())

        ts = pd.to_datetime(chunk["time"], format="%Y-%m-%d %H", errors="coerce")
        bad_times = chunk.loc[ts.isna(), "time"].astype(str)
        if len(sample_bad_times) < 10 and not bad_times.empty:
            sample_bad_times.extend(bad_times.head(10 - len(sample_bad_times)).tolist())

        valid_ts = ts.dropna()
        if not valid_ts.empty:
            current_min = valid_ts.min()
            current_max = valid_ts.max()
            min_time = current_min if min_time is None else min(min_time, current_min)
            max_time = current_max if max_time is None else max(max_time, current_max)
            daily_counts.update(valid_ts.dt.strftime("%Y-%m-%d").value_counts().to_dict())
            hourly_counts.update(valid_ts.dt.hour.value_counts().to_dict())

    audit_rows = [
        ("file_name", paths.data_file.name),
        ("row_count", rows),
        ("column_count", 6),
        ("unique_users", len(unique_users)),
        ("unique_items", len(unique_items)),
        ("unique_categories", len(unique_categories)),
        ("min_time", min_time.strftime("%Y-%m-%d %H:%M:%S")),
        ("max_time", max_time.strftime("%Y-%m-%d %H:%M:%S")),
        ("bad_time_sample_count", len(sample_bad_times)),
        ("user_geohash_missing", null_counts["user_geohash"]),
        ("user_geohash_missing_rate", null_counts["user_geohash"] / rows),
    ]
    audit_df = pd.DataFrame(audit_rows, columns=["metric", "value"])
    audit_df.to_csv(paths.tables_dir / "data_audit_overview.csv", index=False, encoding="utf-8-sig")

    column_df = pd.DataFrame(
        [
            {"column_name": "user_id", "dtype": "int64", "missing_count": null_counts["user_id"], "missing_rate": null_counts["user_id"] / rows, "description": "用户标识"},
            {"column_name": "item_id", "dtype": "int64", "missing_count": null_counts["item_id"], "missing_rate": null_counts["item_id"] / rows, "description": "商品标识"},
            {"column_name": "behavior_type", "dtype": "int64", "missing_count": null_counts["behavior_type"], "missing_rate": null_counts["behavior_type"] / rows, "description": "行为编码"},
            {"column_name": "user_geohash", "dtype": "object", "missing_count": null_counts["user_geohash"], "missing_rate": null_counts["user_geohash"] / rows, "description": "地理位置编码"},
            {"column_name": "item_category", "dtype": "int64", "missing_count": null_counts["item_category"], "missing_rate": null_counts["item_category"] / rows, "description": "商品类目"},
            {"column_name": "time", "dtype": "object", "missing_count": null_counts["time"], "missing_rate": null_counts["time"] / rows, "description": "小时级时间戳字符串"},
        ]
    )
    column_df.to_csv(paths.tables_dir / "data_columns.csv", index=False, encoding="utf-8-sig")

    behavior_df = pd.DataFrame(
        [{"behavior_type": int(k), "count": int(v), "ratio": float(v / rows)} for k, v in sorted(behavior_counts.items())]
    )
    behavior_df.to_csv(paths.tables_dir / "behavior_distribution.csv", index=False, encoding="utf-8-sig")

    daily_df = pd.Series(dict(sorted(daily_counts.items())), name="row_count").rename_axis("date").reset_index()
    daily_df.to_csv(paths.tables_dir / "daily_activity_distribution.csv", index=False, encoding="utf-8-sig")

    hourly_df = pd.Series(dict(sorted(hourly_counts.items())), name="row_count").rename_axis("hour").reset_index()
    hourly_df.to_csv(paths.tables_dir / "hourly_activity_distribution.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(daily_df["date"]), daily_df["row_count"], marker="o", linewidth=1.8)
    ax.set_title("Daily Activity Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rows")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "daily_activity_volume.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly_df["hour"], hourly_df["row_count"], color="#4E79A7")
    ax.set_title("Hourly Activity Distribution")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "hourly_activity_distribution.png", dpi=200)
    plt.close(fig)

    return audit_df, pd.Series(dict(sorted(daily_counts.items())))


def choose_windows(daily_series: pd.Series, paths: Paths) -> WindowConfig:
    """根据样本时间范围定义观察期和预测期。"""
    all_dates = pd.to_datetime(daily_series.index)
    start_date = all_dates.min()
    end_date = all_dates.max()
    prediction_days = 7
    # 固定把最后 7 天留作预测期，确保标签来自未来时间段。
    prediction_start = end_date - pd.Timedelta(days=prediction_days - 1)
    observation_end = prediction_start - pd.Timedelta(hours=1)
    observation_start = start_date
    window = WindowConfig(
        observation_start=pd.Timestamp(observation_start),
        observation_end=pd.Timestamp(observation_end),
        prediction_start=pd.Timestamp(prediction_start),
        prediction_end=pd.Timestamp(end_date + pd.Timedelta(hours=23)),
    )
    window_df = pd.DataFrame(
        [
            {
                "window_type": "observation",
                "start": window.observation_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": window.observation_end.strftime("%Y-%m-%d %H:%M:%S"),
                "days": window.observation_days,
                "design_rationale": "使用前24天构造用户画像，覆盖全部历史行为信息。",
            },
            {
                "window_type": "prediction",
                "start": window.prediction_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": window.prediction_end.strftime("%Y-%m-%d %H:%M:%S"),
                "days": window.prediction_days,
                "design_rationale": "使用最后7天判断用户是否仍然活跃，流失定义为预测期无任何行为。",
            },
        ]
    )
    window_df.to_csv(paths.tables_dir / "time_window_design.csv", index=False, encoding="utf-8-sig")
    return window


def build_modeling_dataset(paths: Paths, window: WindowConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将行为流水聚合为用户级建模宽表，并同步生成流失标签。"""
    total_actions = defaultdict(float)
    weekend_actions = defaultdict(float)
    geohash_missing_actions = defaultdict(float)
    last_7d_actions = defaultdict(float)
    last_3d_actions = defaultdict(float)
    behavior_action_counts: dict[int, defaultdict[int, float]] = {code: defaultdict(float) for code in [1, 2, 3, 4]}
    first_timestamp: dict[int, pd.Timestamp] = {}
    last_timestamp: dict[int, pd.Timestamp] = {}
    unique_items = defaultdict(set)
    unique_categories = defaultdict(set)
    active_days = defaultdict(set)
    active_hours = defaultdict(set)
    prediction_users: set[int] = set()

    last_7d_start = window.observation_end - pd.Timedelta(days=6)
    last_3d_start = window.observation_end - pd.Timedelta(days=2)

    for chunk in pd.read_csv(paths.data_file, chunksize=CHUNKSIZE):
        chunk["timestamp"] = pd.to_datetime(chunk["time"], format="%Y-%m-%d %H", errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"]).copy()
        # 观察期只负责生成特征，预测期只负责确定标签，避免时间泄漏。
        obs_mask = (chunk["timestamp"] >= window.observation_start) & (chunk["timestamp"] <= window.observation_end)
        pred_mask = (chunk["timestamp"] >= window.prediction_start) & (chunk["timestamp"] <= window.prediction_end)

        prediction_users.update(chunk.loc[pred_mask, "user_id"].astype(int).unique().tolist())
        obs = chunk.loc[obs_mask].copy()
        if obs.empty:
            continue

        obs["date"] = obs["timestamp"].dt.date
        obs["hour"] = obs["timestamp"].dt.hour
        obs["is_weekend"] = (obs["timestamp"].dt.weekday >= 5).astype(int)
        obs["geohash_missing"] = obs["user_geohash"].isna().astype(int)

        # 这里按用户持续累计统计量，把行为流水压缩成用户级画像。
        _update_numeric_store(total_actions, obs.groupby("user_id").size())
        _update_numeric_store(weekend_actions, obs.groupby("user_id")["is_weekend"].sum())
        _update_numeric_store(geohash_missing_actions, obs.groupby("user_id")["geohash_missing"].sum())
        _update_min_store(first_timestamp, obs.groupby("user_id")["timestamp"].min())
        _update_max_store(last_timestamp, obs.groupby("user_id")["timestamp"].max())
        _update_set_store(unique_items, obs.groupby("user_id")["item_id"].unique())
        _update_set_store(unique_categories, obs.groupby("user_id")["item_category"].unique())
        _update_set_store(active_days, obs.groupby("user_id")["date"].unique())
        _update_set_store(active_hours, obs.groupby("user_id")["hour"].unique())

        for code in [1, 2, 3, 4]:
            code_rows = obs.loc[obs["behavior_type"] == code]
            if not code_rows.empty:
                _update_numeric_store(behavior_action_counts[code], code_rows.groupby("user_id").size())

        recent_7d = obs.loc[obs["timestamp"] >= last_7d_start]
        if not recent_7d.empty:
            _update_numeric_store(last_7d_actions, recent_7d.groupby("user_id").size())

        recent_3d = obs.loc[obs["timestamp"] >= last_3d_start]
        if not recent_3d.empty:
            _update_numeric_store(last_3d_actions, recent_3d.groupby("user_id").size())

    users = sorted(total_actions.keys())
    data = pd.DataFrame({"user_id": users})
    data["total_actions"] = data["user_id"].map(total_actions).astype(float)
    data["weekend_actions"] = data["user_id"].map(weekend_actions).fillna(0.0)
    data["geohash_missing_actions"] = data["user_id"].map(geohash_missing_actions).fillna(0.0)
    data["last_7d_actions"] = data["user_id"].map(last_7d_actions).fillna(0.0)
    data["last_3d_actions"] = data["user_id"].map(last_3d_actions).fillna(0.0)
    data["unique_items"] = data["user_id"].map(lambda x: len(unique_items[int(x)])).astype(float)
    data["unique_categories"] = data["user_id"].map(lambda x: len(unique_categories[int(x)])).astype(float)
    data["active_days"] = data["user_id"].map(lambda x: len(active_days[int(x)])).astype(float)
    data["active_hours"] = data["user_id"].map(lambda x: len(active_hours[int(x)])).astype(float)
    data["first_timestamp"] = data["user_id"].map(first_timestamp)
    data["last_timestamp"] = data["user_id"].map(last_timestamp)
    data["active_span_days"] = (data["last_timestamp"].dt.normalize() - data["first_timestamp"].dt.normalize()).dt.days + 1
    data["recency_days"] = (window.observation_end.normalize() - data["last_timestamp"].dt.normalize()).dt.days
    data["avg_actions_per_active_day"] = data["total_actions"] / data["active_days"].replace(0, np.nan)
    data["avg_actions_per_item"] = data["total_actions"] / data["unique_items"].replace(0, np.nan)
    data["avg_actions_per_category"] = data["total_actions"] / data["unique_categories"].replace(0, np.nan)
    data["weekend_action_ratio"] = data["weekend_actions"] / data["total_actions"].replace(0, np.nan)
    data["geohash_missing_ratio"] = data["geohash_missing_actions"] / data["total_actions"].replace(0, np.nan)
    data["last_7d_ratio"] = data["last_7d_actions"] / data["total_actions"].replace(0, np.nan)
    data["last_3d_ratio"] = data["last_3d_actions"] / data["total_actions"].replace(0, np.nan)
    data["category_item_ratio"] = data["unique_categories"] / data["unique_items"].replace(0, np.nan)
    data["active_day_ratio"] = data["active_days"] / window.observation_days
    data["active_hour_ratio"] = data["active_hours"] / 24.0

    # 行为类型次数和占比用于描述用户行为结构，而不只是行为总量。
    for code in [1, 2, 3, 4]:
        count_col = f"behavior_{code}_count"
        ratio_col = f"behavior_{code}_ratio"
        data[count_col] = data["user_id"].map(behavior_action_counts[code]).fillna(0.0)
        data[ratio_col] = data[count_col] / data["total_actions"].replace(0, np.nan)

    # 观察期出现过、但预测期完全没有行为的用户记为流失。
    data["label_churn"] = (~data["user_id"].isin(prediction_users)).astype(int)
    data = data.drop(columns=["first_timestamp", "last_timestamp"]).fillna(0.0)
    data.to_csv(paths.tables_dir / "user_modeling_dataset.csv", index=False, encoding="utf-8-sig")

    label_df = pd.DataFrame(
        [
            {"label_churn": 0, "sample_count": int((data["label_churn"] == 0).sum()), "sample_ratio": float((data["label_churn"] == 0).mean()), "label_meaning": "预测期仍有行为，判定为未流失"},
            {"label_churn": 1, "sample_count": int((data["label_churn"] == 1).sum()), "sample_ratio": float((data["label_churn"] == 1).mean()), "label_meaning": "预测期无任何行为，判定为流失"},
        ]
    )
    label_df.to_csv(paths.tables_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")
    return data, label_df


def save_feature_tables(paths: Paths, data: pd.DataFrame) -> pd.DataFrame:
    """导出特征说明、描述统计和相关性结果，便于解释建模变量。"""
    feature_descriptions = [
        ("total_actions", "观察期内总行为次数"),
        ("weekend_actions", "观察期内周末行为次数"),
        ("geohash_missing_actions", "观察期内地理编码缺失的行为次数"),
        ("last_7d_actions", "观察期最后7天行为次数"),
        ("last_3d_actions", "观察期最后3天行为次数"),
        ("unique_items", "观察期内交互商品去重数"),
        ("unique_categories", "观察期内交互类目去重数"),
        ("active_days", "观察期内发生行为的天数"),
        ("active_hours", "观察期内发生行为的小时去重数"),
        ("active_span_days", "首次行为到最后一次行为跨度天数"),
        ("recency_days", "距离观察期结束最近一次行为的间隔天数"),
        ("avg_actions_per_active_day", "平均每活跃日行为次数"),
        ("avg_actions_per_item", "平均每个商品的行为次数"),
        ("avg_actions_per_category", "平均每个类目的行为次数"),
        ("weekend_action_ratio", "周末行为占比"),
        ("geohash_missing_ratio", "地理编码缺失行为占比"),
        ("last_7d_ratio", "最近7天行为占比"),
        ("last_3d_ratio", "最近3天行为占比"),
        ("category_item_ratio", "类目数与商品数之比"),
        ("active_day_ratio", "活跃天数占观察期比例"),
        ("active_hour_ratio", "活跃小时覆盖率"),
        ("behavior_1_count", "行为编码1出现次数"),
        ("behavior_1_ratio", "行为编码1占比"),
        ("behavior_2_count", "行为编码2出现次数"),
        ("behavior_2_ratio", "行为编码2占比"),
        ("behavior_3_count", "行为编码3出现次数"),
        ("behavior_3_ratio", "行为编码3占比"),
        ("behavior_4_count", "行为编码4出现次数"),
        ("behavior_4_ratio", "行为编码4占比"),
    ]
    feature_description_df = pd.DataFrame(feature_descriptions, columns=["feature_name", "description"])
    feature_description_df.to_csv(paths.tables_dir / "feature_description_table.csv", index=False, encoding="utf-8-sig")

    feature_cols = [col for col in data.columns if col not in {"user_id", "label_churn"}]
    stats_df = data[feature_cols].describe().T.reset_index().rename(columns={"index": "feature_name"})
    stats_df.to_csv(paths.tables_dir / "feature_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    corr = data[feature_cols + ["label_churn"]].corr(numeric_only=True)["label_churn"].drop("label_churn").abs().sort_values(ascending=False)
    top_features = corr.head(12).index.tolist()
    corr_matrix = data[top_features + ["label_churn"]].corr(numeric_only=True)
    corr_matrix.to_csv(paths.tables_dir / "feature_correlation_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)
    return feature_description_df


def _evaluate_predictions(model_name: str, stage: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """统一封装分类指标，避免不同阶段结果口径不一致。"""
    return {
        "stage": stage,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def _plot_roc_curve(model_probs: dict[str, np.ndarray], y_true: pd.Series, figure_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, figure_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def train_baseline_models(paths: Paths, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """训练 baseline 模型，建立统一的比较基线。"""
    feature_cols = [col for col in data.columns if col not in {"user_id", "label_churn"}]
    X = data[feature_cols]
    y = data["label_churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
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

    # baseline 阶段尽量保持模型配置简洁，用来观察原始特征本身的可分性。
    results = []
    prediction_store = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append(_evaluate_predictions(model_name, "baseline", y_test, y_pred, y_prob))
        prediction_store[model_name] = {"model": model, "prob": y_prob, "pred": y_pred}

    result_df = pd.DataFrame(results).sort_values(["roc_auc", "f1"], ascending=False)
    result_df.to_csv(paths.tables_dir / "baseline_model_comparison.csv", index=False, encoding="utf-8-sig")
    _plot_roc_curve(
        {name: payload["prob"] for name, payload in prediction_store.items()},
        y_test,
        paths.figures_dir / "baseline_roc_curve.png",
        "Baseline ROC Curve",
    )

    best_model_name = result_df.iloc[0]["model"]
    _plot_confusion_matrix(
        y_test,
        prediction_store[best_model_name]["pred"],
        paths.figures_dir / "baseline_confusion_matrix.png",
        f"Baseline Confusion Matrix - {best_model_name}",
    )

    split_info = {
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prediction_store": prediction_store,
    }
    return result_df, split_info


def optimize_models(paths: Paths, split_info: dict, baseline_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str]]:
    """在统一数据划分上完成特征筛选、类别不平衡处理和参数搜索。"""
    X_train = split_info["X_train"]
    X_test = split_info["X_test"]
    y_train = split_info["y_train"]
    y_test = split_info["y_test"]

    # 先用互信息做一次过滤，减少噪声特征进入后续搜索空间。
    feature_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    selected_scores = (
        pd.DataFrame({"feature_name": X_train.columns, "mutual_info_score": feature_scores})
        .sort_values("mutual_info_score", ascending=False)
        .reset_index(drop=True)
    )
    top_k = min(20, len(selected_scores))
    selected_features = selected_scores.head(top_k)["feature_name"].tolist()
    selected_scores.to_csv(paths.tables_dir / "feature_selection_scores.csv", index=False, encoding="utf-8-sig")

    # 只用筛选后的特征进入优化阶段，控制搜索复杂度并减少噪声干扰。
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    # 正负样本比例用于提升树模型的类别不平衡修正。
    scale_pos_weight = neg / max(pos, 1)

    search_space = {
        "LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__C": np.logspace(-2, 1, 8),
                "clf__solver": ["lbfgs", "liblinear"],
            },
        ),
        "RF": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [6, 10, 16, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.6, 0.8],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            {
                "n_estimators": [200, 300, 500],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_depth": [-1, 6, 10, 16],
                "min_child_samples": [10, 20, 40, 60],
            },
        ),
    }

    optimized_results = []
    optimized_probs = {}
    optimized_models = {}
    best_params_rows = []

    for model_name, (estimator, params) in search_space.items():
        # 统一以 ROC_AUC 做搜索目标，保证不同模型在同一标准下比较。
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=8,
            scoring="roc_auc",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train_sel, y_train)
        best_model = search.best_estimator_
        y_prob = best_model.predict_proba(X_test_sel)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        optimized_results.append(_evaluate_predictions(model_name, "optimized", y_test, y_pred, y_prob))
        optimized_probs[model_name] = {"prob": y_prob, "pred": y_pred}
        optimized_models[model_name] = best_model
        best_params_rows.append(
            {
                "model": model_name,
                "best_cv_score_roc_auc": search.best_score_,
                "selected_feature_count": len(selected_features),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )

    optimized_df = pd.DataFrame(optimized_results).sort_values(["roc_auc", "f1"], ascending=False)
    optimized_df.to_csv(paths.tables_dir / "optimized_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_params_rows).to_csv(paths.tables_dir / "optimized_best_params.csv", index=False, encoding="utf-8-sig")

    before_after = baseline_df.merge(
        optimized_df,
        on="model",
        suffixes=("_baseline", "_optimized"),
    )
    before_after["roc_auc_gain"] = before_after["roc_auc_optimized"] - before_after["roc_auc_baseline"]
    before_after["f1_gain"] = before_after["f1_optimized"] - before_after["f1_baseline"]
    before_after.to_csv(paths.tables_dir / "optimization_before_after_comparison.csv", index=False, encoding="utf-8-sig")

    _plot_roc_curve(
        {name: payload["prob"] for name, payload in optimized_probs.items()},
        y_test,
        paths.figures_dir / "optimized_roc_curve.png",
        "Optimized ROC Curve",
    )

    best_optimized_name = optimized_df.iloc[0]["model"]
    _plot_confusion_matrix(
        y_test,
        optimized_probs[best_optimized_name]["pred"],
        paths.figures_dir / "optimized_confusion_matrix.png",
        f"Optimized Confusion Matrix - {best_optimized_name}",
    )

    return optimized_df, before_after, optimized_models[best_optimized_name], selected_features


def save_feature_importance(paths: Paths, best_model: object, selected_features: list[str], split_info: dict) -> None:
    """输出最优模型的特征重要性结果，用于解释模型判断依据。"""
    X_test = split_info["X_test"][selected_features]
    y_test = split_info["y_test"]

    if hasattr(best_model, "feature_importances_"):
        importance_values = np.asarray(best_model.feature_importances_)
    else:
        # 对没有原生特征重要性的模型退化为 permutation importance。
        result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
        importance_values = result.importances_mean

    importance_df = (
        pd.DataFrame({"feature_name": selected_features, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(paths.tables_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top_df = importance_df.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_df["feature_name"], top_df["importance"], color="#F28E2B")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "feature_importance.png", dpi=200)
    plt.close(fig)


def save_stage_summary(paths: Paths, audit_df: pd.DataFrame, label_df: pd.DataFrame, baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> None:
    """把关键阶段结果压缩成一页摘要，方便快速汇报项目进展。"""
    baseline_best = baseline_df.iloc[0]
    optimized_best = optimized_df.iloc[0]
    summary = pd.DataFrame(
        [
            {"stage": "data_audit", "summary": f"完成真实CSV审计，确认共{int(float(audit_df.loc[audit_df['metric']=='row_count', 'value'].iloc[0])):,}条记录。"},
            {"stage": "label_building", "summary": f"观察期用户共{label_df['sample_count'].sum():,}人，流失占比为{label_df.loc[label_df['label_churn']==1, 'sample_ratio'].iloc[0]:.4f}。"},
            {"stage": "baseline_modeling", "summary": f"Baseline最优模型为{baseline_best['model']}，ROC_AUC={baseline_best['roc_auc']:.4f}，F1={baseline_best['f1']:.4f}。"},
            {"stage": "model_optimization", "summary": f"优化后最优模型为{optimized_best['model']}，ROC_AUC={optimized_best['roc_auc']:.4f}，F1={optimized_best['f1']:.4f}。"},
        ]
    )
    summary.to_csv(paths.tables_dir / "stage_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    paths = get_paths()
    # 主流程顺序严格对应建模链路：先审计，再构造样本，再训练与解释。
    audit_df, daily_series = audit_dataset(paths)
    window = choose_windows(daily_series, paths)
    modeling_df, label_df = build_modeling_dataset(paths, window)
    save_feature_tables(paths, modeling_df)
    baseline_df, split_info = train_baseline_models(paths, modeling_df)
    optimized_df, _, best_model, selected_features = optimize_models(paths, split_info, baseline_df)
    save_feature_importance(paths, best_model, selected_features, split_info)
    save_stage_summary(paths, audit_df, label_df, baseline_df, optimized_df)

    print("Pipeline completed.")
    print(f"Data file: {paths.data_file}")
    print(f"Outputs: {paths.output_dir}")
