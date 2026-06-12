from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Converted"
ID_COL = "User_ID"
THRESHOLDS = np.linspace(0.05, 0.95, 181)


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "Week2-hackathon-datasetsacd318d"
    results_dir = root / "results"
    return {
        "root": root,
        "data": data_dir,
        "results": results_dir,
        "submission": root / "submission.csv",
        "best_config": results_dir / "best_config.json",
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = project_paths()
    data_dir = paths["data"]
    train = pd.read_csv(data_dir / "train.csv")
    public = pd.read_csv(data_dir / "public_test.csv")
    private = pd.read_csv(data_dir / "private_test.csv")
    return train, public, private


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.drop(columns=[ID_COL], errors="ignore")

    for col in ["Age", "Income", "Time_On_Site"]:
        x[f"{col}_missing"] = x[col].isna().astype("int8")

    time_clip = x["Time_On_Site"].clip(lower=0, upper=60)
    pages = x["Pages_Viewed"].replace(0, np.nan)
    products = x["Products_Viewed"].replace(0, np.nan)

    x["Time_On_Site_clip"] = time_clip
    x["log_Time_On_Site"] = np.log1p(time_clip)
    x["log_Income"] = np.log1p(x["Income"].clip(lower=0))
    x["views_total"] = x["Pages_Viewed"] + x["Products_Viewed"]
    x["products_per_page"] = x["Products_Viewed"] / pages
    x["time_per_page"] = time_clip / pages
    x["time_per_product"] = time_clip / products
    x["purchase_x_discount"] = x["Previous_Purchases"] * x["Discount_Seen"]
    x["engagement"] = x["views_total"] * np.log1p(time_clip)

    for col in ["City_Tier", "Discount_Seen", "Browser_Version", "Campaign_Code"]:
        if col in x.columns:
            x[col] = x[col].astype("object")

    return x


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return add_features(df.drop(columns=[TARGET])), df[TARGET].astype(int)


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "min_frequency": 5}
    try:
        return OneHotEncoder(**kwargs, sparse_output=False)
    except TypeError:
        return OneHotEncoder(**kwargs, sparse=False)


def make_preprocessor(x: pd.DataFrame, scaled: bool) -> ColumnTransformer:
    cat_cols = [col for col in x.columns if x[col].dtype == "object"]
    num_cols = [col for col in x.columns if col not in cat_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


class ProbabilityEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: Iterable[tuple[str, BaseEstimator]]):
        self.estimators = list(estimators)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ProbabilityEnsemble":
        self.fitted_estimators_ = []
        self.classes_ = np.array([0, 1])
        for name, estimator in self.estimators:
            model = copy.deepcopy(estimator)
            model.fit(x, y)
            self.fitted_estimators_.append((name, model))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = np.column_stack(
            [model.predict_proba(x)[:, 1] for _, model in self.fitted_estimators_]
        )
        positive = probabilities.mean(axis=1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(columns=self.columns, errors="ignore")


@dataclass
class ModelResult:
    name: str
    f1: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    positive_predictions: int
    model: BaseEstimator
    probabilities: np.ndarray
    predictions: np.ndarray

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.name,
            "public_f1": round(self.f1, 6),
            "threshold": round(self.threshold, 6),
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "positive_predictions": self.positive_predictions,
        }


def make_model_specs(x_train: pd.DataFrame) -> dict[str, BaseEstimator]:
    pre_scaled = make_preprocessor(x_train, scaled=True)
    pre_tree = make_preprocessor(x_train, scaled=False)
    x_no_campaign = x_train.drop(columns=["Campaign_Code"], errors="ignore")
    pre_scaled_no_campaign = make_preprocessor(x_no_campaign, scaled=True)
    pre_tree_no_campaign = make_preprocessor(x_no_campaign, scaled=False)

    specs: dict[str, BaseEstimator] = {
        "logreg_balanced": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=1.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_c0_25": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=0.25,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_c0_5": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=0.5,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_c2": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=2.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_c4": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=4.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_no_campaign": Pipeline(
            [
                ("drop_campaign", ColumnDropper(["Campaign_Code"])),
                ("preprocess", copy.deepcopy(pre_scaled_no_campaign)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=1.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_balanced": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_tree)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        l2_regularization=0.15,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_no_campaign": Pipeline(
            [
                ("drop_campaign", ColumnDropper(["Campaign_Code"])),
                ("preprocess", copy.deepcopy(pre_tree_no_campaign)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=450,
                        learning_rate=0.025,
                        max_leaf_nodes=12,
                        l2_regularization=0.2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_tree)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        l2_regularization=0.15,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_tree)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_features="sqrt",
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocess", copy.deepcopy(pre_tree)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    if importlib.util.find_spec("catboost") is not None:
        from catboost import CatBoostClassifier

        cat_cols = [
            idx
            for idx, col in enumerate(x_train.columns)
            if x_train[col].dtype == "object"
        ]
        specs["catboost"] = CatBoostClassifier(
            iterations=900,
            learning_rate=0.035,
            depth=5,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_cols,
        )

    ensemble_members = [
        ("logreg_balanced", specs["logreg_balanced"]),
        ("hist_gradient_boosting_balanced", specs["hist_gradient_boosting_balanced"]),
        ("extra_trees_balanced", specs["extra_trees_balanced"]),
        ("random_forest_balanced", specs["random_forest_balanced"]),
    ]
    specs["ensemble_balanced_mean"] = ProbabilityEnsemble(ensemble_members)

    return specs


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLDS]
    best_idx = int(np.argmax(scores))
    return float(THRESHOLDS[best_idx]), float(scores[best_idx])


def evaluate_model(
    name: str, model: BaseEstimator, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series
) -> ModelResult:
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]
    threshold, best_f1 = tune_threshold(y_val, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return ModelResult(
        name=name,
        f1=best_f1,
        threshold=threshold,
        accuracy=accuracy_score(y_val, predictions),
        precision=precision_score(y_val, predictions, zero_division=0),
        recall=recall_score(y_val, predictions, zero_division=0),
        positive_predictions=int(predictions.sum()),
        model=model,
        probabilities=probabilities,
        predictions=predictions,
    )


def run_all_experiments() -> tuple[list[ModelResult], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train, public, _ = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(public)
    specs = make_model_specs(x_train)

    results: list[ModelResult] = []
    for name, model in specs.items():
        print(f"Training {name}...")
        result = evaluate_model(name, model, x_train, y_train, x_val, y_val)
        print(
            f"  F1={result.f1:.5f} threshold={result.threshold:.3f} "
            f"precision={result.precision:.5f} recall={result.recall:.5f}"
        )
        results.append(result)

    results.sort(key=lambda item: item.f1, reverse=True)
    return results, x_train, y_train, x_val, y_val


def save_best_config(result: ModelResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.name,
        "threshold": result.threshold,
        "public_f1": result.f1,
        "public_precision": result.precision,
        "public_recall": result.recall,
        "public_accuracy": result.accuracy,
        "positive_predictions_public": result.positive_predictions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_best_config() -> dict[str, object]:
    config_path = project_paths()["best_config"]
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_submission(submission: pd.DataFrame, private: pd.DataFrame) -> None:
    expected_columns = [ID_COL, TARGET]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}, got {list(submission.columns)}")
    if len(submission) != len(private):
        raise ValueError(f"submission must have {len(private)} rows, got {len(submission)}")
    if not submission[ID_COL].equals(private[ID_COL]):
        raise ValueError("submission User_ID order does not match private_test.csv")
    unique_values = set(submission[TARGET].dropna().unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Converted must contain only 0/1, got {sorted(unique_values)}")
    if submission[TARGET].isna().any():
        raise ValueError("Converted contains missing values")
