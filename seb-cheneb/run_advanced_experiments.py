from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from modeling import (
    ID_COL,
    TARGET,
    THRESHOLDS,
    add_features,
    load_data,
    project_paths,
    split_xy,
    validate_submission,
)


def metric_row(name: str, y_true: pd.Series, probabilities: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLDS]
    best_idx = int(np.argmax(scores))
    threshold = float(THRESHOLDS[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    row = {
        "model": name,
        "public_f1": float(scores[best_idx]),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "positive_predictions": int(predictions.sum()),
        "status": "ok",
        "error": "",
    }
    return row, predictions


def feature_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [col for col in x.columns if x[col].dtype == "object"]
    num_cols = [col for col in x.columns if col not in cat_cols]
    return num_cols, cat_cols


def make_ordinal_preprocessor(x: pd.DataFrame, scaled: bool = False) -> ColumnTransformer:
    num_cols, cat_cols = feature_columns(x)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def fit_probability_model(name: str, estimator, x_train, y_train, x_val, y_val, x_private):
    estimator.fit(x_train, y_train)
    public_prob = estimator.predict_proba(x_val)[:, 1]
    private_prob = estimator.predict_proba(x_private)[:, 1]
    row, public_pred = metric_row(name, y_val, public_prob)
    return row, public_prob, public_pred, private_prob, estimator


def run_lightgbm(x_train, y_train, x_val, y_val, x_private):
    from lightgbm import LGBMClassifier

    model = Pipeline(
        [
            ("prep", make_ordinal_preprocessor(x_train)),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=900,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=35,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=5.0,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    return fit_probability_model("lightgbm_balanced", model, x_train, y_train, x_val, y_val, x_private)


def run_xgboost(x_train, y_train, x_val, y_val, x_private):
    from xgboost import XGBClassifier

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    model = Pipeline(
        [
            ("prep", make_ordinal_preprocessor(x_train)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=750,
                    learning_rate=0.025,
                    max_depth=3,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=8.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return fit_probability_model("xgboost_balanced", model, x_train, y_train, x_val, y_val, x_private)


def run_tabicl(x_train, y_train, x_val, y_val, x_private):
    from tabicl import TabICLClassifier

    prep = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = prep.fit_transform(x_train)
    val_arr = prep.transform(x_val)
    private_arr = prep.transform(x_private)
    model = TabICLClassifier(
        n_estimators=4,
        batch_size=4,
        device="cpu",
        random_state=42,
        verbose=False,
    )
    return fit_probability_model("tabicl_cpu_n4", model, train_arr, y_train, val_arr, y_val, private_arr)


def run_tabpfn(x_train, y_train, x_val, y_val, x_private):
    from tabpfn import TabPFNClassifier

    prep = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = prep.fit_transform(x_train)
    val_arr = prep.transform(x_val)
    private_arr = prep.transform(x_private)
    model = TabPFNClassifier(
        n_estimators=4,
        device="cpu",
        random_state=42,
        show_progress_bar=False,
        ignore_pretraining_limits=True,
    )
    return fit_probability_model("tabpfn_cpu_n4", model, train_arr, y_train.to_numpy(), val_arr, y_val, private_arr)


def run_autogluon(train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame):
    from autogluon.tabular import TabularPredictor

    paths = project_paths()
    model_dir = paths["results"] / "autogluon_public_fit"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    predictor = TabularPredictor(
        label=TARGET,
        problem_type="binary",
        eval_metric="f1",
        path=str(model_dir),
        verbosity=1,
    )
    predictor.fit(
        train_data=train.drop(columns=[ID_COL]),
        tuning_data=public.drop(columns=[ID_COL]),
        presets="medium_quality",
        time_limit=180,
        hyperparameters="default",
    )
    public_prob = predictor.predict_proba(public.drop(columns=[ID_COL, TARGET]))[1].to_numpy()
    private_prob = predictor.predict_proba(private.drop(columns=[ID_COL]))[1].to_numpy()
    row, public_pred = metric_row("autogluon_medium_180s", public[TARGET].astype(int), public_prob)
    return row, public_prob, public_pred, private_prob, predictor


def save_advanced_submission(best_row: dict[str, object], private: pd.DataFrame, private_prob: np.ndarray) -> None:
    paths = project_paths()
    threshold = float(best_row["threshold"])
    predictions = (private_prob >= threshold).astype(int)
    private_predictions = pd.DataFrame(
        {
            ID_COL: private[ID_COL],
            "predicted_probability": private_prob,
            TARGET: predictions,
        }
    )
    private_predictions.to_csv(paths["results"] / "advanced_private_probabilities.csv", index=False)

    submission = private_predictions[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / "advanced_submission.csv", index=False)


def main() -> None:
    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)
    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(public)
    x_private = add_features(private)

    rows: list[dict[str, object]] = []
    private_prob_by_model: dict[str, np.ndarray] = {}

    best_config_path = paths["best_config"]
    if best_config_path.exists():
        best_config = json.loads(best_config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": "current_best_from_base_runner",
                "public_f1": float(best_config["public_f1"]),
                "threshold": float(best_config["threshold"]),
                "accuracy": float(best_config["public_accuracy"]),
                "precision": float(best_config["public_precision"]),
                "recall": float(best_config["public_recall"]),
                "positive_predictions": int(best_config["positive_predictions_public"]),
                "status": "reference",
                "error": "",
            }
        )

    candidates = [
        ("lightgbm_balanced", lambda: run_lightgbm(x_train, y_train, x_val, y_val, x_private)),
        ("xgboost_balanced", lambda: run_xgboost(x_train, y_train, x_val, y_val, x_private)),
        ("tabicl_cpu_n4", lambda: run_tabicl(x_train, y_train, x_val, y_val, x_private)),
        ("tabpfn_cpu_n4", lambda: run_tabpfn(x_train, y_train, x_val, y_val, x_private)),
        ("autogluon_medium_180s", lambda: run_autogluon(train, public, private)),
    ]

    for name, runner in candidates:
        print(f"\nRunning {name}...")
        try:
            row, public_prob, public_pred, private_prob, _ = runner()
            private_prob_by_model[str(row["model"])] = private_prob
            print(
                f"  F1={row['public_f1']:.6f} threshold={row['threshold']:.3f} "
                f"precision={row['precision']:.6f} recall={row['recall']:.6f}"
            )
        except Exception as exc:
            row = {
                "model": name,
                "public_f1": np.nan,
                "threshold": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "positive_predictions": np.nan,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {row['error']}")
            traceback.print_exc()
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False, na_position="last")
    leaderboard.to_csv(paths["results"] / "advanced_model_results.csv", index=False)

    best = leaderboard.iloc[0].to_dict()
    (paths["results"] / "advanced_best_config.json").write_text(
        json.dumps(best, indent=2, default=str),
        encoding="utf-8",
    )

    best_name = str(best["model"])
    if best_name in private_prob_by_model:
        save_advanced_submission(best, private, private_prob_by_model[best_name])
        print(f"\nAdvanced winner submission saved to: {paths['root'] / 'advanced_submission.csv'}")
    else:
        print("\nReference base model is still best; existing submission.csv remains the best measured file.")

    print("\nAdvanced leaderboard:")
    print(leaderboard.to_string(index=False))
    print(f"\nSaved advanced results to: {paths['results'] / 'advanced_model_results.csv'}")


if __name__ == "__main__":
    main()
