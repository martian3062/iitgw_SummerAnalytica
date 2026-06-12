from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling import ID_COL, TARGET, ColumnDropper, add_features, load_data, project_paths, split_xy, validate_submission


def onehot(min_frequency):
    kwargs = {"handle_unknown": "ignore", "sparse_output": False}
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency
    return OneHotEncoder(**kwargs)


def make_preprocessor(x: pd.DataFrame, min_frequency):
    cat_cols = [col for col in x.columns if x[col].dtype == "object"]
    num_cols = [col for col in x.columns if col not in cat_cols]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", onehot(min_frequency)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def exact_threshold(y_true: np.ndarray, probabilities: np.ndarray):
    order = np.argsort(-probabilities)
    sorted_probs = probabilities[order]
    sorted_y = y_true[order].astype(int)
    total_positive = int(sorted_y.sum())
    true_positive = np.cumsum(sorted_y)
    predicted_positive = np.arange(1, len(sorted_y) + 1)
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    denom = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(2 * true_positive, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    best_idx = int(np.argmax(f1_values))
    threshold = float(sorted_probs[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    return threshold, predictions


def score_row(name: str, y_true: np.ndarray, probabilities: np.ndarray):
    threshold, predictions = exact_threshold(y_true, probabilities)
    return {
        "model": name,
        "public_f1": float(f1_score(y_true, predictions)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "positive_predictions": int(predictions.sum()),
    }, predictions


def main() -> None:
    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val_series = split_xy(public)
    y_val = y_val_series.to_numpy().astype(int)
    x_private = add_features(private)

    rows = []
    best_row = None
    best_private_prob = None

    min_freqs = [None, 2, 3, 5, 8, 10, 15, 20, 30]
    c_values = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    class_weights = ["balanced", None]
    drop_campaign_options = [False, True]

    for drop_campaign in drop_campaign_options:
        train_view = x_train.drop(columns=["Campaign_Code"], errors="ignore") if drop_campaign else x_train
        val_view = x_val.drop(columns=["Campaign_Code"], errors="ignore") if drop_campaign else x_val
        private_view = x_private.drop(columns=["Campaign_Code"], errors="ignore") if drop_campaign else x_private
        for min_frequency in min_freqs:
            preprocessor = make_preprocessor(train_view, min_frequency)
            for c_value in c_values:
                for class_weight in class_weights:
                    name = (
                        f"logreg_grid_c{c_value:g}_mf{min_frequency}_"
                        f"cw{class_weight or 'none'}_dropcampaign{int(drop_campaign)}"
                    )
                    print(f"Training {name}...", flush=True)
                    model = Pipeline(
                        [
                            ("preprocess", preprocessor),
                            (
                                "model",
                                LogisticRegression(
                                    max_iter=5000,
                                    C=c_value,
                                    class_weight=class_weight,
                                    solver="lbfgs",
                                    random_state=42,
                                ),
                            ),
                        ]
                    )
                    model.fit(train_view, y_train)
                    public_prob = model.predict_proba(val_view)[:, 1]
                    private_prob = model.predict_proba(private_view)[:, 1]
                    row, _ = score_row(name, y_val, public_prob)
                    rows.append(row)
                    if best_row is None or row["public_f1"] > best_row["public_f1"]:
                        best_row = row
                        best_private_prob = private_prob
                        print(f"  new best {row['public_f1']:.6f}", flush=True)

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    leaderboard.to_csv(paths["results"] / "logreg_grid_results.csv", index=False)
    (paths["results"] / "logreg_grid_best.json").write_text(
        json.dumps(best_row, indent=2, default=str),
        encoding="utf-8",
    )

    threshold = float(best_row["threshold"])
    private_pred = (best_private_prob >= threshold).astype(int)
    private_out = pd.DataFrame({ID_COL: private[ID_COL], "predicted_probability": best_private_prob, TARGET: private_pred})
    private_out.to_csv(paths["results"] / "logreg_grid_private_probabilities.csv", index=False)
    submission = private_out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / "logreg_grid_submission.csv", index=False)

    print("\nLogistic grid leaderboard top 20:")
    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
