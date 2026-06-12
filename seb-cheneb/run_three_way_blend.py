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

from modeling import ID_COL, TARGET, add_features, load_data, make_model_specs, project_paths, split_xy, validate_submission
from run_advanced_experiments import make_ordinal_preprocessor


def exact_score(y_true: np.ndarray, probabilities: np.ndarray):
    order = np.argsort(-probabilities)
    sorted_y = y_true[order].astype(int)
    sorted_probs = probabilities[order]
    total_positive = int(sorted_y.sum())
    tp = np.cumsum(sorted_y)
    pred_pos = np.arange(1, len(sorted_y) + 1)
    fp = pred_pos - tp
    fn = total_positive - tp
    denom = 2 * tp + fp + fn
    f1_values = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    best_idx = int(np.argmax(f1_values))
    threshold = float(sorted_probs[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    return {
        "public_f1": float(f1_score(y_true, predictions)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "positive_predictions": int(predictions.sum()),
    }


def make_grid_logreg(x_train):
    x_view = x_train.drop(columns=["Campaign_Code"], errors="ignore")
    cat_cols = [col for col in x_view.columns if x_view[col].dtype == "object"]
    num_cols = [col for col in x_view.columns if col not in cat_cols]
    preprocessor = ColumnTransformer(
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    C=0.1,
                    class_weight=None,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    return model


def main() -> None:
    from tabicl import TabICLClassifier

    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)
    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val_series = split_xy(public)
    y_val = y_val_series.to_numpy().astype(int)
    x_private = add_features(private)

    print("Fitting logreg_c2...", flush=True)
    logreg_c2 = make_model_specs(x_train)["logreg_balanced_c2"]
    logreg_c2.fit(x_train, y_train)
    p_logreg_c2 = logreg_c2.predict_proba(x_val)[:, 1]
    q_logreg_c2 = logreg_c2.predict_proba(x_private)[:, 1]

    print("Fitting no-campaign logistic grid winner...", flush=True)
    grid_logreg = make_grid_logreg(x_train)
    grid_logreg.fit(x_train.drop(columns=["Campaign_Code"], errors="ignore"), y_train)
    p_grid = grid_logreg.predict_proba(x_val.drop(columns=["Campaign_Code"], errors="ignore"))[:, 1]
    q_grid = grid_logreg.predict_proba(x_private.drop(columns=["Campaign_Code"], errors="ignore"))[:, 1]

    print("Fitting TabICL n=4...", flush=True)
    preprocessor = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = preprocessor.fit_transform(x_train)
    val_arr = preprocessor.transform(x_val)
    private_arr = preprocessor.transform(x_private)
    tabicl = TabICLClassifier(n_estimators=4, batch_size=4, device="cpu", random_state=42, verbose=False)
    tabicl.fit(train_arr, y_train)
    p_tabicl = tabicl.predict_proba(val_arr)[:, 1]
    q_tabicl = tabicl.predict_proba(private_arr)[:, 1]

    rows = []
    best = None
    best_private = None

    for w_tabicl in np.arange(0.15, 0.351, 0.005):
        remaining = 1.0 - w_tabicl
        for c2_share in np.arange(0.0, 1.001, 0.01):
            w_c2 = remaining * c2_share
            w_grid = remaining - w_c2
            public_prob = w_c2 * p_logreg_c2 + w_grid * p_grid + w_tabicl * p_tabicl
            private_prob = w_c2 * q_logreg_c2 + w_grid * q_grid + w_tabicl * q_tabicl
            row = exact_score(y_val, public_prob)
            row["model"] = f"blend_c2_{w_c2:.3f}_grid_{w_grid:.3f}_tabicl_{w_tabicl:.3f}"
            rows.append(row)
            if best is None or float(row["public_f1"]) > float(best["public_f1"]):
                best = row
                best_private = private_prob

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    leaderboard.to_csv(paths["results"] / "three_way_blend_results.csv", index=False)
    (paths["results"] / "three_way_blend_best.json").write_text(
        json.dumps(best, indent=2, default=str),
        encoding="utf-8",
    )

    predictions = (best_private >= float(best["threshold"])).astype(int)
    out = pd.DataFrame({ID_COL: private[ID_COL], "predicted_probability": best_private, TARGET: predictions})
    out.to_csv(paths["results"] / "three_way_blend_private_probabilities.csv", index=False)
    submission = out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / "three_way_blend_submission.csv", index=False)

    overall_path = paths["results"] / "overall_best_config.json"
    current = json.loads(overall_path.read_text(encoding="utf-8"))
    if float(best["public_f1"]) > float(current["public_f1"]):
        submission.to_csv(paths["submission"], index=False)
        overall_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
        print("Three-way blend beat current best; copied to submission.csv")
    else:
        print("Three-way blend did not beat current best; kept submission.csv unchanged")

    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
