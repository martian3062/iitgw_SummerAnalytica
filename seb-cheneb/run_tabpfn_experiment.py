from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabpfn import TabPFNClassifier

from modeling import ID_COL, TARGET, THRESHOLDS, add_features, load_data, project_paths, split_xy, validate_submission


def make_tabpfn_matrix_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
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


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float, np.ndarray]:
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLDS]
    best_idx = int(np.argmax(scores))
    threshold = float(THRESHOLDS[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    return threshold, float(scores[best_idx]), predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=0, help="Use 0 for all train.csv rows.")
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    train, public, private = load_data()
    if args.max_train and args.max_train < len(train):
        train = train.sample(n=args.max_train, random_state=42).sort_index()

    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(public)
    x_private = add_features(private)

    preprocessor = make_tabpfn_matrix_preprocessor(x_train)
    x_train_arr = preprocessor.fit_transform(x_train).astype("float32")
    x_val_arr = preprocessor.transform(x_val).astype("float32")
    x_private_arr = preprocessor.transform(x_private).astype("float32")

    model = TabPFNClassifier(
        n_estimators=args.n_estimators,
        device=args.device,
        random_state=42,
        show_progress_bar=True,
        ignore_pretraining_limits=True,
        fit_mode="low_memory",
        memory_saving_mode=True,
    )

    started = time.time()
    model.fit(x_train_arr, y_train.to_numpy())
    public_prob = model.predict_proba(x_val_arr)[:, 1]
    threshold, public_f1, public_pred = tune_threshold(y_val, public_prob)
    elapsed = time.time() - started

    private_prob = model.predict_proba(x_private_arr)[:, 1]
    private_pred = (private_prob >= threshold).astype(int)

    row = {
        "model": f"tabpfn_{args.device}_n{args.n_estimators}_train{len(train)}",
        "public_f1": public_f1,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_val, public_pred)),
        "precision": float(precision_score(y_val, public_pred, zero_division=0)),
        "recall": float(recall_score(y_val, public_pred, zero_division=0)),
        "positive_predictions": int(public_pred.sum()),
        "elapsed_seconds": round(elapsed, 3),
        "train_rows": int(len(train)),
    }

    result_path = paths["results"] / f"tabpfn_result_train{len(train)}_n{args.n_estimators}.json"
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            ID_COL: public[ID_COL],
            "true_converted": y_val.to_numpy(),
            "predicted_probability": public_prob,
            TARGET: public_pred,
        }
    ).to_csv(paths["results"] / f"tabpfn_public_predictions_train{len(train)}_n{args.n_estimators}.csv", index=False)

    private_out = pd.DataFrame(
        {
            ID_COL: private[ID_COL],
            "predicted_probability": private_prob,
            TARGET: private_pred,
        }
    )
    private_out.to_csv(paths["results"] / f"tabpfn_private_probabilities_train{len(train)}_n{args.n_estimators}.csv", index=False)
    submission = private_out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / f"tabpfn_submission_train{len(train)}_n{args.n_estimators}.csv", index=False)

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
