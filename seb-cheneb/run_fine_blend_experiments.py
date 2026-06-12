from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modeling import (
    ID_COL,
    TARGET,
    add_features,
    load_data,
    make_model_specs,
    project_paths,
    split_xy,
    validate_submission,
)
from run_advanced_experiments import make_ordinal_preprocessor


def best_exact_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, np.ndarray]:
    order = np.argsort(-probabilities)
    sorted_probs = probabilities[order]
    sorted_y = y_true[order].astype(int)

    total_positive = int(sorted_y.sum())
    true_positive = np.cumsum(sorted_y)
    predicted_positive = np.arange(1, len(sorted_y) + 1)
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    denom = (2 * true_positive + false_positive + false_negative).astype(float)
    f1_values = np.divide(2 * true_positive, denom, out=np.zeros_like(denom), where=denom > 0)

    best_idx = int(np.argmax(f1_values))
    cutoff = float(sorted_probs[best_idx])
    predictions = (probabilities >= cutoff).astype(int)
    return cutoff, float(f1_score(y_true, predictions)), predictions


def score(name: str, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    threshold, public_f1, predictions = best_exact_threshold(y_true, probabilities)
    return {
        "model": name,
        "public_f1": public_f1,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "positive_predictions": int(predictions.sum()),
    }


def fit_logreg_c2(x_train, y_train, x_val, x_private) -> tuple[np.ndarray, np.ndarray]:
    model = make_model_specs(x_train)["logreg_balanced_c2"]
    model.fit(x_train, y_train)
    return model.predict_proba(x_val)[:, 1], model.predict_proba(x_private)[:, 1]


def fit_tabicl(x_train, y_train, x_val, x_private, n_estimators: int) -> tuple[np.ndarray, np.ndarray]:
    from tabicl import TabICLClassifier

    preprocessor = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = preprocessor.fit_transform(x_train)
    val_arr = preprocessor.transform(x_val)
    private_arr = preprocessor.transform(x_private)
    model = TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=4,
        device="cpu",
        random_state=42,
        verbose=False,
    )
    model.fit(train_arr, y_train)
    return model.predict_proba(val_arr)[:, 1], model.predict_proba(private_arr)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabicl-estimators", type=int, default=4)
    parser.add_argument("--weight-min", type=float, default=0.50)
    parser.add_argument("--weight-max", type=float, default=0.90)
    parser.add_argument("--weight-step", type=float, default=0.005)
    args = parser.parse_args()

    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val_series = split_xy(public)
    y_val = y_val_series.to_numpy().astype(int)
    x_private = add_features(private)

    print("Fitting logreg_balanced_c2...", flush=True)
    logreg_public, logreg_private = fit_logreg_c2(x_train, y_train, x_val, x_private)

    print(f"Fitting TabICL n_estimators={args.tabicl_estimators}...", flush=True)
    tabicl_public, tabicl_private = fit_tabicl(
        x_train,
        y_train,
        x_val,
        x_private,
        n_estimators=args.tabicl_estimators,
    )

    rows: list[dict[str, object]] = [
        score("logreg_c2_exact", y_val, logreg_public),
        score(f"tabicl_n{args.tabicl_estimators}_exact", y_val, tabicl_public),
    ]

    best_row = rows[0]
    best_private = logreg_private

    weights = np.arange(args.weight_min, args.weight_max + args.weight_step / 2, args.weight_step)
    for logreg_weight in weights:
        tabicl_weight = 1.0 - logreg_weight
        public_prob = logreg_weight * logreg_public + tabicl_weight * tabicl_public
        private_prob = logreg_weight * logreg_private + tabicl_weight * tabicl_private
        row = score(
            f"fine_blend_logreg_{logreg_weight:.3f}_tabicl_{tabicl_weight:.3f}",
            y_val,
            public_prob,
        )
        rows.append(row)
        if float(row["public_f1"]) > float(best_row["public_f1"]):
            best_row = row
            best_private = private_prob

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    result_path = paths["results"] / f"fine_blend_results_tabicl_n{args.tabicl_estimators}.csv"
    config_path = paths["results"] / f"fine_blend_best_tabicl_n{args.tabicl_estimators}.json"
    private_path = paths["results"] / f"fine_blend_private_probabilities_tabicl_n{args.tabicl_estimators}.csv"
    submission_path = paths["root"] / f"fine_blend_submission_tabicl_n{args.tabicl_estimators}.csv"

    leaderboard.to_csv(result_path, index=False)
    config_path.write_text(json.dumps(best_row, indent=2, default=str), encoding="utf-8")

    threshold = float(best_row["threshold"])
    predictions = (best_private >= threshold).astype(int)
    private_out = pd.DataFrame(
        {
            ID_COL: private[ID_COL],
            "predicted_probability": best_private,
            TARGET: predictions,
        }
    )
    private_out.to_csv(private_path, index=False)
    submission = private_out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(submission_path, index=False)

    overall_best_path = paths["results"] / "overall_best_config.json"
    fallback_best_path = paths["results"] / "blend_best_config.json"
    current_best_path = overall_best_path if overall_best_path.exists() else fallback_best_path
    current_best = json.loads(current_best_path.read_text(encoding="utf-8"))
    if float(best_row["public_f1"]) > float(current_best["public_f1"]):
        submission.to_csv(paths["submission"], index=False)
        overall_best_path.write_text(
            json.dumps(best_row, indent=2, default=str),
            encoding="utf-8",
        )
        print("Fine blend beat current best; copied to submission.csv", flush=True)
    else:
        print("Fine blend did not beat current best; kept submission.csv unchanged", flush=True)

    print("\nFine blend leaderboard top 15:")
    print(leaderboard.head(15).to_string(index=False))
    print(f"\nSaved results to: {result_path}")
    print(f"Saved candidate submission to: {submission_path}")


if __name__ == "__main__":
    main()
