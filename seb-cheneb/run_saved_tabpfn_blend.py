from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modeling import ID_COL, TARGET, add_features, load_data, make_model_specs, project_paths, split_xy, validate_submission
from run_advanced_experiments import make_ordinal_preprocessor


def exact_score(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    order = np.argsort(-probabilities)
    sorted_y = y_true[order].astype(int)
    sorted_probs = probabilities[order]
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
    return (
        {
            "public_f1": float(f1_score(y_true, predictions)),
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "positive_predictions": int(predictions.sum()),
        },
        predictions,
    )


def fit_logreg_and_tabicl():
    from tabicl import TabICLClassifier

    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(public)
    x_private = add_features(private)

    logreg = make_model_specs(x_train)["logreg_balanced_c2"]
    logreg.fit(x_train, y_train)
    logreg_public = logreg.predict_proba(x_val)[:, 1]
    logreg_private = logreg.predict_proba(x_private)[:, 1]

    preprocessor = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = preprocessor.fit_transform(x_train)
    val_arr = preprocessor.transform(x_val)
    private_arr = preprocessor.transform(x_private)
    tabicl = TabICLClassifier(n_estimators=4, batch_size=4, device="cpu", random_state=42, verbose=False)
    tabicl.fit(train_arr, y_train)
    tabicl_public = tabicl.predict_proba(val_arr)[:, 1]
    tabicl_private = tabicl.predict_proba(private_arr)[:, 1]

    return public, private, y_val.to_numpy().astype(int), logreg_public, logreg_private, tabicl_public, tabicl_private


def main() -> None:
    paths = project_paths()
    tabpfn_public_path = paths["results"] / "tabpfn_public_predictions_train10000_n2.csv"
    tabpfn_private_path = paths["results"] / "tabpfn_private_probabilities_train10000_n2.csv"
    if not tabpfn_public_path.exists() or not tabpfn_private_path.exists():
        raise FileNotFoundError("Run run_tabpfn_experiment.py full-train first.")

    public, private, y_val, logreg_public, logreg_private, tabicl_public, tabicl_private = fit_logreg_and_tabicl()
    tabpfn_public = pd.read_csv(tabpfn_public_path)["predicted_probability"].to_numpy()
    tabpfn_private = pd.read_csv(tabpfn_private_path)["predicted_probability"].to_numpy()

    rows = []
    best = None
    best_private = None

    for tabpfn_weight in np.arange(0.0, 0.151, 0.01):
        remaining = 1.0 - tabpfn_weight
        for logreg_share in np.arange(0.65, 0.86, 0.005):
            logreg_weight = remaining * logreg_share
            tabicl_weight = remaining - logreg_weight
            public_prob = (
                logreg_weight * logreg_public
                + tabicl_weight * tabicl_public
                + tabpfn_weight * tabpfn_public
            )
            private_prob = (
                logreg_weight * logreg_private
                + tabicl_weight * tabicl_private
                + tabpfn_weight * tabpfn_private
            )
            row, _ = exact_score(y_val, public_prob)
            row["model"] = (
                f"blend_logreg_{logreg_weight:.3f}_tabicl_{tabicl_weight:.3f}_tabpfn_{tabpfn_weight:.3f}"
            )
            rows.append(row)
            if best is None or float(row["public_f1"]) > float(best["public_f1"]):
                best = row
                best_private = private_prob

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    leaderboard.to_csv(paths["results"] / "saved_tabpfn_blend_results.csv", index=False)
    (paths["results"] / "saved_tabpfn_blend_best.json").write_text(
        json.dumps(best, indent=2, default=str),
        encoding="utf-8",
    )

    threshold = float(best["threshold"])
    predictions = (best_private >= threshold).astype(int)
    out = pd.DataFrame({ID_COL: private[ID_COL], "predicted_probability": best_private, TARGET: predictions})
    out.to_csv(paths["results"] / "saved_tabpfn_blend_private_probabilities.csv", index=False)
    submission = out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / "saved_tabpfn_blend_submission.csv", index=False)

    overall_path = paths["results"] / "overall_best_config.json"
    current = json.loads(overall_path.read_text(encoding="utf-8"))
    if float(best["public_f1"]) > float(current["public_f1"]):
        submission.to_csv(paths["submission"], index=False)
        overall_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
        print("TabPFN blend beat current best; copied to submission.csv")
    else:
        print("TabPFN blend did not beat current best; kept submission.csv unchanged")

    print(leaderboard.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
