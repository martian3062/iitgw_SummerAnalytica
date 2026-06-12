from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7,99")
    parser.add_argument("--n-estimators", type=int, default=4)
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    from tabicl import TabICLClassifier

    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)
    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val_series = split_xy(public)
    y_val = y_val_series.to_numpy().astype(int)
    x_private = add_features(private)

    print("Fitting logreg_c2...", flush=True)
    logreg = make_model_specs(x_train)["logreg_balanced_c2"]
    logreg.fit(x_train, y_train)
    p_logreg = logreg.predict_proba(x_val)[:, 1]
    q_logreg = logreg.predict_proba(x_private)[:, 1]

    print("Preparing TabICL matrix...", flush=True)
    preprocessor = make_ordinal_preprocessor(x_train, scaled=True)
    train_arr = preprocessor.fit_transform(x_train)
    val_arr = preprocessor.transform(x_val)
    private_arr = preprocessor.transform(x_private)

    tabicl_public = []
    tabicl_private = []
    seed_rows = []
    for seed in seeds:
        print(f"Fitting TabICL seed={seed} n={args.n_estimators}...", flush=True)
        model = TabICLClassifier(
            n_estimators=args.n_estimators,
            batch_size=4,
            device="cpu",
            random_state=seed,
            verbose=False,
        )
        model.fit(train_arr, y_train)
        public_prob = model.predict_proba(val_arr)[:, 1]
        private_prob = model.predict_proba(private_arr)[:, 1]
        tabicl_public.append(public_prob)
        tabicl_private.append(private_prob)
        row = exact_score(y_val, public_prob)
        row["model"] = f"tabicl_seed_{seed}_n{args.n_estimators}"
        seed_rows.append(row)
        print(f"  seed F1={row['public_f1']:.6f}", flush=True)

    tabicl_public_matrix = np.column_stack(tabicl_public)
    tabicl_private_matrix = np.column_stack(tabicl_private)
    tabicl_avg_public = tabicl_public_matrix.mean(axis=1)
    tabicl_avg_private = tabicl_private_matrix.mean(axis=1)

    rows = list(seed_rows)
    rows.append({**exact_score(y_val, tabicl_avg_public), "model": "tabicl_seed_average"})

    best = None
    best_private = None
    for logreg_weight in np.arange(0.60, 0.86, 0.001):
        tabicl_weight = 1.0 - logreg_weight
        public_prob = logreg_weight * p_logreg + tabicl_weight * tabicl_avg_public
        private_prob = logreg_weight * q_logreg + tabicl_weight * tabicl_avg_private
        row = exact_score(y_val, public_prob)
        row["model"] = f"blend_logreg_{logreg_weight:.3f}_tabicl_seedavg_{tabicl_weight:.3f}"
        rows.append(row)
        if best is None or float(row["public_f1"]) > float(best["public_f1"]):
            best = row
            best_private = private_prob

    # Random convex weights among TabICL seeds plus logreg, deterministic and bounded.
    rng = np.random.default_rng(42)
    for idx in range(4000):
        tabicl_total = rng.uniform(0.15, 0.35)
        logreg_weight = 1.0 - tabicl_total
        seed_weights = rng.dirichlet(np.ones(len(seeds))) * tabicl_total
        public_prob = logreg_weight * p_logreg + tabicl_public_matrix @ seed_weights
        private_prob = logreg_weight * q_logreg + tabicl_private_matrix @ seed_weights
        row = exact_score(y_val, public_prob)
        row["model"] = (
            f"random_seed_blend_{idx}_logreg_{logreg_weight:.3f}_"
            + "_".join(f"s{seed}_{weight:.3f}" for seed, weight in zip(seeds, seed_weights))
        )
        rows.append(row)
        if float(row["public_f1"]) > float(best["public_f1"]):
            best = row
            best_private = private_prob

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    leaderboard.to_csv(paths["results"] / "tabicl_seed_blend_results.csv", index=False)
    (paths["results"] / "tabicl_seed_blend_best.json").write_text(
        json.dumps(best, indent=2, default=str),
        encoding="utf-8",
    )

    predictions = (best_private >= float(best["threshold"])).astype(int)
    out = pd.DataFrame({ID_COL: private[ID_COL], "predicted_probability": best_private, TARGET: predictions})
    out.to_csv(paths["results"] / "tabicl_seed_blend_private_probabilities.csv", index=False)
    submission = out[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["root"] / "tabicl_seed_blend_submission.csv", index=False)

    overall_path = paths["results"] / "overall_best_config.json"
    current = json.loads(overall_path.read_text(encoding="utf-8"))
    if float(best["public_f1"]) > float(current["public_f1"]):
        submission.to_csv(paths["submission"], index=False)
        overall_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
        print("Seed blend beat current best; copied to submission.csv")
    else:
        print("Seed blend did not beat current best; kept submission.csv unchanged")

    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
