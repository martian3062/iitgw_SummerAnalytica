from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modeling import (
    ID_COL,
    TARGET,
    THRESHOLDS,
    add_features,
    load_data,
    make_model_specs,
    project_paths,
    split_xy,
    validate_submission,
)
from run_advanced_experiments import run_lightgbm, run_tabicl, run_xgboost


def score_probabilities(name: str, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, object]:
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLDS]
    best_idx = int(np.argmax(scores))
    threshold = float(THRESHOLDS[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "public_f1": float(scores[best_idx]),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "positive_predictions": int(predictions.sum()),
    }


def fit_base_logreg(x_train, y_train, x_val, x_private):
    specs = make_model_specs(x_train)
    model = specs["logreg_balanced_c2"]
    model.fit(x_train, y_train)
    return (
        model.predict_proba(x_val)[:, 1],
        model.predict_proba(x_private)[:, 1],
    )


def weight_grid(n_models: int, step: float = 0.10):
    ticks = np.arange(0, 1 + step / 2, step)
    for weights in itertools.product(ticks, repeat=n_models):
        total = sum(weights)
        if abs(total - 1.0) < 1e-9 and max(weights) < 1.0:
            yield np.array(weights, dtype=float)


def main() -> None:
    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    train, public, private = load_data()
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(public)
    x_private = add_features(private)

    public_probs: dict[str, np.ndarray] = {}
    private_probs: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    print("Fitting logreg_balanced_c2...", flush=True)
    public_probs["logreg_c2"], private_probs["logreg_c2"] = fit_base_logreg(
        x_train, y_train, x_val, x_private
    )

    for name, runner in [
        ("tabicl_n4", lambda: run_tabicl(x_train, y_train, x_val, y_val, x_private)),
        ("xgboost", lambda: run_xgboost(x_train, y_train, x_val, y_val, x_private)),
        ("lightgbm", lambda: run_lightgbm(x_train, y_train, x_val, y_val, x_private)),
    ]:
        print(f"Fitting {name}...", flush=True)
        try:
            _, public_prob, _, private_prob, _ = runner()
            public_probs[name] = public_prob
            private_probs[name] = private_prob
        except Exception as exc:
            print(f"  skipped {name}: {type(exc).__name__}: {exc}", flush=True)

    for name, probabilities in public_probs.items():
        rows.append(score_probabilities(name, y_val, probabilities))

    names = list(public_probs)
    best_blend: tuple[float, dict[str, object], np.ndarray, np.ndarray] | None = None

    for size in range(2, min(4, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            combo_public = np.column_stack([public_probs[name] for name in combo])
            combo_private = np.column_stack([private_probs[name] for name in combo])
            for weights in weight_grid(size, step=0.10):
                blend_public = combo_public @ weights
                row = score_probabilities(
                    "blend__"
                    + "__".join(f"{name}_{weight:.2f}" for name, weight in zip(combo, weights)),
                    y_val,
                    blend_public,
                )
                rows.append(row)
                if best_blend is None or row["public_f1"] > best_blend[0]:
                    blend_private = combo_private @ weights
                    best_blend = (float(row["public_f1"]), row, blend_public, blend_private)

    leaderboard = pd.DataFrame(rows).sort_values("public_f1", ascending=False)
    leaderboard.to_csv(paths["results"] / "blend_model_results.csv", index=False)

    best = leaderboard.iloc[0].to_dict()
    (paths["results"] / "blend_best_config.json").write_text(
        json.dumps(best, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nBlend leaderboard top 20:")
    print(leaderboard.head(20).to_string(index=False))

    best_name = str(best["model"])
    if best_name.startswith("blend__") and best_blend is not None:
        best_private = best_blend[3]
    else:
        best_private = private_probs.get(best_name)

    if best_private is not None:
        predictions = (best_private >= float(best["threshold"])).astype(int)
        out = pd.DataFrame(
            {
                ID_COL: private[ID_COL],
                "predicted_probability": best_private,
                TARGET: predictions,
            }
        )
        out.to_csv(paths["results"] / "blend_private_probabilities.csv", index=False)
        submission = out[[ID_COL, TARGET]].copy()
        validate_submission(submission, private)
        submission.to_csv(paths["root"] / "blend_submission.csv", index=False)
        print(f"\nSaved blend submission to: {paths['root'] / 'blend_submission.csv'}")

        base_config = json.loads(paths["best_config"].read_text(encoding="utf-8"))
        if float(best["public_f1"]) > float(base_config["public_f1"]):
            submission.to_csv(paths["submission"], index=False)
            print("Blend beat base model; copied blend submission to submission.csv")
        else:
            print("Blend did not beat base model; kept existing submission.csv")

    print(f"Saved blend results to: {paths['results'] / 'blend_model_results.csv'}")


if __name__ == "__main__":
    main()
