from __future__ import annotations

import pandas as pd

from modeling import (
    ID_COL,
    TARGET,
    add_features,
    load_best_config,
    load_data,
    make_model_specs,
    project_paths,
    validate_submission,
)


def main() -> None:
    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    train, public, private = load_data()
    config = load_best_config()

    labeled = pd.concat([train, public], ignore_index=True)
    x_labeled = add_features(labeled.drop(columns=[TARGET]))
    y_labeled = labeled[TARGET].astype(int)
    x_private = add_features(private)

    specs = make_model_specs(x_labeled)
    model_name = str(config["model"])
    if model_name not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Best model {model_name!r} is unavailable. Available models: {available}")

    threshold = float(config["threshold"])
    model = specs[model_name]

    print(f"Training final model on train + public: {model_name}")
    model.fit(x_labeled, y_labeled)
    probabilities = model.predict_proba(x_private)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    private_probabilities = pd.DataFrame(
        {
            ID_COL: private[ID_COL],
            "predicted_probability": probabilities,
            TARGET: predictions,
        }
    )
    private_probabilities.to_csv(paths["results"] / "private_probabilities.csv", index=False)

    submission = private_probabilities[[ID_COL, TARGET]].copy()
    validate_submission(submission, private)
    submission.to_csv(paths["submission"], index=False)

    print(f"Saved private probabilities to: {paths['results'] / 'private_probabilities.csv'}")
    print(f"Saved submission to: {paths['submission']}")
    print("\nSubmission checks passed:")
    print(f"  rows: {len(submission)}")
    print(f"  columns: {list(submission.columns)}")
    print(f"  positive predictions: {int(submission[TARGET].sum())}")
    print(f"  threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
