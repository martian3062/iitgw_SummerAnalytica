from __future__ import annotations

import pandas as pd

from modeling import ID_COL, TARGET, project_paths, run_all_experiments, save_best_config


def main() -> None:
    paths = project_paths()
    paths["results"].mkdir(parents=True, exist_ok=True)

    results, _, _, _, y_val = run_all_experiments()
    best = results[0]

    leaderboard = pd.DataFrame([result.as_row() for result in results])
    leaderboard.to_csv(paths["results"] / "model_results.csv", index=False)

    public_predictions = pd.DataFrame(
        {
            ID_COL: pd.read_csv(paths["data"] / "public_test.csv")[ID_COL],
            "true_converted": y_val.to_numpy(),
            "predicted_probability": best.probabilities,
            TARGET: best.predictions,
        }
    )
    public_predictions.to_csv(paths["results"] / "public_predictions.csv", index=False)
    save_best_config(best, paths["best_config"])

    print("\nLeaderboard sorted by public F1:")
    print(leaderboard.to_string(index=False))
    print("\nBest model:")
    print(
        f"{best.name} | public_f1={best.f1:.6f} | threshold={best.threshold:.3f} "
        f"| precision={best.precision:.6f} | recall={best.recall:.6f}"
    )
    print(f"\nSaved results to: {paths['results']}")
    print(f"Saved best config to: {paths['best_config']}")


if __name__ == "__main__":
    main()
