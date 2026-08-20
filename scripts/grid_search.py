#!/usr/bin/env python
"""Grid search over LightGBM hyperparameters, every combination logged to MLflow.

Reuses train.py's run_training() (same purged CV + held-out test split, same
"autocall-duration" experiment) so grid runs sit next to single-run training
in the MLflow UI and can be compared there: sort the runs table by
cv_mae_mean, or use the parallel-coordinates plot to see which
hyperparameters actually move it.

Grid runs skip the per-round loss curve and model artifact (tags.grid_search
= "true") to keep the sweep's wall time down, since the CV/test summary
metrics are all a comparison needs. Once the sweep finishes, the best
combination (lowest cv_mae_mean) is re-run once with both enabled
(tags.role = "best"), so the winner still ends up with a full training curve
and a loadable model.

PARAM_GRID's 4*4*4*3*3 = 576 combinations (num_leaves, learning_rate,
min_child_samples, n_estimators, and random_state, the model-init seed, to
see how much the ranking moves just from re-seeding) take roughly an hour at
~5-6s/combo without curve logging. Trim PARAM_GRID for a quicker pass.

    docker compose up -d mlflow
    docker compose run --rm dev python scripts/grid_search.py

Then compare runs at http://localhost:5000.
"""

import argparse
import itertools
from pathlib import Path

import joblib
import mlflow

from preprocess import latest_version
from train import DEFAULT_LGB_PARAMS, EXPERIMENT_NAME, load_dataset, run_training

PARAM_GRID = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [5, 10, 20, 50],
    "n_estimators": [300, 500, 1000],
    "random_state": [0, 1, 2],  # model-initialization seed, also checks how seed-sensitive the ranking is
}


def grid_combinations(param_grid):
    keys = list(param_grid)
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--version", type=int, default=None,
        help="Dataset version to train on, e.g. 3 for v3/ (default: latest existing)",
    )
    parser.add_argument(
        "--model-out", type=Path, default=None,
        help="Where to save the best model (default: data/processed/vN/model.joblib)",
    )
    args = parser.parse_args()

    version = args.version or latest_version(args.data_dir)
    features, splits = load_dataset(args.data_dir, version)
    combos = list(grid_combinations(PARAM_GRID))

    print(f"v{version}: {len(features):,} rows, {len(combos)} combinations over {list(PARAM_GRID)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for i, overrides in enumerate(combos):
        lgb_params = {**DEFAULT_LGB_PARAMS, **overrides}
        _, cv_results, test_mae, _, _ = run_training(
            features, splits, version, lgb_params,
            run_name=f"grid_v{version}_{i:03d}",
            log_model=False, log_fold_curves=False, log_final_curve=False,
            tags={"grid_search": "true"},
        )
        cv_mae_mean = cv_results.MAE.mean()
        results.append({**overrides, "cv_mae_mean": cv_mae_mean, "test_mae": test_mae})
        print(f"[{i + 1}/{len(combos)}] {overrides} -> cv_mae_mean={cv_mae_mean:.4f} test_mae={test_mae:.4f}")

    best = min(results, key=lambda r: r["cv_mae_mean"])
    best_params = {k: best[k] for k in PARAM_GRID}
    print(f"\nBest: {best_params} -> cv_mae_mean={best['cv_mae_mean']:.4f} test_mae={best['test_mae']:.4f}")

    print("Re-running best combination with the full model + loss curve logged...")
    lgb_params = {**DEFAULT_LGB_PARAMS, **best_params}
    model, cv_results, test_mae, product_breakdown, percentile_breakdown = run_training(
        features, splits, version, lgb_params,
        run_name=f"grid_v{version}_best",
        log_model=True, log_fold_curves=False, log_final_curve=True,
        tags={"grid_search": "true", "role": "best"},
    )
    print("\nBest model, test MAE by product_type:")
    print(product_breakdown.to_string(index=False))
    print("\nBest model, test MAE by target quartile:")
    print(percentile_breakdown.to_string(index=False))

    model_out = args.model_out or args.data_dir / f"v{version}" / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"saved: {model_out}")


if __name__ == "__main__":
    main()
