#!/usr/bin/env python
"""Train a LightGBM duration regressor on the purged, date-based folds.

Reads data/processed/vN/{features.csv,splits.csv} (row-aligned, see
split.py). Runs the 5-fold purged CV first to report an honest out-of-time
MAE for the fold_0..fold_4 columns split.py wrote (purged rows are excluded
from both train and val), then fits one final model on the full train_val
pool and scores it on the held-out test split.

Every run is logged to MLflow (experiment "autocall-duration"): params,
per-fold and summary metrics, and the final model. `run_training()` takes
lgb_params as an argument specifically so scripts/grid_search.py can call it
per parameter combination and compare runs in the MLflow UI:
`docker compose up mlflow` then http://localhost:5000. Also logged as
per-boosting-round step metrics: `final_train_mae` / `final_test_mae` (the
final model's train-vs-test loss curve) and `final_gap_mae`
(test_mae - train_mae, the overfitting gap); pass --fold-curves to get the
same three per CV fold too (fold{k}_train_mae etc.).

This container logs over HTTP (MLFLOW_TRACKING_URI=http://mlflow:5000, set
in docker-compose.yml) using mlflow-skinny (see requirements-dev.txt): full
mlflow currently pins pandas<3, conflicting with this project's
pandas==3.0.5, so the tracking server and its sqlite store live only in the
separate `mlflow` image/service. This container never touches them
directly.

    python scripts/train.py
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocess import TARGET, latest_version

N_FOLDS = 5
EXPERIMENT_NAME = "autocall-duration"
DEFAULT_LGB_PARAMS = dict(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    random_state=0,
    verbosity=-1,
)


def load_dataset(data_dir: Path, version: int):
    v_dir = data_dir / f"v{version}"
    features = pd.read_csv(v_dir / "features.csv")
    splits = pd.read_csv(v_dir / "splits.csv")
    if len(features) != len(splits):
        raise ValueError(
            f"features ({len(features)} rows) and splits ({len(splits)} rows) "
            f"disagree for v{version}, were they generated together?"
        )
    return features, splits


def fit_predict(train_df, eval_df, feature_cols, lgb_params, eval_name="val", early_stopping_rounds=50):
    """Fits with both train_df and eval_df as LightGBM eval sets (not just
    eval_df), so model.evals_result_ carries a per-round curve for each;
    that's what log_loss_curve() below turns into MLflow step metrics.
    """
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train_df[feature_cols], train_df[TARGET],
        eval_X=(train_df[feature_cols], eval_df[feature_cols]),
        eval_y=(train_df[TARGET], eval_df[TARGET]),
        eval_names=["train", eval_name],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model, model.predict(eval_df[feature_cols])


def log_loss_curve(model, eval_name, prefix=""):
    """Per-boosting-round train/eval MAE and their gap, as MLflow step
    metrics: the training curve, not just the final-iteration score.
    """
    train_curve = model.evals_result_["train"]["l1"]
    eval_curve = model.evals_result_[eval_name]["l1"]
    for step, (train_mae, eval_mae) in enumerate(zip(train_curve, eval_curve)):
        mlflow.log_metrics(
            {
                f"{prefix}train_mae": train_mae,
                f"{prefix}{eval_name}_mae": eval_mae,
                f"{prefix}gap_mae": eval_mae - train_mae,
            },
            step=step,
        )


def cross_validate(features, splits, feature_cols, lgb_params, n_folds=N_FOLDS, log_curves=False):
    rows = []
    for k in range(n_folds):
        col = f"fold_{k}"
        train_mask = splits[col] == "train"
        val_mask = splits[col] == "val"
        model, preds = fit_predict(features[train_mask], features[val_mask], feature_cols, lgb_params)
        if log_curves:
            log_loss_curve(model, eval_name="val", prefix=f"fold{k}_")
        mae = mean_absolute_error(features.loc[val_mask, TARGET], preds)
        rows.append({"fold": k, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()), "MAE": round(mae, 4)})
    return pd.DataFrame(rows)


def fit_final(features, splits, feature_cols, lgb_params, log_curves=True):
    train_val_mask = splits.split == "train_val"
    test_mask = splits.split == "test"
    model, preds = fit_predict(
        features[train_val_mask], features[test_mask], feature_cols, lgb_params, eval_name="test"
    )
    if log_curves:
        log_loss_curve(model, eval_name="test", prefix="final_")
    mae = mean_absolute_error(features.loc[test_mask, TARGET], preds)
    return model, mae, int(test_mask.sum())


def run_training(features, splits, version, lgb_params, run_name=None, log_model=True, log_fold_curves=False):
    """One MLflow run: purged CV, then a final fit against the test split.

    log_fold_curves logs a full train/val/gap curve for every CV fold too
    (fold{k}_train_mae etc.), not just the final model's (final_*), off by
    default since it's 5x the step-metric HTTP calls for a diagnostic that's
    mostly useful when actually chasing an overfitting fold.

    Returns (model, cv_results, test_mae) so callers (main(), grid_search.py)
    can inspect results without re-reading MLflow.
    """
    feature_cols = [c for c in features.columns if c != TARGET]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(lgb_params)
        mlflow.log_params({"dataset_version": version, "n_folds": N_FOLDS, "n_features": len(feature_cols)})

        cv_results = cross_validate(features, splits, feature_cols, lgb_params, log_curves=log_fold_curves)
        for _, row in cv_results.iterrows():
            mlflow.log_metric("val_mae", row.MAE, step=int(row.fold))
        mlflow.log_metric("cv_mae_mean", cv_results.MAE.mean())
        mlflow.log_metric("cv_mae_std", cv_results.MAE.std())

        model, test_mae, n_test = fit_final(features, splits, feature_cols, lgb_params)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_param("n_test", n_test)

        if log_model:
            mlflow.lightgbm.log_model(model, name="model")

    return model, cv_results, test_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--version", type=int, default=None,
        help="Dataset version to train on, e.g. 3 for v3/ (default: latest existing)",
    )
    parser.add_argument(
        "--model-out", type=Path, default=None,
        help="Where to save the final model (default: data/processed/vN/model.joblib)",
    )
    parser.add_argument(
        "--fold-curves", action="store_true",
        help="Also log a per-round train/val/gap MAE curve for every CV fold, not just the final model",
    )
    args = parser.parse_args()

    version = args.version or latest_version(args.data_dir)
    features, splits = load_dataset(args.data_dir, version)

    print(f"v{version}: {len(features):,} rows, {features.shape[1] - 1} features")

    mlflow.set_experiment(EXPERIMENT_NAME)
    model, cv_results, test_mae = run_training(
        features, splits, version, DEFAULT_LGB_PARAMS, run_name=f"v{version}", log_fold_curves=args.fold_curves
    )

    print("\nPurged 5-fold CV:")
    print(cv_results.to_string(index=False))
    print(f"mean MAE: {cv_results.MAE.mean():.4f}  std: {cv_results.MAE.std():.4f}")
    print(f"\nFinal model trained on train_val, scored on held-out test rows: MAE = {test_mae:.4f}")

    model_out = args.model_out or args.data_dir / f"v{version}" / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"saved: {model_out}")


if __name__ == "__main__":
    main()
