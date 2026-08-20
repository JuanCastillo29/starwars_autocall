#!/usr/bin/env python
"""Train a LightGBM duration regressor on the purged, date-based folds.

Reads data/processed/vN/{features.csv,splits.csv} (row-aligned, see
split.py). Runs the 5-fold purged CV first to report an honest out-of-time
MAE for the fold_0..fold_4 columns split.py wrote (purged rows are excluded
from both train and val), then fits one final model on the full train_val
pool and scores it on the held-out test split.

    python scripts/train.py
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocess import TARGET, latest_version

N_FOLDS = 5
LGB_PARAMS = dict(
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


def fit_predict(train_df, eval_df, feature_cols, early_stopping_rounds=50):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        train_df[feature_cols], train_df[TARGET],
        eval_X=eval_df[feature_cols], eval_y=eval_df[TARGET],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model, model.predict(eval_df[feature_cols])


def cross_validate(features, splits, feature_cols, n_folds=N_FOLDS):
    rows = []
    for k in range(n_folds):
        col = f"fold_{k}"
        train_mask = splits[col] == "train"
        val_mask = splits[col] == "val"
        _, preds = fit_predict(features[train_mask], features[val_mask], feature_cols)
        mae = mean_absolute_error(features.loc[val_mask, TARGET], preds)
        rows.append({"fold": k, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()), "MAE": round(mae, 4)})
    return pd.DataFrame(rows)


def fit_final(features, splits, feature_cols):
    train_val_mask = splits.split == "train_val"
    test_mask = splits.split == "test"
    model, preds = fit_predict(features[train_val_mask], features[test_mask], feature_cols)
    mae = mean_absolute_error(features.loc[test_mask, TARGET], preds)
    return model, mae, int(test_mask.sum())


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
    args = parser.parse_args()

    version = args.version or latest_version(args.data_dir)
    features, splits = load_dataset(args.data_dir, version)
    feature_cols = [c for c in features.columns if c != TARGET]

    print(f"v{version}: {len(features):,} rows, {len(feature_cols)} features")

    cv_results = cross_validate(features, splits, feature_cols)
    print("\nPurged 5-fold CV:")
    print(cv_results.to_string(index=False))
    print(f"mean MAE: {cv_results.MAE.mean():.4f}  std: {cv_results.MAE.std():.4f}")

    model, test_mae, n_test = fit_final(features, splits, feature_cols)
    print(f"\nFinal model trained on train_val, scored on {n_test:,} held-out test rows: MAE = {test_mae:.4f}")

    model_out = args.model_out or args.data_dir / f"v{version}" / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"saved: {model_out}")


if __name__ == "__main__":
    main()
