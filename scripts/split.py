#!/usr/bin/env python
"""Date-based test split plus a purged, gapped 5-fold CV over the train/val pool.

`rv_mean`/`rv_max`/`rv_min`/`rv_spread`/`vol_regime_ratio` are all built from
`realized_vol_63d`, a 63-trading-day rolling statistic. A row just
across a fold boundary shares up to 63 trading days of the same underlying
daily-vol history with a row just on the other side, so a plain date split
leaks that overlap between train and validation. Each fold purges a
`PURGE_DAYS` gap (63 trading days -> ~88 calendar days, rounded up) on both
sides of its validation block before assigning the remainder to training.

Writes data/processed/vN/splits.csv (same version as the features.csv it
matches): one row per RFQ (rfq_id, requested_date, in the same order as
features.csv), a `split` column (train_val / test) and one fold_0..fold_4
column per CV fold, valued train / val / purged.

    python scripts/split.py
"""

import argparse
from pathlib import Path

import pandas as pd

from preprocess import assemble, latest_version

N_FOLDS = 5
TEST_FRAC = 0.2
PURGE_DAYS = 90


def block_index(dates: pd.Series, n_blocks: int) -> pd.Series:
    """Assign each row to one of `n_blocks` contiguous, roughly equal-sized
    time blocks. `dates` must already be sorted ascending.
    """
    rank = pd.Series(range(len(dates)), index=dates.index)
    return pd.qcut(rank, n_blocks, labels=False)


def train_val_test_split(dates: pd.Series, test_frac: float, purge_days: int) -> pd.Series:
    """'test' for the most recent `test_frac` of rows, 'train_val' for the
    rest, with a purge gap dropped from train_val at the boundary so the
    final holdout isn't contaminated by the 63-day vol window either.
    """
    cutoff = dates.quantile(1 - test_frac)
    gap = pd.Timedelta(days=purge_days)
    split = pd.Series("purged", index=dates.index)
    split[dates > cutoff] = "test"
    split[dates <= cutoff - gap] = "train_val"
    return split


def purged_kfold(dates: pd.Series, n_folds: int, purge_days: int) -> pd.DataFrame:
    """Blocked, purged/embargoed K-fold on `dates` (already the train_val
    pool, sorted ascending). Returns one train/val/purged column per fold.
    """
    gap = pd.Timedelta(days=purge_days)
    block = block_index(dates, n_folds)

    folds = {}
    for k in range(n_folds):
        val_mask = block == k
        val_dates = dates[val_mask]
        lo, hi = val_dates.min(), val_dates.max()
        col = pd.Series("train", index=dates.index)
        col[val_mask] = "val"
        purged_mask = ~val_mask & (dates > lo - gap) & (dates < hi + gap)
        col[purged_mask] = "purged"
        folds[f"fold_{k}"] = col
    return pd.DataFrame(folds, index=dates.index)


def build_splits(raw_dir: Path) -> pd.DataFrame:
    rfqs = assemble(raw_dir)
    dates = rfqs.requested_date

    out = rfqs[["rfq_id", "requested_date"]].copy()
    out["split"] = train_val_test_split(dates, TEST_FRAC, PURGE_DAYS)

    pool_mask = out.split == "train_val"
    cv = purged_kfold(dates[pool_mask], N_FOLDS, PURGE_DAYS)
    for col in cv.columns:
        out[col] = "n/a"
        out.loc[pool_mask, col] = cv[col]

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--version", type=int, default=None,
        help="Dataset version to build splits for, e.g. 3 for v3/ (default: latest existing)",
    )
    args = parser.parse_args()

    version = args.version or latest_version(args.out_dir)
    out_dir = args.out_dir / f"v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "splits.csv"

    splits = build_splits(args.raw_dir)
    splits.to_csv(out_path, index=False)

    counts = splits.split.value_counts()
    print(f"{out_path}: {len(splits):,} rows, " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    for col in [c for c in splits.columns if c.startswith("fold_")]:
        vc = splits.loc[splits.split == "train_val", col].value_counts()
        print(f"  {col}: " + ", ".join(f"{k}={v:,}" for k, v in vc.items()))


if __name__ == "__main__":
    main()
