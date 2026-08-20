#!/usr/bin/env python
"""Feature-engineering pipeline for the autocall duration model.

Turns the three raw tables into one feature matrix.
    python scripts/preprocess.py
"""

import argparse
from pathlib import Path

import pandas as pd

TARGET = "avg_duration_months"

FREQ_MONTHS = {
    "1d": 1 / 21, "daily": 1 / 21, "diario": 1 / 21,
    "1m": 1, "m": 1, "monthly": 1, "mensual": 1, "1 month": 1,
    "2m": 2, "2 months": 2,
    "3m": 3, "q": 3, "quarterly": 3, "trimestral": 3, "3 months": 3,
    "6m": 6, "semiannual": 6, "semestral": 6, "6 months": 6,
    "1y": 12, "12m": 12, "y": 12, "annual": 12, "anual": 12, "12 months": 12,
}


FEATURE_COLUMNS = [
    "product_type",
    "no_call_period_months",
    "obs_freq_months",
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "quoted_implied_vol",
    "tenor_months",
    "rv_mean",
    "rv_max",
    "rv_min",
    "rv_spread",
    "sb_mean",
    "mkt_vol",
    "vol_regime_ratio",
]


def load_raw(raw_dir: Path):
    rfqs = pd.read_csv(
        raw_dir / "rfqs.csv", parse_dates=["requested_date", "start_date", "end_date"]
    )
    vol = pd.read_csv(raw_dir / "daily_volatility.csv", parse_dates=["date"])
    ref = pd.read_csv(raw_dir / "underlyings_reference.csv")
    return rfqs, vol, ref


def add_derived_columns(rfqs: pd.DataFrame) -> pd.DataFrame:
    rfqs = rfqs.copy()
    rfqs["tenor_months"] = (rfqs.end_date - rfqs.start_date).dt.days / 30.44
    rfqs["basket_size"] = rfqs.underlyings.str.split("|").str.len()
    return rfqs


def assert_basket_size(rfqs: pd.DataFrame) -> None:
    """basket_size is fully determined by product_type. Checked here,
    not trained as a feature: it's free to compute from `underlyings` and catches a
    malformed request or an unrecognised product_type before it silently mispredicts.
    """
    expected = rfqs.groupby("product_type").basket_size.agg(lambda s: s.mode().iloc[0])
    mismatched = rfqs[rfqs.basket_size != rfqs.product_type.map(expected)]
    if not mismatched.empty:
        raise ValueError(
            f"{len(mismatched)} rows have basket_size inconsistent with product_type: "
            f"{mismatched.rfq_id.tolist()[:10]}"
        )


def normalise_observation_frequency(rfqs: pd.DataFrame) -> pd.DataFrame:
    rfqs = rfqs.copy()
    rfqs["obs_freq_months"] = rfqs.observation_frequency.str.strip().str.lower().map(FREQ_MONTHS)
    unmapped = rfqs.obs_freq_months.isna()
    if unmapped.any():
        raise ValueError(
            f"unmapped observation_frequency labels: "
            f"{sorted(rfqs.loc[unmapped, 'observation_frequency'].unique())}"
        )
    return rfqs


def join_volatility(rfqs: pd.DataFrame, vol: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """As-of join of realised and structural volatility onto each RFQ.

    rv_mean/rv_max/rv_min/rv_spread summarise the basket's legs as of
    `requested_date`; sb_mean is the static per-ticker reference level;
    mkt_vol is the cross-sectional daily mean realised vol across all 14
    tickers, joined as-of the same date.
    """
    legs = (
        rfqs.assign(underlying=rfqs.underlyings.str.split("|"))[["rfq_id", "requested_date", "underlying"]]
        .explode("underlying")
    )

    legs_vol = pd.merge_asof(
        legs.sort_values("requested_date"),
        vol[["date", "underlying", "realized_vol_63d"]].sort_values("date"),
        left_on="requested_date", right_on="date", by="underlying", direction="backward",
    ).merge(ref[["underlying", "structural_base_vol"]], on="underlying", how="left")

    vol_agg = legs_vol.groupby("rfq_id").agg(
        rv_mean=("realized_vol_63d", "mean"),
        rv_max=("realized_vol_63d", "max"),
        rv_min=("realized_vol_63d", "min"),
        sb_mean=("structural_base_vol", "mean"),
    )
    vol_agg["rv_spread"] = vol_agg.rv_max - vol_agg.rv_min
    vol_agg["vol_regime_ratio"] = vol_agg.rv_mean / vol_agg.sb_mean

    market_daily = (
        vol.groupby("date", as_index=False).realized_vol_63d.mean()
        .rename(columns={"realized_vol_63d": "mkt_vol"}).sort_values("date")
    )
    mkt_vol = pd.merge_asof(
        rfqs[["rfq_id", "requested_date"]].sort_values("requested_date"), market_daily,
        left_on="requested_date", right_on="date", direction="backward",
    )[["rfq_id", "mkt_vol"]]

    return rfqs.merge(vol_agg, on="rfq_id", how="left").merge(mkt_vol, on="rfq_id", how="left")


def filter_modelling_universe(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Executed RFQs with a labelled target.

    Also drops the 303 rows where avg_duration_months exceeds the nominal
    tenor implied by end_date - start_date.
    """
    lab = rfqs[rfqs.executed.astype(bool) & rfqs[TARGET].notna()].copy()
    return lab.loc[lab[TARGET] <= lab.tenor_months]


def assemble(raw_dir: Path) -> pd.DataFrame:
    """The filtered, joined modelling universe.
    `split.py` needs the dates to build the CV folds, so this
    stops one step short of the model matrix `to_model_matrix` produces.
    """
    rfqs, vol, ref = load_raw(raw_dir)
    rfqs = add_derived_columns(rfqs)
    assert_basket_size(rfqs)
    rfqs = normalise_observation_frequency(rfqs)
    rfqs = join_volatility(rfqs, vol, ref)
    rfqs = filter_modelling_universe(rfqs)
    return rfqs.sort_values("requested_date").reset_index(drop=True)


def to_model_matrix(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers and one-hot encode the categorical, in row order."""
    features = rfqs[FEATURE_COLUMNS + [TARGET]]
    return pd.get_dummies(features, columns=["product_type"])


def build_features(raw_dir: Path) -> pd.DataFrame:
    return to_model_matrix(assemble(raw_dir))


def existing_versions(base_dir: Path) -> list[int]:
    """Version numbers already present as `v1/`, `v2/`, ... under `base_dir`."""
    if not base_dir.exists():
        return []
    return sorted(
        int(p.name[1:]) for p in base_dir.iterdir()
        if p.is_dir() and p.name[1:].isdigit() and p.name.startswith("v")
    )


def next_version(base_dir: Path) -> int:
    """The next unused version number, for writing a new dataset version."""
    versions = existing_versions(base_dir)
    return versions[-1] + 1 if versions else 1


def latest_version(base_dir: Path) -> int:
    """The most recent existing version number, for reading a dataset that
    preprocess.py has already produced (e.g. from split.py or training).
    """
    versions = existing_versions(base_dir)
    if not versions:
        raise FileNotFoundError(f"no versioned dataset found under {base_dir}, run preprocess.py first")
    return versions[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--version", type=int, default=None,
        help="Dataset version to write, e.g. 3 for v3/ (default: next unused version)",
    )
    args = parser.parse_args()

    version = args.version or next_version(args.out_dir)
    out_dir = args.out_dir / f"v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(args.raw_dir)
    features.to_csv(out_dir / "features.csv", index=False)
    print(f"{out_dir / 'features.csv'}: {len(features):,} rows, {features.shape[1]} columns")


if __name__ == "__main__":
    main()
