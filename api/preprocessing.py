"""Turns a raw-shaped RFQ request into the feature row the model expects.

Runs the request through the same feature-engineering functions
scripts/preprocess.py uses for training (add_derived_columns,
normalise_observation_frequency, join_volatility), so /predict input
mirrors data/raw/rfqs.csv rather than a pre-engineered feature vector.

join_volatility does an as-of (backward) merge on realized_vol_63d, so a
requested_date past the end of daily_volatility.csv already falls back to
the most recent value available for that underlying, which is inherent to
an as-of join, not something we add here. What we add is *detecting* that
fallback and surfacing it as a warning, since a silent stale-vol fallback
would otherwise look like a normal, current prediction.
"""

import re

import pandas as pd
from fastapi import HTTPException

from api.market_data import MarketData
from api.model import MODEL
from api.schemas import RawRFQRequest
from scripts.preprocess import (
    FEATURE_COLUMNS,
    add_derived_columns,
    join_volatility,
    normalise_observation_frequency,
)


def _rfq_frame(request: RawRFQRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "rfq_id": "live-request",
        "product_type": request.product_type,
        "underlyings": request.underlyings,
        "autocall_barrier_pct": request.autocall_barrier_pct,
        "protection_barrier_pct": request.protection_barrier_pct,
        "no_call_period_months": request.no_call_period_months,
        "observation_frequency": request.observation_frequency,
        "quoted_implied_vol": request.quoted_implied_vol,
        "requested_date": pd.Timestamp(request.requested_date),
        "start_date": pd.Timestamp(request.start_date),
        "end_date": pd.Timestamp(request.end_date),
    }])


def _check_known_underlyings(underlyings: list[str], market: MarketData) -> None:
    known = set(market.vol.underlying.unique()) & set(market.ref.underlying.unique())
    unknown = sorted(set(underlyings) - known)
    if unknown:
        raise HTTPException(status_code=422, detail=f"no volatility/reference data for underlyings: {unknown}")


def _volatility_warnings(requested_date: pd.Timestamp, underlyings: list[str], market: MarketData) -> list[str]:
    """Flag any leg (or the market-wide series) whose as-of join fell back
    to a value older than requested_date, because no data at/after that
    date exists yet.
    """
    warnings = []
    last_by_underlying = market.vol.groupby("underlying").date.max()
    for u in underlyings:
        last = last_by_underlying[u]
        if requested_date > last:
            days_stale = (requested_date - last).days
            warnings.append(
                f"no realized_vol_63d for {u} on/after {last.date()}; "
                f"falling back to the last available value ({days_stale} days stale)"
            )
    market_last = market.vol.date.max()
    if requested_date > market_last:
        warnings.append(
            f"no market-wide volatility data on/after {market_last.date()}; "
            f"mkt_vol falls back to the last available value as of {market_last.date()}"
        )
    return warnings


def engineer_features(request: RawRFQRequest, market: MarketData) -> tuple[pd.Series, list[str]]:
    """Run the request through the training feature pipeline.

    Returns the engineered FEATURE_COLUMNS row plus any staleness warnings.
    """
    underlyings = request.underlyings.split("|")
    _check_known_underlyings(underlyings, market)

    rfqs = add_derived_columns(_rfq_frame(request))
    try:
        rfqs = normalise_observation_frequency(rfqs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    # merge_asof requires matching datetime dtypes; a request built from a
    # plain date doesn't necessarily land on the same unit pandas infers
    # when parsing daily_volatility.csv (e.g. datetime64[s] vs [us]).
    rfqs["requested_date"] = rfqs["requested_date"].astype(market.vol["date"].dtype)
    rfqs = join_volatility(rfqs, market.vol, market.ref)

    warnings = _volatility_warnings(rfqs.loc[0, "requested_date"], underlyings, market)
    return rfqs.loc[0, FEATURE_COLUMNS], warnings


def build_model_row(feature_row: pd.Series) -> pd.DataFrame:
    """Turn the engineered feature row into the exact one-hot row the loaded model expects.

    Columns come from model.feature_name_, not FEATURE_COLUMNS in
    preprocess.py, so a model trained with a different feature set than the
    one currently in preprocess.py still gets a correctly-shaped row (or a
    clear error) instead of a silent mismatch.
    """
    data = feature_row.to_dict()
    product_type = data.pop("product_type")

    product_cols = [c for c in MODEL.feature_names if c.startswith("product_type_")]
    numeric_cols = [c for c in MODEL.feature_names if c not in product_cols]

    missing = set(numeric_cols) - set(data)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"loaded model expects features the pipeline doesn't provide: {sorted(missing)}",
        )

    # LightGBM sanitizes feature names at training time, turning e.g.
    # "product_type_Mandalorian Twin-Win" into "..._Mandalorian_Twin-Win";
    # normalize the same way so a product_type matching the raw label
    # (as seen in data/raw/rfqs.csv) still resolves to the right column.
    product_col = f"product_type_{re.sub(r'\s+', '_', product_type.strip())}"
    if product_col not in product_cols:
        known = sorted(c.removeprefix("product_type_").replace("_", " ") for c in product_cols)
        raise HTTPException(
            status_code=422, detail=f"unknown product_type {product_type!r}; expected one of {known}"
        )

    row = {c: 0 for c in product_cols}
    row.update({c: data[c] for c in numeric_cols})
    row[product_col] = 1
    return pd.DataFrame([row], columns=MODEL.feature_names)
