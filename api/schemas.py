"""Pydantic request/response models for the inference API."""

from datetime import date

from pydantic import BaseModel, Field, field_validator


class RawRFQRequest(BaseModel):
    """Mirrors the data/raw/rfqs.csv columns needed to engineer features.

    /predict runs this through the same feature-engineering pipeline used
    for training (scripts/preprocess.py) rather than taking a
    pre-engineered feature vector, so callers submit an RFQ the way it
    would appear in the raw data, not the model's internal representation.
    Columns irrelevant to the model (basket_type, notional_credits,
    counterparty, trader_id, ...) are omitted.
    """

    product_type: str
    underlyings: str = Field(description="Pipe-separated tickers, e.g. 'CLNE|DRC'")
    autocall_barrier_pct: float = Field(gt=0)
    protection_barrier_pct: float = Field(gt=0)
    no_call_period_months: float = Field(ge=0)
    observation_frequency: str
    quoted_implied_vol: float = Field(gt=0)
    requested_date: date
    start_date: date
    end_date: date

    @field_validator("underlyings")
    @classmethod
    def _non_empty_underlyings(cls, v: str) -> str:
        v = v.strip()
        if not v or not all(v.split("|")):
            raise ValueError("underlyings must be a non-empty '|'-separated list of tickers")
        return v


class PredictionResponse(BaseModel):
    predicted_avg_duration_months: float
    model_path: str
    stale_volatility: bool
    warnings: list[str] = []


class ReloadRequest(BaseModel):
    model_path: str | None = None
    raw_dir: str | None = None
