"""Reference/market data backing feature engineering.

daily_volatility.csv and underlyings_reference.csv are held in memory and
loaded at startup, same pattern as the model in api/model.py:

  - RAW_DATA_DIR env var, read at startup (default: data/raw).
  - POST /reload also refreshes this alongside the model, so a new
    daily_volatility.csv drop doesn't need a service restart.
"""

import os
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", "data/raw"))


def resolve_raw_dir(override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("RAW_DATA_DIR")
    return Path(env) if env else RAW_DATA_DIR


class MarketData:
    """Holds the realised-vol and underlyings-reference tables used to engineer features."""

    def __init__(self):
        self.raw_dir: Path | None = None
        self.vol: pd.DataFrame | None = None
        self.ref: pd.DataFrame | None = None

    def load(self, raw_dir: Path):
        vol_path = raw_dir / "daily_volatility.csv"
        ref_path = raw_dir / "underlyings_reference.csv"
        if not vol_path.exists():
            raise FileNotFoundError(f"volatility file not found: {vol_path}")
        if not ref_path.exists():
            raise FileNotFoundError(f"reference file not found: {ref_path}")
        self.vol = pd.read_csv(vol_path, parse_dates=["date"])
        self.ref = pd.read_csv(ref_path)
        self.raw_dir = raw_dir


MARKET = MarketData()
