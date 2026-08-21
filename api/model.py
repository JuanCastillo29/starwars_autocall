"""Loading and swapping the trained model.

The trained model is an independent, runtime-configurable parameter so pointing the
API at a different model is a config change, not a code change:

  - MODEL_PATH env var, read at startup (default: the latest
    data/processed/vN/model.joblib, via preprocess.latest_version).
  - POST /reload swaps the loaded model at runtime, either back to
    MODEL_PATH/the latest version, or to any other path passed in.
"""

import os
from pathlib import Path

import joblib

from scripts.preprocess import latest_version

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/processed"))


def default_model_path() -> Path:
    return DATA_DIR / f"v{latest_version(DATA_DIR)}" / "model.joblib"


def resolve_model_path(override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("MODEL_PATH")
    return Path(env) if env else default_model_path()


class LoadedModel:
    """Holds the currently loaded model so /reload can swap it in place."""

    def __init__(self):
        self.path: Path | None = None
        self.model = None
        self.feature_names: list[str] = []

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        model = joblib.load(path)
        self.model = model
        self.path = path
        self.feature_names = list(model.feature_name_)


MODEL = LoadedModel()
