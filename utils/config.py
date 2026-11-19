"""Config loading and project paths."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/default.yaml"):
    """Load a YAML config. Relative paths are resolved from the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as fh:
        return yaml.safe_load(fh)
