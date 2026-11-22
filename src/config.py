"""
src/config.py
=============
Lightweight config loader.  Reads configs/default.yaml and returns a
plain dict.  All pipeline scripts import this instead of using
hard-coded constants.

Usage
-----
from src.config import load_config
cfg = load_config()          # loads configs/default.yaml
cfg = load_config("configs/custom.yaml")   # override path
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML config as a plain nested dict.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML file.  Defaults to ``configs/default.yaml``
        relative to the repository root.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Make sure you are running from the repository root."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    return cfg


def get_seed(cfg: dict[str, Any]) -> int:
    """Return the global random seed from config (default 42)."""
    return int(cfg.get("seed", 42))


def get_test_size(cfg: dict[str, Any]) -> float:
    """Return the test split fraction from config (default 0.20)."""
    return float(cfg["data"].get("test_size", 0.20))
