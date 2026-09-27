"""
Configuration loading utilities using OmegaConf.
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge override config into base config."""
    return OmegaConf.merge(base, override)


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to plain Python dict."""
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save config to YAML file."""
    OmegaConf.save(config, path)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """Safely get a nested config value using dot notation."""
    return OmegaConf.select(config, key, default=default)
