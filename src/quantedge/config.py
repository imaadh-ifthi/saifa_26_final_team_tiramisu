from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration and validate required sections."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"project", "data", "estimation", "portfolio", "benchmarks", "var", "outputs"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    return config
