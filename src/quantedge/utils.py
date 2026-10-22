from pathlib import Path
import logging
import numpy as np
import pandas as pd


def ensure_output_dirs(root: Path = Path(".")) -> None:
    for folder in ("data/raw", "data/processed", "outputs/tables", "outputs/charts", "outputs/models", "docs"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.dropna()
    return float(np.prod(1 + clean) ** (periods / len(clean)) - 1) if not clean.empty else float("nan")


def get_logger(name: str = "quantedge") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)
