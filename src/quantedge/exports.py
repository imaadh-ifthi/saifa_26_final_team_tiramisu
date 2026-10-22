from pathlib import Path
import json
import pandas as pd


def save_frame(frame: pd.DataFrame, path: Path, csv: bool = True, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if csv:
        frame.to_csv(path, index=frame.index.name is not None)
    if parquet:
        try:
            frame.to_parquet(path.with_suffix(".parquet"))
        except ImportError:
            pass


def save_selected_portfolio(path: Path, value: float, target: float, weights: pd.Series, expected_return: float, volatility: float, constraints: dict, estimation_window: dict, notes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"portfolio_value_usd": value, "target_return_annual": target, "weights": {k: float(v) for k, v in weights.items()}, "expected_return_annual": expected_return, "volatility_annual": volatility, "constraints": constraints, "estimation_window": estimation_window, "notes": notes}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
