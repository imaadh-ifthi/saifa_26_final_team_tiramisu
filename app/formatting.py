"""Consistent formatting utilities for currency, percentages, ratios, and booleans."""
from datetime import date, datetime
import numpy as np
import pandas as pd


def parse_boolean(value) -> bool:
    """Explicitly parse boolean values, preventing 'False' string from evaluating as True."""
    if isinstance(value, bool):
        return value
    if pd.isna(value) or value is None:
        return False
    s = str(value).strip().lower()
    if s in ("true", "1", "t", "yes", "y"):
        return True
    if s in ("false", "0", "f", "no", "n", "none", "nan", ""):
        return False
    return bool(value)


def format_currency(value: float | int | None, decimals: int = 0, include_sign: bool = False) -> str:
    """Format numeric value as USD currency."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "N/A"
    val = float(value)
    if abs(val) < 1e-9:
        val = 0.0
    if val < 0:
        abs_val = abs(val)
        if decimals == 0:
            return f"-${abs_val:,.0f}"
        return f"-${abs_val:,.{decimals}f}"
    prefix = "+" if include_sign and val > 0 else ""
    if decimals == 0:
        return f"{prefix}${val:,.0f}"
    return f"{prefix}${val:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 2, include_sign: bool = False) -> str:
    """Format numeric decimal value as percentage string."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "N/A"
    val = float(value)
    if abs(val) < 1e-9:
        val = 0.0
    pct = val * 100.0
    prefix = "+" if include_sign and pct > 0 else ""
    return f"{prefix}{pct:.{decimals}f}%"


def format_number(value: float | int | None, decimals: int = 2, include_sign: bool = False) -> str:
    """Format numeric ratio or count."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "N/A"
    val = float(value)
    if abs(val) < 1e-9:
        val = 0.0
    prefix = "+" if include_sign and val > 0 else ""
    return f"{prefix}{val:,.{decimals}f}"


def format_date(value) -> str:
    """Format date timestamp to YYYY-MM-DD."""
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    s = str(value).split("T")[0].split(" ")[0]
    return s
