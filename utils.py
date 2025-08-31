# utils.py
from pathlib import Path
import pandas as pd


def read_data(path: str | Path) -> pd.DataFrame:
    """Read input CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NA and add total column (price * qty)."""
    out = df.dropna().copy()
    if "price" in out.columns and "qty" in out.columns:
        out["total"] = out["price"].astype(float) * out["qty"].astype(float)
    return out


def write_data(df: pd.DataFrame, path: str | Path) -> None:
    """Write output CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def pipeline(in_path: str | Path, out_path: str | Path) -> None:
    """End-to-end pipeline: read → clean → write."""
    df = read_data(in_path)
    df = clean_data(df)
    write_data(df, out_path)
