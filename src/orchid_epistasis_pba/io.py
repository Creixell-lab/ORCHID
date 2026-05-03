from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_library_csv(path: str | Path, *, sep: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def add_position_columns(
    df: pd.DataFrame,
    *,
    variant_col: str,
    n: int,
    position_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Notebook equivalent of:
        for position in range(n):
            df[str(position)] = df[variant_column].str[position]
    """
    out = df.copy()
    if variant_col not in out.columns:
        raise ValueError(f"variant_col='{variant_col}' not in dataframe columns")

    if position_cols is None:
        position_cols = [str(i) for i in range(n)]
    if len(position_cols) != n:
        raise ValueError("position_cols length must equal n")

    s = out[variant_col].astype(str)
    for i, col in enumerate(position_cols):
        out[col] = s.str[i]

    return out


def infer_alphabet_from_variant_col(df: pd.DataFrame, *, variant_col: str, wildcard: str="*") -> list[str]:
    seen: list[str] = []
    for s in df[variant_col].astype(str):
        for ch in s:
            if ch == wildcard:
                continue
            if ch not in seen:
                seen.append(ch)
    if not seen:
        raise ValueError(f"Could not infer alphabet from column '{variant_col}'")
    return seen

