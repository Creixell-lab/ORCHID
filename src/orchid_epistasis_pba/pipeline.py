from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .io import add_position_columns, read_library_csv
from .model import epistasis_code, y_from_epi


@dataclass
class EpistasisRunConfig:
    input_path: str | Path
    outdir: str | Path
    variant_col: str
    phenotype_col: str
    n: int
    max_order: int
    k: int | None = None
    wildcard: str = "*"
    alphabet: Sequence[str] | None = None  # if None -> ["0","1",...,"k-1"]


def run_epistasis_pipeline(cfg: EpistasisRunConfig) -> dict[str, Path]:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_library_csv(cfg.input_path)
    df = add_position_columns(df, variant_col=cfg.variant_col, n=cfg.n)

    from .io import infer_alphabet_from_variant_col

    alphabet = list(cfg.alphabet) if cfg.alphabet is not None else infer_alphabet_from_variant_col(
        df, variant_col=cfg.variant_col, wildcard=cfg.wildcard
    )

    k = cfg.k if cfg.k is not None else len(alphabet)
    if k != len(alphabet):
        raise ValueError(f"k={k} but alphabet has {len(alphabet)} symbols")


    # E0 (mean)
    pheno = cfg.phenotype_col
    E0 = float(np.nanmean(df[pheno].to_numpy(dtype=float)))
    E0_df = pd.DataFrame({"Epi": ["*" * cfg.n], "E_partial": [E0]})

    # E1..Emax
    dfs = [E0_df]
    for e in range(1, cfg.max_order + 1):
        dfs.append(
            epistasis_code(k=cfg.k, n=cfg.n, e=e, df=df, pheno=pheno, alphabet=alphabet)
        )
    En_df = pd.concat(dfs, ignore_index=True)

    # predict
    pred = y_from_epi(En_df, df, variant_name=cfg.variant_col)
    pred_df = pd.DataFrame({cfg.variant_col: df[cfg.variant_col].astype(str), "y_pred": pred})

    terms_path = outdir / "epistasis_terms.csv"
    pred_path = outdir / "predicted.csv"

    En_df.to_csv(terms_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    return {"terms": terms_path, "predicted": pred_path}

