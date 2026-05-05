"""CLI for the linearisation candidate-fitting tool.

Fits a catalog of nonlinear link functions ``y = f(x; theta)`` to a
(observed phenotype, first-order prediction) pair from your dataset,
ranks them by R^2 / AIC / BIC, and writes per-method scatter+fit plots
plus a combined comparison plot.

Two ways to supply the prediction column:

* ``--predicted-col COL``           -- use ``COL`` straight from the input CSV
* (no ``--predicted-col``)          -- auto-fit a first-order ORCHID model
  on the input via the existing ``run_epistasis_pipeline`` and use its
  predictions.  Requires ``--variant-col``, ``--n`` and ``--alphabet``
  (or ``--k``).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .linearisation import (
    LINK_SPECS,
    best_method,
    fit_all_methods,
    plot_combined,
    plot_per_method,
    summary_dataframe,
)


# ---------------------------------------------------------------------------
# Auto-fit first-order predictions via the existing PBA pipeline
# ---------------------------------------------------------------------------

def _auto_fit_first_order(
    *,
    df: pd.DataFrame,
    input_path: Path,
    variant_col: str,
    phenotype_col: str,
    n: int,
    k: int | None,
    alphabet: list[str] | None,
) -> pd.DataFrame:
    """Run ``run_epistasis_pipeline(max_order=1)`` and return df with
    a new ``_first_order_pred`` column attached."""
    from .io import infer_alphabet_from_variant_col
    from .pipeline import EpistasisRunConfig, run_epistasis_pipeline

    if alphabet is None:
        alphabet = infer_alphabet_from_variant_col(df, variant_col=variant_col)
    if k is None:
        k = len(alphabet)
    if k != len(alphabet):
        raise SystemExit(f"--k {k} disagrees with alphabet length {len(alphabet)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = EpistasisRunConfig(
            input_path=input_path,
            outdir=tmp_path,
            variant_col=variant_col,
            phenotype_col=phenotype_col,
            n=n,
            max_order=1,
            k=k,
            alphabet=alphabet,
        )
        outs = run_epistasis_pipeline(cfg)
        pred_df = pd.read_csv(outs["predicted"])

    # `predicted.csv` is keyed on the variant column; merge it back.
    if variant_col not in pred_df.columns:
        raise RuntimeError(
            "run_epistasis_pipeline did not return the expected 'variant' "
            f"column: {pred_df.columns.tolist()}"
        )
    pred_df = pred_df.rename(columns={"y_pred": "_first_order_pred"})
    pred_df = pred_df[[variant_col, "_first_order_pred"]]
    df = df.merge(pred_df, on=variant_col, how="left", suffixes=("", "_dup"))
    if df["_first_order_pred"].isna().any():
        n_missing = int(df["_first_order_pred"].isna().sum())
        raise RuntimeError(
            f"Auto-fit produced no first-order prediction for {n_missing} rows "
            "(possibly duplicate variants). Provide --predicted-col explicitly."
        )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    method_names = ", ".join(LINK_SPECS.keys())
    p = argparse.ArgumentParser(
        prog="orchid-linearise",
        description=(
            "Fit a catalog of nonlinear link functions y = f(x) between an "
            "observed phenotype and a first-order ORCHID prediction. Reports "
            "R^2 / AIC / BIC for every method and saves per-method scatter+fit "
            "plots plus a combined comparison plot. If --predicted-col is not "
            "supplied, a first-order ORCHID model is auto-fit on --input."
        ),
    )
    p.add_argument("--input", required=True, type=Path, help="Input CSV (or TSV).")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    p.add_argument(
        "--observed-col",
        required=True,
        help="Column with the observed phenotype (the y values).",
    )
    p.add_argument(
        "--predicted-col",
        default=None,
        help=(
            "Column with the first-order / linear-motif prediction (the x values). "
            "If omitted, run the existing ORCHID first-order pipeline on the "
            "input to make one (requires --variant-col, --n, and --alphabet/--k)."
        ),
    )
    p.add_argument(
        "--methods",
        default=None,
        help=(
            f"Comma-separated subset of methods to fit. Default: all of {method_names}."
        ),
    )
    p.add_argument(
        "--criterion",
        default="r2",
        choices=("r2", "aic", "bic"),
        help="Which criterion picks the 'best' method in the report. Default r2.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip plotting.")

    auto = p.add_argument_group(
        "auto-fit first-order (only used when --predicted-col is omitted)"
    )
    auto.add_argument("--variant-col", default=None, help="Sequence/variant column.")
    auto.add_argument("--n", type=int, default=None, help="Sequence length.")
    auto.add_argument(
        "--k",
        type=int,
        default=None,
        help="Alphabet size. If omitted, inferred from --alphabet.",
    )
    auto.add_argument(
        "--alphabet",
        default=None,
        help="Comma-separated symbols, e.g. 'a,b,c'. If omitted, inferred from data.",
    )
    return p


def _read_csv_smart(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = _read_csv_smart(args.input).reset_index(drop=True)

    if args.observed_col not in df.columns:
        raise SystemExit(
            f"--observed-col {args.observed_col!r} not found. "
            f"Columns: {df.columns.tolist()}"
        )

    if args.predicted_col is None:
        if args.variant_col is None or args.n is None:
            raise SystemExit(
                "Either pass --predicted-col, or pass --variant-col and --n "
                "(and --alphabet / --k) to auto-fit a first-order model."
            )
        alphabet = (
            [s.strip() for s in args.alphabet.split(",") if s.strip()]
            if args.alphabet
            else None
        )
        print(
            f"[orchid-linearise] no --predicted-col given; auto-fitting "
            f"first-order ORCHID model (n={args.n}, "
            f"alphabet={alphabet}, k={args.k})..."
        )
        df = _auto_fit_first_order(
            df=df,
            input_path=args.input,
            variant_col=args.variant_col,
            phenotype_col=args.observed_col,
            n=args.n,
            k=args.k,
            alphabet=alphabet,
        )
        predicted_col = "_first_order_pred"
    else:
        if args.predicted_col not in df.columns:
            raise SystemExit(
                f"--predicted-col {args.predicted_col!r} not found. "
                f"Columns: {df.columns.tolist()}"
            )
        predicted_col = args.predicted_col

    x = df[predicted_col].astype(float).to_numpy()
    y = df[args.observed_col].astype(float).to_numpy()

    methods = (
        [s.strip() for s in args.methods.split(",") if s.strip()]
        if args.methods
        else None
    )

    print(f"[orchid-linearise] fitting {len(methods) if methods else len(LINK_SPECS)} methods on {len(x)} rows...")
    results = fit_all_methods(x, y, methods=methods)

    summary = summary_dataframe(results)
    summary_path = args.outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    df_out = df[[predicted_col, args.observed_col]].copy()
    df_out = df_out.rename(columns={predicted_col: "x_predicted", args.observed_col: "y_observed"})
    for name, r in results.items():
        if r.status == "ok" and r.transformed is not None:
            df_out[f"transformed_{name}"] = np.nan
            mask = np.isfinite(x) & np.isfinite(y)
            df_out.loc[mask, f"transformed_{name}"] = r.transformed
    df_out.to_csv(args.outdir / "transformed_predictions.csv", index=False)

    plots: dict[str, Path] = {}
    combined_path: Path | None = None
    if not args.no_plots:
        plots = plot_per_method(x, y, results, args.outdir / "plots")
        combined_path = plot_combined(x, y, results, args.outdir, "fit_comparison.png")

    try:
        chosen = best_method(results, criterion=args.criterion)
    except ValueError as exc:
        print(f"[orchid-linearise] WARNING: {exc}")
        chosen = ""
    best_path = args.outdir / "best_method.txt"
    best_path.write_text(
        f"criterion: {args.criterion}\n"
        f"best_method: {chosen}\n"
    )

    print()
    print("--- Linearisation summary ---")
    pd.set_option("display.max_colwidth", 80)
    print(summary.to_string(index=False))
    print()
    if chosen:
        r = results[chosen]
        print(
            f"Best by {args.criterion}: {chosen} "
            f"(R^2 = {r.r2:.6f}, AIC = {r.aic:.2f}, BIC = {r.bic:.2f}, "
            f"params = {r.param_dict})"
        )
    print()
    print("Wrote:")
    print(f"  summary.csv:               {summary_path}")
    print(f"  transformed_predictions.csv: {args.outdir / 'transformed_predictions.csv'}")
    print(f"  best_method.txt:           {best_path}")
    if combined_path is not None:
        print(f"  fit_comparison.png:        {combined_path}")
    if plots:
        print(f"  plots/fit_<method>.png:    {len(plots)} per-method plots in {args.outdir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
