"""CLI for ``orchid-wt-epistasis``.

Compute one wild-type-dependent (reference-based, variant-specific,
idiosyncratic) epistasis term between a starting reference sequence and
a destination sequence.  The order ``n`` of the term is the Hamming
distance between the two sequences (typically 2 for an "epistasis
square" or 3 for an "epistasis cube"; higher orders are also supported
when all ``2**n`` intermediate variants are present in the dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .wt_epistasis import compute_wt_epistasis, find_differing_positions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchid-wt-epistasis",
        description=(
            "Compute the WT-dependent (reference-based, variant-specific, "
            "idiosyncratic) epistasis term between a reference sequence and "
            "a destination sequence. The order n is the Hamming distance "
            "between them; the value is the standard finite-difference / "
            "inclusion-exclusion sum over all 2**n intermediate variants."
        ),
    )
    p.add_argument("--input", required=True, type=Path, help="Input CSV (or TSV).")
    p.add_argument(
        "--variant-col",
        required=True,
        help="Column with the encoded variant strings.",
    )
    p.add_argument(
        "--phenotype-col",
        required=True,
        help="Column with the phenotype values to use.",
    )
    p.add_argument(
        "--reference",
        required=True,
        help='Reference / starting sequence, e.g. "aaaaaa".',
    )
    p.add_argument(
        "--destination",
        required=True,
        help='Destination sequence, e.g. "bbcaaa".',
    )
    p.add_argument(
        "--max-order",
        type=int,
        default=None,
        help=(
            "Optional sanity guard: refuse to run when the Hamming "
            "distance exceeds this. Default: no upper limit."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for the per-variant inclusion-exclusion "
            "breakdown (variant, coefficient, phenotype, contribution)."
        ),
    )
    return p


def _read_csv_smart(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    df = _read_csv_smart(args.input)
    if args.variant_col not in df.columns:
        raise SystemExit(
            f"--variant-col {args.variant_col!r} not in dataset columns: "
            f"{df.columns.tolist()}"
        )
    if args.phenotype_col not in df.columns:
        raise SystemExit(
            f"--phenotype-col {args.phenotype_col!r} not in dataset columns: "
            f"{df.columns.tolist()}"
        )

    # Surface a clear error before the full computation if the inputs
    # are obviously incompatible.
    diff = find_differing_positions(args.reference, args.destination)
    if not diff:
        raise SystemExit(
            f"--reference and --destination are identical ({args.reference!r}); "
            "no epistasis term to compute."
        )

    result = compute_wt_epistasis(
        df,
        variant_col=args.variant_col,
        phenotype_col=args.phenotype_col,
        reference=args.reference,
        destination=args.destination,
        max_order=args.max_order,
    )

    print(f"Reference:        {result.reference}")
    print(f"Destination:      {result.destination}")
    print(f"Hamming distance: {result.order}")
    print(f"Differing positions (0-indexed): {result.differing_positions}")
    diff_pretty = ", ".join(
        f"pos{p}: {a!r} -> {b!r}"
        for p, (a, b) in zip(result.differing_positions, result.differing_states)
    )
    print(f"Mutations:        {diff_pretty}")
    print()

    breakdown = result.breakdown_dataframe()
    print("--- Inclusion-exclusion breakdown ---")
    print(breakdown.to_string(index=False))
    print()
    print(
        f"Epistasis^({result.order}) "
        f"({_human_label(result.order)}) = {result.epistasis:.6f}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        breakdown.to_csv(args.out, index=False)
        print(f"\nWrote breakdown CSV: {args.out}")

    return 0


def _human_label(order: int) -> str:
    if order == 1:
        return "single-mutation effect"
    if order == 2:
        return "epistasis square"
    if order == 3:
        return "epistasis cube"
    return f"order-{order} hypercube"


if __name__ == "__main__":
    raise SystemExit(main())
