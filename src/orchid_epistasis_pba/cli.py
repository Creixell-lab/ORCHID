from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import EpistasisRunConfig, run_epistasis_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchid-epistasis-pba")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--variant-col", required=True,  help="The column of the dataset with the genotype of interest")
    p.add_argument("--phenotype-col", required=True, help="The column of the dataset with the phenotype of interest: enrichment score etc")
    p.add_argument("--n", required=True, type=int, help="Number of positions in a protein of interest, note: massive N^k values are unfeasible")
    p.add_argument("--k", type=int, default=None, help="Alphabet size. If omitted, inferred from --alphabet or data.")
    p.add_argument("--max-order", default=1, type=int)

    p.add_argument(
        "--alphabet",
        default=None,
        help="Comma-separated symbols, e.g. '0,1,2' or 'a,b,c'. If omitted defaults to 0..k-1.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    alphabet = None
    if args.alphabet is not None:
        alphabet = [x.strip() for x in args.alphabet.split(",") if x.strip()]

    cfg = EpistasisRunConfig(
        input_path=args.input,
        outdir=args.outdir,
        variant_col=args.variant_col,
        phenotype_col=args.phenotype_col,
        n=args.n,
        k=args.k,
        max_order=args.max_order,
        alphabet=alphabet,
    )
    outs = run_epistasis_pipeline(cfg)
    print("Wrote:")
    for k, v in outs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

