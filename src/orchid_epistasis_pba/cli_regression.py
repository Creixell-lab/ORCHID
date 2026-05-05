"""
CLI entry points for the bundled k=3 regression benchmarks.

Two scripts are exposed in ``pyproject.toml``:

* ``orchid-epistasis-regression-benchmark``  -> :func:`main_regression_benchmark`
  uses :data:`regression.V_PHYS`

* ``wh-extension-regression``                -> :func:`main_wh_extension_regression`
  uses :data:`regression.V1_INV`

Both commands accept a single optional flag, ``-n/--n-jobs``, to control the
size of the joblib worker pool (default: ``-1``, i.e. all available cores).
Output (``results.csv``, ``raw_folds.csv``, ``r2_vs_fraction.png``) is written
into a named subdirectory of the current working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .regression import (
    INNER_FOLDS,
    MAX_ORDER,
    N_REPEATS,
    OUTER_FOLDS,
    V1_INV,
    V_PHYS,
    plot_benchmark,
    run_benchmark,
)


def _build_parser(prog: str, basis_label: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            f"{basis_label} ElasticNet regression benchmark on the bundled "
            "PIN1 36-variant CDMS library (n=6, k=3, max_order="
            f"{MAX_ORDER}). Inputs and hyperparameters are fixed; output "
            "is written to a fresh subdirectory of the current working "
            f"directory. Total compute is ~{N_REPEATS} repeats x "
            f"{OUTER_FOLDS} outer folds per truncation level."
        ),
    )
    p.add_argument(
        "-n",
        "--n-jobs",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Number of parallel worker processes for joblib. Default -1 "
            "uses every available CPU core. Pass a positive integer to "
            "cap the pool (e.g. `-n 4` to leave headroom on a laptop)."
        ),
    )
    return p


def _run(V: np.ndarray, name: str, outdir_name: str, *, n_jobs: int) -> int:
    outdir = Path.cwd() / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    cores_label = "all CPU cores" if n_jobs == -1 else f"{n_jobs} CPU core(s)"
    print(f"[{name}] running benchmark on bundled PIN1 36-variant library ({cores_label})")
    print(f"[{name}] writing results into {outdir}")
    res = run_benchmark(V, name, n_jobs=n_jobs)

    summary_path = outdir / "results.csv"
    raw_path = outdir / "raw_folds.csv"
    plot_path = outdir / "r2_vs_fraction.png"

    res.summary.to_csv(summary_path, index=False)
    res.raw.to_csv(raw_path, index=False)
    plot_benchmark(res.summary, name, plot_path)

    print()
    print(f"--- {name} summary ---")
    print(res.summary.to_string(index=False))
    print()
    print("Wrote:")
    print(f"  results.csv:        {summary_path}")
    print(f"  raw_folds.csv:      {raw_path}")
    print(f"  r2_vs_fraction.png: {plot_path}")
    return 0


def main_regression_benchmark(argv: list[str] | None = None) -> int:
    args = _build_parser(
        "orchid-epistasis-regression-benchmark",
        "V_phys (physics-correct simplex)",
    ).parse_args(argv)
    return _run(
        V_PHYS,
        "orchid-epistasis-regression-benchmark",
        "orchid_epistasis_regression_benchmark_output",
        n_jobs=args.n_jobs,
    )


def main_wh_extension_regression(argv: list[str] | None = None) -> int:
    args = _build_parser(
        "wh-extension-regression",
        "V1_inv (Walsh-Hadamard extension / direct Faure marginal contrasts)",
    ).parse_args(argv)
    return _run(
        V1_INV,
        "wh-extension-regression",
        "wh_extension_regression_output",
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main_regression_benchmark())
