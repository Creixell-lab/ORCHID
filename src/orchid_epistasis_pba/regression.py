"""
Nested-CV ElasticNet regression benchmarks for the bundled PIN1 36-variant
CDMS library (n = 6 positions, k = 3 alphabet, max-order = 3).

Two basis matrices are kept here, each exposed as a dedicated CLI command in
``cli_regression.py``:

* ``V_PHYS``  -- physics-correct simplex contrast matrix
                (entry point: ``orchid-epistasis-regression-benchmark``)
* ``V1_INV``  -- direct Fauré marginal contrasts / Walsh-Hadamard extension
                (entry point: ``WH-extension-regression``)

Inputs, columns, alphabet and hyperparameter grids are intentionally fixed:
this module is a reproducible benchmark on the bundled example, not a general
regression tool.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed configuration (matches the published reference benchmark)
# ---------------------------------------------------------------------------

TRUNCATION_LEVELS: tuple[float, ...] = (
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00,
)
N_REPEATS: int = 20
OUTER_FOLDS: int = 5
INNER_FOLDS: int = 5
MAX_ORDER: int = 3

ALPHA_GRID: np.ndarray = np.logspace(-4, 2, 25)
L1_RATIOS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 0.8, 0.95)

DATA_RESOURCE: str = "data/210825_PIN1_36_library.csv"
SEQ_COL: str = "pep_"
Y_COL: str = "PD_input_mean"
SORT_COL: str = "pep_encoded"
ALPHABET_SIZE: int = 3

# ---------------------------------------------------------------------------
# Basis matrices (k = 3)
# ---------------------------------------------------------------------------

# Direct Fauré / WH-extension marginal contrast matrix.
V1_INV: np.ndarray = np.array(
    [
        [1.0, -1.0 / 3.0, -1.0 / 3.0],
        [1.0,  2.0 / 3.0, -1.0 / 3.0],
        [1.0, -1.0 / 3.0,  2.0 / 3.0],
    ],
    dtype=float,
)

# Physics-correct equilateral simplex (centred, equal pairwise distances,
# unit-variance columns). Columns are: bias, axis-1 contrast, axis-2 contrast.
V_PHYS: np.ndarray = np.array(
    [
        [1.0,  np.sqrt(2.0),                0.0],
        [1.0, -np.sqrt(2.0) / 2.0,  np.sqrt(3.0 / 2.0)],
        [1.0, -np.sqrt(2.0) / 2.0, -np.sqrt(3.0 / 2.0)],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedLibrary:
    seq_idx: np.ndarray            # (N, L) int, position-wise alphabet index
    y: np.ndarray                  # (N,) float phenotype
    alphabet_per_site: list[list[str]]
    L: int


def load_example_library() -> LoadedLibrary:
    """Load the bundled PIN1 36-variant CSV from package resources."""
    csv_path = files("orchid_epistasis_pba").joinpath(DATA_RESOURCE)
    with csv_path.open("rb") as fh:
        df = pd.read_csv(fh)

    df = df.sort_values(SORT_COL).reset_index(drop=True)
    seqs = df[SEQ_COL].astype(str).values
    y = df[Y_COL].astype(float).values
    L = len(seqs[0])

    alphabet_per_site = [sorted({s[i] for s in seqs}) for i in range(L)]
    if any(len(a) != ALPHABET_SIZE for a in alphabet_per_site):
        raise ValueError(
            f"Bundled library is required to be k={ALPHABET_SIZE} per site; "
            f"got per-site sizes {[len(a) for a in alphabet_per_site]}."
        )

    seq_idx = np.zeros((len(seqs), L), dtype=int)
    for i, s in enumerate(seqs):
        for pos, ch in enumerate(s):
            seq_idx[i, pos] = alphabet_per_site[pos].index(ch)

    return LoadedLibrary(seq_idx=seq_idx, y=y, alphabet_per_site=alphabet_per_site, L=L)


# ---------------------------------------------------------------------------
# Kronecker design matrix builders
# ---------------------------------------------------------------------------

def build_design_matrix(V: np.ndarray, L: int, max_order: int = MAX_ORDER) -> np.ndarray:
    """L-site Kronecker product of V, restricted to columns of order <= max_order."""
    orders_1d = (0, 1, 1)
    col_orders = [sum(comb) for comb in itertools.product(orders_1d, repeat=L)]
    valid_cols = [i for i, o in enumerate(col_orders) if o <= max_order]

    V_full = V
    for _ in range(1, L):
        V_full = np.kron(V_full, V)
    return V_full[:, valid_cols]


def map_to_design(seq_idx: np.ndarray, X_full: np.ndarray, q: int = ALPHABET_SIZE) -> np.ndarray:
    """Index into the full design matrix using base-q decoding of position indices."""
    L = seq_idx.shape[1]
    powers = (q ** np.arange(L - 1, -1, -1)).astype(int)
    row_idx = (seq_idx * powers).sum(axis=1)
    return X_full[row_idx]


# ---------------------------------------------------------------------------
# Nested CV worker
# ---------------------------------------------------------------------------

def _evaluate_one(
    trunc: float,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
) -> list[tuple[float, int, int, float]]:
    """One (truncation, seed) draw -> per-fold R^2 rows."""
    n_samples = int(round(trunc * len(y)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_samples, replace=False)

    X_sub, y_sub = X[idx], y[idx]
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    rows: list[tuple[float, int, int, float]] = []

    for fold_i, (tr, te) in enumerate(outer.split(X_sub)):
        Xtr, Xte = X_sub[tr], X_sub[te]
        ytr, yte = y_sub[tr], y_sub[te]

        inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 1337)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "encv",
                    ElasticNetCV(
                        l1_ratio=list(L1_RATIOS),
                        alphas=ALPHA_GRID,
                        cv=inner,
                        fit_intercept=True,
                        max_iter=10_000,
                        # n_jobs=1: outer joblib pool already parallelises.
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(Xtr, ytr)
        rows.append((trunc, seed, fold_i, float(r2_score(yte, model.predict(Xte)))))

    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    summary: pd.DataFrame   # one row per (Frac, N): mean_R2, std_R2, n_folds
    raw: pd.DataFrame       # one row per (Frac, Seed, Fold): R2


def run_benchmark(
    V: np.ndarray,
    name: str,
    *,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Run the fixed nested-CV ElasticNet benchmark with basis matrix ``V``."""
    lib = load_example_library()
    X_full = build_design_matrix(V, lib.L, MAX_ORDER)
    # Drop the global intercept column; ElasticNetCV refits its own intercept.
    X = map_to_design(lib.seq_idx, X_full, q=ALPHABET_SIZE)[:, 1:]

    tasks = [
        (trunc, seed)
        for trunc in TRUNCATION_LEVELS
        for seed in range(N_REPEATS)
    ]

    iterator = (
        tqdm(tasks, desc=name, ncols=100) if show_progress else tasks
    )

    nested = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_one)(trunc, seed, X, lib.y) for trunc, seed in iterator
    )
    flat = [row for sub in nested for row in sub]

    raw_df = pd.DataFrame(flat, columns=["Frac", "Seed", "Fold", "R2"])
    raw_df["N"] = (raw_df["Frac"] * len(lib.y)).round().astype(int)

    summary = (
        raw_df.groupby(["Frac", "N"])["R2"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_R2", "std": "std_R2", "count": "n_folds"})
        .sort_values("Frac")
        .reset_index(drop=True)
    )
    return BenchmarkResult(summary=summary, raw=raw_df)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark(summary: pd.DataFrame, name: str, outpath: Path) -> None:
    """Render the standard 'R^2 vs training fraction' figure with error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(
        summary["Frac"].to_numpy() * 100.0,
        summary["mean_R2"].to_numpy(),
        yerr=summary["std_R2"].to_numpy(),
        fmt="o-",
        capsize=3,
        color="C0",
    )
    ax.set_xlabel("Training fraction of full library (%)")
    ax.set_ylabel("R\u00b2 on held-out outer fold")
    ax.set_title(
        f"{name}\nElasticNet nested CV "
        f"({N_REPEATS} repeats \u00d7 {OUTER_FOLDS}-fold outer / {INNER_FOLDS}-fold inner)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
