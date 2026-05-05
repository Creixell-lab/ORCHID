"""General QR-derived ORCHID/Helmert-style ElasticNet regression CLI.

For an arbitrary alphabet size ``K`` and number of positions ``n``, this
script:

* Constructs the one-site redundant ORCHID interpretable matrix
  ``H_redundant`` (``(K+1) x K``) and its reduced square form ``H'``
  (``K x K``).
* Runs QR on ``H'`` and applies a deterministic sign convention to obtain
  the Euclidean-orthonormal rows ``U``.  Rescales to averaged-inner-product
  orthonormality (``Phi @ Phi.T / K == I``) to get ``Phi``.
* Builds the one-site coefficient-extraction matrix
  ``G_tilde = Phi / K``, the one-site inverse/design matrix
  ``V = G_tilde^-1``, and the redundant original ORCHID extractor
  ``G_original``.
* Lifts each of these to ``n``-site form by Kronecker power, using
  ``T_full = T_one_site ^kron(n)`` where ``T = G_original @ V``.
* Fits ElasticNet models for each requested epistasis order and writes:
    * ``predictions_by_order.csv``           -- in-sample and OOF predictions
    * ``epsilon_tilde_by_order.csv``         -- coefficients in the
      QR-derived ORCHID basis
    * ``epsilon_original_redundant_by_order.csv`` -- intuitive redundant
      original-ORCHID coefficients (``T_order @ beta_tilde``)
    * ``r2_by_order.csv``                    -- summary R^2 per order
    * ``cv_folds_by_order.csv``              -- per-fold detail
    * five basis-diagnostic CSVs

CPU parallelism is across epistasis orders, controlled by ``-j/--n-jobs``.
``-n`` is intentionally NOT used as the cores flag: ``--n`` already means
"number of sequence positions" elsewhere in this package.

This full Kronecker construction is well suited to small / mid-sized CDMS
alphabets (typical PIN1-style libraries with ``K = 3``, ``K = 4`` and
``n <= 8``, but also up to ``K = 20`` for short positions).  For very
large ``K`` or ``n`` the ``K**n x K**n`` design matrix becomes infeasible;
NumPy will raise ``MemoryError`` naturally at allocation time.  An
optional safety cap is provided via ``--max-genotype-space`` (default
``20**6 = 64_000_000``); raise it explicitly if your machine can host a
larger ``V_full``.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Basis construction
# ============================================================

def make_orchid_raw_H(K: int) -> np.ndarray:
    """Redundant one-site ORCHID-style matrix of shape ``(K + 1, K)``.

    Row 0 is the mean direction ``[1, 1, ..., 1]``.
    Rows 1..K are symmetric one-vs-rest contrast rows.

    For ``K = 3`` this returns::

        [ 1,  1,  1]
        [ 2, -1, -1]
        [-1,  2, -1]
        [-1, -1,  2]
    """
    mean_row = np.ones((1, K), dtype=float)
    contrasts = K * np.eye(K, dtype=float) - np.ones((K, K), dtype=float)
    return np.vstack([mean_row, contrasts])


def make_orthonormal_G_one_site(K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the QR-derived one-site orthonormal ORCHID transform.

    Returns
    -------
    G_tilde : np.ndarray
        Coefficient-extraction matrix.  ``eps_tilde = G_tilde @ y``.
    Phi : np.ndarray
        Averaged-inner-product orthonormal basis: ``Phi @ Phi.T / K == I``.
    V : np.ndarray
        Inverse / reconstruction / design matrix: ``y = V @ eps_tilde``.
    H_prime : np.ndarray
        Reduced interpretable starting matrix
        (mean row + first ``K - 1`` contrast rows).
    """
    if K < 2:
        raise ValueError("K must be >= 2.")

    H_redundant = make_orchid_raw_H(K)

    # Keep mean + first K-1 contrast rows.  This removes the one linearly
    # dependent contrast row (the K state contrasts sum to zero).
    H_prime = H_redundant[:K, :]

    # QR expects basis vectors as columns, so transpose.
    Q, _ = np.linalg.qr(H_prime.T)

    # Rows of U are Euclidean-orthonormal basis vectors.
    U = Q.T

    # Deterministic sign convention: align each QR-derived row with the
    # original interpretable row.
    signs = np.sign(np.sum(U * H_prime, axis=1, keepdims=True))
    signs = np.where(signs == 0, 1.0, signs)
    U = U * signs

    # Convert Euclidean orthonormality to averaged-inner-product
    # orthonormality.
    Phi = np.sqrt(K) * U

    # Coefficient-extraction matrix under the averaged inner product.
    G_tilde = Phi / K

    # Inverse / design matrix.
    V = np.linalg.inv(G_tilde)

    return G_tilde, Phi, V, H_prime


def make_original_orchid_G_one_site(K: int) -> np.ndarray:
    """Original redundant ORCHID-style coefficient-extraction matrix.

    Shape ``(K + 1, K)``.

    For ``K = 3`` this corresponds to::

        diag(1/3, 1/2, 1/2, 1/2)
            @
        [[ 1,  1,  1],
         [ 2, -1, -1],
         [-1,  2, -1],
         [-1, -1,  2]]

    giving one mean coefficient and ``K`` redundant symmetric
    state-specific contrast coefficients.
    """
    H_raw = make_orchid_raw_H(K)
    scales = np.array(
        [1.0 / K] + [1.0 / (K - 1)] * K,
        dtype=float,
    )
    return np.diag(scales) @ H_raw


def make_tilde_to_original_one_site(K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-site map from ``eps_tilde`` to redundant original ORCHID epsilons.

    Returns ``(T, G_tilde, G_original)`` where
    ``eps_original_redundant = T @ eps_tilde`` and ``T`` has shape
    ``(K + 1, K)``.
    """
    G_tilde, _, _, _ = make_orthonormal_G_one_site(K)
    G_original = make_original_orchid_G_one_site(K)
    T = G_original @ np.linalg.inv(G_tilde)
    return T, G_tilde, G_original


def kron_power(M: np.ndarray, n: int) -> np.ndarray:
    """Kronecker power ``M^{otimes n}``."""
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(out, M)
    return out


def tilde_coefficient_orders(K: int, n: int) -> np.ndarray:
    """Interaction order for each tilde basis coefficient.

    One-site basis index 0 = mean; one-site basis indices ``1..K-1`` =
    contrasts.  The order of an n-site coefficient is the number of
    contrasted positions.
    """
    return np.array(
        [sum(idx != 0 for idx in combo) for combo in itertools.product(range(K), repeat=n)],
        dtype=int,
    )


def original_coefficient_orders(K: int, n: int) -> np.ndarray:
    """Interaction order for each redundant original ORCHID coefficient.

    One-site basis index 0 = mean; one-site basis indices ``1..K`` =
    state-specific redundant contrasts.
    """
    return np.array(
        [sum(idx != 0 for idx in combo) for combo in itertools.product(range(K + 1), repeat=n)],
        dtype=int,
    )


def build_term_table_tilde(K: int, n: int, alphabet: list[str]) -> pd.DataFrame:
    rows = []
    for idx, combo in enumerate(itertools.product(range(K), repeat=n)):
        order = sum(x != 0 for x in combo)
        pieces = []
        for pos, x in enumerate(combo, start=1):
            if x == 0:
                continue
            pieces.append(f"pos{pos}:tilde_c{x - 1}")
        rows.append(
            {
                "tilde_index": idx,
                "order": order,
                "basis_combo": "|".join(map(str, combo)),
                "term": "mean" if not pieces else ";".join(pieces),
            }
        )
    return pd.DataFrame(rows)


def build_term_table_original(K: int, n: int, alphabet: list[str]) -> pd.DataFrame:
    rows = []
    for idx, combo in enumerate(itertools.product(range(K + 1), repeat=n)):
        order = sum(x != 0 for x in combo)
        pieces = []
        for pos, x in enumerate(combo, start=1):
            if x == 0:
                continue
            state = alphabet[x - 1] if x - 1 < len(alphabet) else f"state{x - 1}"
            pieces.append(f"pos{pos}:eps_{state}")
        rows.append(
            {
                "original_index": idx,
                "order": order,
                "basis_combo": "|".join(map(str, combo)),
                "term": "mean" if not pieces else ";".join(pieces),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Encoding and design matrix
# ============================================================

def parse_alphabet(
    alphabet_arg: str | None,
    df: pd.DataFrame,
    variant_col: str,
    n: int,
    k: int,
) -> list[str]:
    if alphabet_arg:
        alphabet = [x.strip() for x in alphabet_arg.split(",") if x.strip()]
    else:
        chars = sorted(set("".join(df[variant_col].astype(str).tolist())))
        alphabet = chars

    if len(alphabet) != k:
        raise ValueError(
            f"Alphabet length must equal k. Got alphabet={alphabet} with length "
            f"{len(alphabet)}, but k={k}."
        )
    return alphabet


def encode_variants(
    df: pd.DataFrame,
    variant_col: str,
    alphabet: list[str],
    n: int,
) -> np.ndarray:
    """Encode variants into integer state indices.

    Assumes each variant is a string of length ``n`` whose characters are
    in ``alphabet`` and that the alphabet is shared across all positions.
    """
    alphabet_map = {a: i for i, a in enumerate(alphabet)}
    seqs = df[variant_col].astype(str).tolist()
    bad_lengths = [s for s in seqs if len(s) != n]
    if bad_lengths:
        raise ValueError(
            f"All variants must have length n={n}. Example bad variant: {bad_lengths[0]!r}"
        )
    X_idx = np.zeros((len(seqs), n), dtype=int)
    for row, s in enumerate(seqs):
        for pos, ch in enumerate(s):
            if ch not in alphabet_map:
                raise ValueError(
                    f"Character {ch!r} in variant {s!r} is not in alphabet {alphabet}."
                )
            X_idx[row, pos] = alphabet_map[ch]
    return X_idx


def genotype_row_indices(seq_idx: np.ndarray, K: int) -> np.ndarray:
    """Map genotype state indices to row indices in the Kronecker order
    that matches ``itertools.product(range(K), repeat=n)``."""
    n = seq_idx.shape[1]
    powers = (K ** np.arange(n - 1, -1, -1)).astype(int)
    return (seq_idx * powers).sum(axis=1)


def build_design_for_order(
    V_full: np.ndarray,
    row_idx: np.ndarray,
    orders: np.ndarray,
    max_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix for coefficients up to ``max_order`` (intercept kept
    as column 0).
    """
    keep = np.where(orders <= max_order)[0]
    X = V_full[row_idx][:, keep]
    return X, keep


# ============================================================
# Regression helpers
# ============================================================

@dataclass
class OrderFitResult:
    order: int
    predictions_full: np.ndarray
    predictions_oof: np.ndarray
    beta_tilde: np.ndarray
    kept_tilde_indices: np.ndarray
    epsilon_tilde_df: pd.DataFrame
    epsilon_original_df: pd.DataFrame
    r2_row: dict
    fold_rows: list[dict]


def make_elasticnet_pipeline(
    alphas: np.ndarray,
    l1_ratios: list[float],
    cv: KFold,
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "encv",
                ElasticNetCV(
                    l1_ratio=l1_ratios,
                    alphas=alphas,
                    cv=cv,
                    fit_intercept=True,
                    max_iter=max_iter,
                    n_jobs=1,
                ),
            ),
        ]
    )


def extract_unscaled_coefficients(model: Pipeline) -> tuple[float, np.ndarray]:
    """Reverse ``StandardScaler`` so coefficients come back in the V-basis
    units, not z-scored units.

    For ``x_scaled = (x - mean) / scale`` and the fitted relation
    ``y = intercept_scaled + sum(coef_scaled * x_scaled)``, the unscaled
    equivalents are::

        coef_raw      = coef_scaled / scale
        intercept_raw = intercept_scaled - sum(coef_scaled * mean / scale)
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    encv: ElasticNetCV = model.named_steps["encv"]

    coef_scaled = encv.coef_.astype(float)
    scale = scaler.scale_.astype(float)
    mean = scaler.mean_.astype(float)

    coef_raw = coef_scaled / scale
    intercept_raw = float(encv.intercept_ - np.sum(coef_scaled * mean / scale))
    return intercept_raw, coef_raw


def fit_one_order(
    order: int,
    X_with_intercept: np.ndarray,
    kept_tilde_indices: np.ndarray,
    y: np.ndarray,
    variant_ids: np.ndarray,
    tilde_terms: pd.DataFrame,
    original_terms: pd.DataFrame,
    T_full: np.ndarray,
    original_orders: np.ndarray,
    cv_folds: int,
    cv_repeats: int,
    random_seed: int,
    alphas: np.ndarray,
    l1_ratios: list[float],
    max_iter: int,
) -> OrderFitResult:
    """Fit a single epistasis order.

    ``X_with_intercept`` includes the constant mean-basis column as
    column 0.  We drop it before passing to sklearn and let
    ``fit_intercept=True`` handle the constant.
    """
    if X_with_intercept.shape[1] < 1:
        raise ValueError("Design matrix must include at least the intercept column.")

    X = X_with_intercept[:, 1:]
    n_samples = len(y)

    oof_sum = np.zeros(n_samples, dtype=float)
    oof_count = np.zeros(n_samples, dtype=float)
    fold_rows: list[dict] = []

    for rep in range(cv_repeats):
        outer = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_seed + rep,
        )
        for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
            inner = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_seed + 10_000 + rep,
            )
            model = make_elasticnet_pipeline(
                alphas=alphas,
                l1_ratios=l1_ratios,
                cv=inner,
                max_iter=max_iter,
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            oof_sum[test_idx] += pred
            oof_count[test_idx] += 1.0
            fold_rows.append(
                {
                    "order": order,
                    "repeat": rep,
                    "fold": fold,
                    "r2": r2_score(y[test_idx], pred),
                    "alpha": float(model.named_steps["encv"].alpha_),
                    "l1_ratio": float(model.named_steps["encv"].l1_ratio_),
                    "n_features": X.shape[1],
                }
            )

    predictions_oof = oof_sum / np.maximum(oof_count, 1.0)

    final_inner = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed + 999_999,
    )
    final_model = make_elasticnet_pipeline(
        alphas=alphas,
        l1_ratios=l1_ratios,
        cv=final_inner,
        max_iter=max_iter,
    )
    final_model.fit(X, y)
    predictions_full = final_model.predict(X)

    intercept_raw, coef_raw = extract_unscaled_coefficients(final_model)

    beta_tilde = np.zeros(len(kept_tilde_indices), dtype=float)
    beta_tilde[0] = intercept_raw
    beta_tilde[1:] = coef_raw

    # Tilde epsilon output
    tilde_subset = tilde_terms.loc[kept_tilde_indices].copy()
    tilde_subset.insert(0, "fit_order", order)
    tilde_subset["epsilon_tilde"] = beta_tilde

    # Convert tilde epsilons to intuitive redundant original ORCHID epsilons.
    original_keep = np.where(original_orders <= order)[0]
    T_order = T_full[np.ix_(original_keep, kept_tilde_indices)]
    beta_original = T_order @ beta_tilde

    original_subset = original_terms.loc[original_keep].copy()
    original_subset.insert(0, "fit_order", order)
    original_subset["epsilon_original"] = beta_original

    fold_r2s = np.array(
        [r["r2"] for r in fold_rows if r["order"] == order],
        dtype=float,
    )
    r2_row = {
        "order": order,
        "n_samples": n_samples,
        "n_features_without_intercept": X.shape[1],
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "cv_r2_std": float(np.std(fold_r2s, ddof=0)),
        "oof_r2": float(r2_score(y, predictions_oof)),
        "full_fit_r2": float(r2_score(y, predictions_full)),
        "final_alpha": float(final_model.named_steps["encv"].alpha_),
        "final_l1_ratio": float(final_model.named_steps["encv"].l1_ratio_),
    }

    return OrderFitResult(
        order=order,
        predictions_full=predictions_full,
        predictions_oof=predictions_oof,
        beta_tilde=beta_tilde,
        kept_tilde_indices=kept_tilde_indices,
        epsilon_tilde_df=tilde_subset,
        epsilon_original_df=original_subset,
        r2_row=r2_row,
        fold_rows=fold_rows,
    )


# ============================================================
# CLI
# ============================================================

DEFAULT_MAX_GENOTYPE_SPACE: int = 64_000_000  # 20**6, big enough for any PIN-style library


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchid-regression-general",
        description=(
            "General QR-derived ORCHID/Helmert-style ElasticNet regression. "
            "Automatically builds G by QR decomposition, obtains V = G^-1, "
            "fits ElasticNet models per epistasis order, and outputs "
            "predictions, epsilon_tilde (in the QR-derived basis), the "
            "intuitive redundant original-ORCHID epsilons, and R^2 values. "
            "Use -j / --n-jobs for CPU parallelism across orders. "
            "(-n is reserved for sequence positions and is NOT a cores flag.)"
        ),
    )
    p.add_argument("--input", required=True, help="Input CSV file.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--variant-col", required=True, help="Column containing encoded variants.")
    p.add_argument("--phenotype-col", required=True, help="Phenotype column.")
    p.add_argument("--n", type=int, required=True, help="Number of variable positions.")
    p.add_argument("--k", type=int, required=True, help="Number of states per position.")
    p.add_argument("--max-order", type=int, required=True, help="Maximum epistasis order.")
    p.add_argument(
        "--alphabet",
        default=None,
        help="Comma-separated alphabet, e.g. 'a,b,c' or '0,1,2'. If omitted, inferred from data.",
    )
    p.add_argument(
        "--orders",
        default=None,
        help=(
            "Comma-separated model orders to fit, e.g. '1,2,3'. "
            "Default: all orders from 1 to --max-order."
        ),
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=1)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument(
        "--alphas",
        default="-4,2,25",
        help=(
            "ElasticNet alpha grid as 'log10_start,log10_stop,num'. "
            "Default '-4,2,25' is np.logspace(-4, 2, 25)."
        ),
    )
    p.add_argument(
        "--l1-ratios",
        default="0.05,0.1,0.2,0.5,0.8,0.95",
        help="Comma-separated ElasticNet l1_ratio values.",
    )
    p.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel jobs across epistasis orders. "
            "Default 1 (sequential). Use -j 4 for 4 CPU cores."
        ),
    )
    p.add_argument(
        "--max-genotype-space",
        type=int,
        default=DEFAULT_MAX_GENOTYPE_SPACE,
        help=(
            "Refuse to build the K**n x K**n V matrix when K**n exceeds this "
            f"value (default {DEFAULT_MAX_GENOTYPE_SPACE:,}, = 20**6). Raise "
            "explicitly for a bigger run; NumPy will still raise MemoryError "
            "naturally when V_full really is too large for your machine."
        ),
    )
    return p


def parse_alpha_grid(alpha_arg: str) -> np.ndarray:
    parts = [float(x) for x in alpha_arg.split(",")]
    if len(parts) != 3:
        raise ValueError("--alphas must be 'log10_start,log10_stop,num', e.g. '-4,2,25'.")
    start, stop, num = parts
    return np.logspace(start, stop, int(num))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Pre-flight memory sanity check.  V_full is K**n x K**n, so the
    # bound on K**n directly bounds the matrix dimension; the actual
    # memory cost is K**(2n) entries.  NumPy will raise MemoryError on
    # its own when V_full really cannot be allocated; this guard is just
    # a clearer error message at the very largest scales.
    genotype_space = args.k ** args.n
    if genotype_space > args.max_genotype_space:
        raise ValueError(
            f"K**n = {args.k}**{args.n} = {genotype_space:,} exceeds "
            f"--max-genotype-space ({args.max_genotype_space:,}). "
            f"The Kronecker design matrix would have {genotype_space ** 2:,} "
            "entries. Raise `--max-genotype-space` if you really do want a "
            "larger run, or use `orchid-epistasis-pba` (the WH-PBA pipeline) "
            "for large alphabets where the full Kronecker matrix is "
            "intractable."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.variant_col not in df.columns:
        raise ValueError(f"Variant column {args.variant_col!r} not found.")
    if args.phenotype_col not in df.columns:
        raise ValueError(f"Phenotype column {args.phenotype_col!r} not found.")

    df = df.reset_index(drop=True).copy()
    y = df[args.phenotype_col].astype(float).to_numpy()

    alphabet = parse_alphabet(args.alphabet, df, args.variant_col, args.n, args.k)
    seq_idx = encode_variants(df, args.variant_col, alphabet, args.n)
    row_idx = genotype_row_indices(seq_idx, args.k)

    if args.orders is None:
        orders_to_fit = list(range(1, args.max_order + 1))
    else:
        orders_to_fit = [int(x.strip()) for x in args.orders.split(",") if x.strip()]
    if max(orders_to_fit) > args.max_order:
        raise ValueError("--orders cannot contain values greater than --max-order.")

    alphas = parse_alpha_grid(args.alphas)
    l1_ratios = [float(x) for x in args.l1_ratios.split(",") if x.strip()]

    print("Building one-site QR-derived ORCHID basis...")
    G1, Phi1, V1, H1_prime = make_orthonormal_G_one_site(args.k)
    T1_full, _, G1_original_full = make_tilde_to_original_one_site(args.k)

    print("Building n-site Kronecker matrices...")
    V_full = kron_power(V1, args.n)
    T_full = kron_power(T1_full, args.n)
    tilde_orders = tilde_coefficient_orders(args.k, args.n)
    original_orders = original_coefficient_orders(args.k, args.n)
    tilde_terms = build_term_table_tilde(args.k, args.n, alphabet)
    original_terms = build_term_table_original(args.k, args.n, alphabet)

    # Save basis diagnostics
    pd.DataFrame(G1).to_csv(outdir / "G_one_site_tilde.csv", index=False)
    pd.DataFrame(V1).to_csv(outdir / "V_one_site_inverse_design.csv", index=False)
    pd.DataFrame(Phi1).to_csv(outdir / "Phi_one_site_orthonormal_basis.csv", index=False)
    pd.DataFrame(H1_prime).to_csv(outdir / "H_one_site_reduced_starting_matrix.csv", index=False)
    pd.DataFrame(G1_original_full).to_csv(outdir / "G_one_site_original_redundant.csv", index=False)
    pd.DataFrame(T1_full).to_csv(outdir / "T_one_site_tilde_to_original.csv", index=False)

    print(f"Input rows: {len(df)}")
    print(f"n={args.n}, k={args.k}, max_order={args.max_order}")
    print(f"Alphabet: {alphabet}")
    print(f"Orders to fit: {orders_to_fit}")
    print(f"V_full shape: {V_full.shape}")

    def _run_order(order: int) -> OrderFitResult:
        print(f"Fitting order {order}...")
        X_order, keep_tilde = build_design_for_order(
            V_full=V_full,
            row_idx=row_idx,
            orders=tilde_orders,
            max_order=order,
        )
        return fit_one_order(
            order=order,
            X_with_intercept=X_order,
            kept_tilde_indices=keep_tilde,
            y=y,
            variant_ids=df[args.variant_col].to_numpy(),
            tilde_terms=tilde_terms,
            original_terms=original_terms,
            T_full=T_full,
            original_orders=original_orders,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            random_seed=args.random_seed,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=args.max_iter,
        )

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_order)(order) for order in orders_to_fit
    )
    results = sorted(results, key=lambda r: r.order)

    # ========================================================
    # Write predictions
    # ========================================================
    pred_df = df[[args.variant_col, args.phenotype_col]].copy()
    for res in results:
        pred_df[f"pred_order_{res.order}"] = res.predictions_full
        pred_df[f"cv_pred_order_{res.order}"] = res.predictions_oof
    pred_df.to_csv(outdir / "predictions_by_order.csv", index=False)

    # ========================================================
    # Write epsilon_tilde coefficients
    # ========================================================
    eps_tilde_df = pd.concat(
        [res.epsilon_tilde_df for res in results],
        ignore_index=True,
    )
    eps_tilde_df.to_csv(outdir / "epsilon_tilde_by_order.csv", index=False)

    # ========================================================
    # Write intuitive original redundant ORCHID epsilons
    # ========================================================
    eps_original_df = pd.concat(
        [res.epsilon_original_df for res in results],
        ignore_index=True,
    )
    eps_original_df.to_csv(outdir / "epsilon_original_redundant_by_order.csv", index=False)

    # ========================================================
    # Write R^2 outputs
    # ========================================================
    r2_df = pd.DataFrame([res.r2_row for res in results])
    r2_df.to_csv(outdir / "r2_by_order.csv", index=False)

    folds_df = pd.DataFrame(
        [row for res in results for row in res.fold_rows]
    )
    folds_df.to_csv(outdir / "cv_folds_by_order.csv", index=False)

    print("\n--- R^2 by epistasis order ---")
    print(r2_df.to_string(index=False))
    print("\nWrote:")
    print(f"  {outdir / 'predictions_by_order.csv'}")
    print(f"  {outdir / 'epsilon_tilde_by_order.csv'}")
    print(f"  {outdir / 'epsilon_original_redundant_by_order.csv'}")
    print(f"  {outdir / 'r2_by_order.csv'}")
    print(f"  {outdir / 'cv_folds_by_order.csv'}")
    print(f"  {outdir / 'G_one_site_tilde.csv'}")
    print(f"  {outdir / 'V_one_site_inverse_design.csv'}")
    print(f"  {outdir / 'T_one_site_tilde_to_original.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
