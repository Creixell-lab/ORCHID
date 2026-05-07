"""Wild-type-dependent (reference-based, variant-specific, idiosyncratic) epistasis.

Given a *reference* (starting) sequence and a *destination* sequence, this
module computes the n-th order WT-dependent epistasis term where ``n`` is
the Hamming distance between them.  This is the classical
finite-difference / inclusion-exclusion epistasis used throughout the
protein-engineering and DMS literature -- distinct from the
reference-free, ensemble-averaged ORCHID terms produced by
``orchid-epistasis-pba``.

The formula is the alternating sum::

    eps^(n) = sum over all subsets S of the n differing positions of:
        (-1)^(n - |S|) * y(variant whose differing positions in S
                           are set to destination, all other differing
                           positions left at reference)

Concretely:

* **Pairwise / "epistasis square"** (n = 2)::

      eps^(2) = y(AB) - y(A) - y(B) + y(WT)

* **Triplet / "epistasis cube"** (n = 3)::

      eps^(3) = y(ABC)
                - (y(AB) + y(AC) + y(BC))
                + (y(A)  + y(B)  + y(C))
                - y(WT)

For a Hamming distance of ``n``, the function looks up all ``2**n``
intermediate variants in the supplied dataframe; if any are missing it
raises ``KeyError`` (these terms are by definition not computable when
the corner of the n-cube is unmeasured).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WTEpistasisResult:
    """Container for one WT-dependent epistasis computation."""

    order: int
    reference: str
    destination: str
    differing_positions: list[int]
    differing_states: list[tuple[str, str]]  # (ref_state, dest_state) per diff position
    variants: list[str]                       # 2**n variants in inclusion-exclusion order
    coefficients: list[int]                   # +1 / -1 weights aligned with variants
    phenotypes: list[float]                   # phenotype lookups (averaged over duplicates)
    epistasis: float                          # the n-th order term

    def breakdown_dataframe(self) -> pd.DataFrame:
        """One row per intermediate variant, with its sign and contribution."""
        return pd.DataFrame(
            {
                "variant": self.variants,
                "coefficient": self.coefficients,
                "phenotype": self.phenotypes,
                "contribution": [c * y for c, y in zip(self.coefficients, self.phenotypes)],
            }
        )


def find_differing_positions(reference: str, destination: str) -> list[int]:
    """0-indexed positions at which ``reference`` and ``destination`` differ.

    Raises
    ------
    ValueError
        If the two strings have different lengths.
    """
    if len(reference) != len(destination):
        raise ValueError(
            f"reference (length {len(reference)}) and destination "
            f"(length {len(destination)}) must have the same length"
        )
    return [i for i, (a, b) in enumerate(zip(reference, destination)) if a != b]


def compute_wt_epistasis(
    df: pd.DataFrame,
    *,
    variant_col: str,
    phenotype_col: str,
    reference: str,
    destination: str,
    max_order: int | None = None,
    duplicate_aggregator: Callable[[np.ndarray], float] = np.mean,
) -> WTEpistasisResult:
    """Compute the n-th order WT-dependent epistasis term.

    Parameters
    ----------
    df
        Library data frame; must contain ``variant_col`` and ``phenotype_col``.
    variant_col, phenotype_col
        Column names.
    reference, destination
        Two sequences of equal length.  ``n`` (the order) is their
        Hamming distance.
    max_order
        Optional sanity guard; raises if the Hamming distance exceeds
        this.  ``None`` means no upper limit.
    duplicate_aggregator
        Callable applied to the phenotype values when a variant appears
        more than once in ``df``.  Defaults to ``np.mean``.

    Returns
    -------
    WTEpistasisResult

    Raises
    ------
    ValueError
        If the two sequences are identical (no mutations to resolve) or
        if ``max_order`` is exceeded.
    KeyError
        If any of the ``2**n`` intermediate variants is missing from the
        dataframe.
    """
    diff_positions = find_differing_positions(reference, destination)
    n = len(diff_positions)
    if n == 0:
        raise ValueError(
            f"reference == destination ({reference!r}); no mutations -> "
            "no epistasis term to compute."
        )
    if max_order is not None and n > max_order:
        raise ValueError(
            f"Hamming distance is {n} but --max-order is {max_order}. "
            "Either increase --max-order or pick a closer destination."
        )

    diff_states = [(reference[p], destination[p]) for p in diff_positions]

    variants: list[str] = []
    coeffs: list[int] = []
    phenotypes: list[float] = []

    # Iterate by subset size so the breakdown table reads naturally:
    # the first row is the all-reference variant (sign = (-1)^n), the
    # last row is the all-destination variant (sign = +1).
    for subset_size in range(n + 1):
        sign = (-1) ** (n - subset_size)
        for subset in itertools.combinations(diff_positions, subset_size):
            v = list(reference)
            for pos in subset:
                v[pos] = destination[pos]
            v_str = "".join(v)

            sub = df[df[variant_col].astype(str) == v_str]
            if sub.empty:
                raise KeyError(
                    f"variant {v_str!r} (subset of differing positions = {subset}) "
                    "is not present in the dataset; cannot compute "
                    f"order-{n} WT-dependent epistasis between "
                    f"{reference!r} and {destination!r}."
                )
            y = float(
                duplicate_aggregator(sub[phenotype_col].astype(float).to_numpy())
            )
            variants.append(v_str)
            coeffs.append(int(sign))
            phenotypes.append(y)

    eps = float(sum(c * y for c, y in zip(coeffs, phenotypes)))
    return WTEpistasisResult(
        order=n,
        reference=reference,
        destination=destination,
        differing_positions=diff_positions,
        differing_states=diff_states,
        variants=variants,
        coefficients=coeffs,
        phenotypes=phenotypes,
        epistasis=eps,
    )


def compute_pairwise_wt_epistasis(
    df: pd.DataFrame,
    *,
    variant_col: str,
    phenotype_col: str,
    reference: str,
    destination: str,
) -> WTEpistasisResult:
    """Convenience wrapper enforcing Hamming distance == 2 (epistasis square)."""
    return compute_wt_epistasis(
        df,
        variant_col=variant_col,
        phenotype_col=phenotype_col,
        reference=reference,
        destination=destination,
        max_order=2,
    )


def compute_triplet_wt_epistasis(
    df: pd.DataFrame,
    *,
    variant_col: str,
    phenotype_col: str,
    reference: str,
    destination: str,
) -> WTEpistasisResult:
    """Convenience wrapper enforcing Hamming distance == 3 (epistasis cube)."""
    return compute_wt_epistasis(
        df,
        variant_col=variant_col,
        phenotype_col=phenotype_col,
        reference=reference,
        destination=destination,
        max_order=3,
    )
