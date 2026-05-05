"""Linearisation / global-nonlinearity link functions for ORCHID.

Given an observed phenotype ``y`` and a first-order ("genetic-score" /
"linear-motif") prediction ``x``, this module fits a catalog of common
link functions ``y = f(x; theta)`` and reports which one best removes
the nonlinear distortion.  The motivation is the same as in Metzger,
Park & Thornton 2024 (Nat Commun, 51895-5): bounded dynamic range and
other forms of nonspecific epistasis cause `y` to be a saturating /
sigmoidal function of the underlying genetic score, and re-fitting the
linear model on a properly linearised `y` (or, equivalently, on the
score scale via the inverse link) can dramatically improve the
first-order R^2.

Catalog of link functions (all fit by ``scipy.optimize.curve_fit``):

* ``identity``           -- baseline, ``y = a x + b``
* ``sigmoid_2p``         -- raw Metzger sigmoid, ``y = L + (U-L)/(1+exp(-x))``
* ``sigmoid_4p``         -- 4PL sigmoid, ``y = L + (U-L)/(1+exp(-k(x-x0)))``
* ``sigmoid_5p``         -- Richards (asymmetric 5PL)
* ``tanh_4p``            -- ``y = a tanh(k(x-x0)) + b``
* ``erf_6p``             -- the dual-affine erf blend used in the
                            original ORCHID paper (Mingxuan's
                            ``error_function3``)
* ``bounded_linear_4p``  -- linear with smooth top and bottom clamps

Each fit yields a :class:`FitResult` with the parameters, the
transformed predictions, R^2 (sklearn coefficient of determination),
Pearson r^2, AIC and BIC.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------

def link_identity(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def link_sigmoid_2p(x: np.ndarray, L: float, U: float) -> np.ndarray:
    """Metzger raw sigmoid: y = L + (U - L) / (1 + e**-x).

    Best when ``x`` is already a centred genetic score with x = 0 at
    half-occupancy.
    """
    return L + (U - L) / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def link_sigmoid_4p(x: np.ndarray, L: float, U: float, k: float, x0: float) -> np.ndarray:
    """4-parameter logistic (4PL): y = L + (U - L) / (1 + exp(-k (x - x0)))."""
    z = np.clip(k * (x - x0), -700.0, 700.0)
    return L + (U - L) / (1.0 + np.exp(-z))


def link_sigmoid_5p(
    x: np.ndarray, L: float, U: float, k: float, x0: float, m: float
) -> np.ndarray:
    """Richards / asymmetric 5PL.

    ``y = L + (U - L) / (1 + exp(-k (x - x0))) ** m``
    """
    z = np.clip(k * (x - x0), -700.0, 700.0)
    base = 1.0 + np.exp(-z)
    return L + (U - L) / np.power(base, np.clip(m, 1e-3, 50.0))


def link_tanh_4p(x: np.ndarray, a: float, b: float, k: float, x0: float) -> np.ndarray:
    """y = a * tanh(k (x - x0)) + b."""
    return a * np.tanh(k * (x - x0)) + b


def link_erf_6p(
    x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float
) -> np.ndarray:
    """Mingxuan's original ORCHID-paper link.

    A smooth blend between two affine regimes ``a x + b`` and ``c x + d``
    via an erf transition centred at ``e`` with steepness ``f``::

        t1 = a x + b
        t2 = c x + d
        y  = t1 + t2 - (t1 - t2) * erf((x - e) * f)
    """
    t1 = a * x + b
    t2 = c * x + d
    return t1 + t2 - (t1 - t2) * special.erf((x - e) * f)


def _softplus(z: np.ndarray, beta: float = 5.0) -> np.ndarray:
    return np.log1p(np.exp(np.clip(beta * z, -500.0, 500.0))) / beta


def link_bounded_linear_4p(
    x: np.ndarray, a: float, b: float, L: float, U: float
) -> np.ndarray:
    """Affine line with smooth (softplus) top and bottom clamps.

    ``inner = a x + b``; the result is ``inner`` clamped between ``L``
    and ``U`` using a softplus smoothing so that ``curve_fit`` sees a
    differentiable surface.  As ``inner`` goes to +/-infty the output
    saturates at ``U`` / ``L`` respectively.
    """
    inner = a * x + b
    raw = U - _softplus(U - inner)            # upper clamp (smooth min)
    return L + _softplus(raw - L)              # lower clamp (smooth max)


# ---------------------------------------------------------------------------
# LinkSpec: how to fit each one
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkSpec:
    name: str
    func: Callable[..., np.ndarray]
    n_params: int
    param_names: tuple[str, ...]
    p0_fn: Callable[[np.ndarray, np.ndarray], list[float]]
    bounds_fn: Callable[[np.ndarray, np.ndarray], tuple[list[float], list[float]]]
    description: str


def _stats_for_init(y: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.min(y)), float(np.max(y)), float(np.mean(y)), float(np.std(y))


def _x_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x))


_INF = np.inf


def _identity_p0(x, y):
    s_y = max(np.std(y), 1e-9)
    s_x = max(np.std(x), 1e-9)
    a0 = (np.cov(x, y, ddof=0)[0, 1]) / (s_x ** 2)
    b0 = float(np.mean(y) - a0 * np.mean(x))
    return [float(a0), b0]


def _identity_bounds(x, y):
    return ([-_INF, -_INF], [_INF, _INF])


def _sigmoid2_p0(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    return [L0, U0]


def _sigmoid2_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    span = max(U0 - L0, 1e-6)
    return ([L0 - 5 * span, L0 - 5 * span], [U0 + 5 * span, U0 + 5 * span])


def _sigmoid4_p0(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, x_mean, x_std = _x_stats(x)
    k0 = 4.0 / max(x_max - x_min, 1e-6)
    return [L0, U0, k0, x_mean]


def _sigmoid4_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, _, _ = _x_stats(x)
    span_y = max(U0 - L0, 1e-6)
    return (
        [L0 - 5 * span_y, L0 - 5 * span_y, 1e-6, x_min - 5 * (x_max - x_min)],
        [U0 + 5 * span_y, U0 + 5 * span_y, 1e3 / max(x_max - x_min, 1e-6), x_max + 5 * (x_max - x_min)],
    )


def _sigmoid5_p0(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, x_mean, _ = _x_stats(x)
    k0 = 4.0 / max(x_max - x_min, 1e-6)
    return [L0, U0, k0, x_mean, 1.0]


def _sigmoid5_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, _, _ = _x_stats(x)
    span_y = max(U0 - L0, 1e-6)
    return (
        [L0 - 5 * span_y, L0 - 5 * span_y, 1e-6, x_min - 5 * (x_max - x_min), 1e-2],
        [U0 + 5 * span_y, U0 + 5 * span_y, 1e3 / max(x_max - x_min, 1e-6), x_max + 5 * (x_max - x_min), 50.0],
    )


def _tanh4_p0(x, y):
    L0, U0, y_mean, _ = _stats_for_init(y)
    x_min, x_max, x_mean, _ = _x_stats(x)
    a0 = (U0 - L0) / 2.0
    b0 = (U0 + L0) / 2.0
    k0 = 4.0 / max(x_max - x_min, 1e-6)
    return [a0, b0, k0, x_mean]


def _tanh4_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, _, _ = _x_stats(x)
    span_y = max(U0 - L0, 1e-6)
    return (
        [-5 * span_y, L0 - 5 * span_y, 1e-6, x_min - 5 * (x_max - x_min)],
        [5 * span_y, U0 + 5 * span_y, 1e3 / max(x_max - x_min, 1e-6), x_max + 5 * (x_max - x_min)],
    )


def _erf6_p0(x, y):
    a0, b0 = _identity_p0(x, y)
    x_mean = float(np.mean(x))
    x_span = max(np.std(x), 1e-6)
    # Two affine pieces both initialised to the OLS line; the erf will
    # smoothly interpolate between them and curve_fit can pull them
    # apart if doing so improves the fit.  ``f`` controls the steepness
    # of the transition and is initialised to span the data.
    return [a0, b0 / 2.0, a0, b0 / 2.0, x_mean, 1.0 / x_span]


def _erf6_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    x_min, x_max, _, _ = _x_stats(x)
    span_y = max(U0 - L0, 1e-6)
    big = 100 * span_y
    return (
        [-big, -big, -big, -big, x_min - 5 * (x_max - x_min), -1e3],
        [big, big, big, big, x_max + 5 * (x_max - x_min), 1e3],
    )


def _bounded_linear_p0(x, y):
    a0, b0 = _identity_p0(x, y)
    L0, U0, _, _ = _stats_for_init(y)
    return [float(a0), b0, L0, U0]


def _bounded_linear_bounds(x, y):
    L0, U0, _, _ = _stats_for_init(y)
    span_y = max(U0 - L0, 1e-6)
    return (
        [-_INF, -_INF, L0 - 5 * span_y, L0 - 5 * span_y],
        [_INF, _INF, U0 + 5 * span_y, U0 + 5 * span_y],
    )


LINK_SPECS: dict[str, LinkSpec] = {
    "identity": LinkSpec(
        "identity",
        link_identity,
        2,
        ("a", "b"),
        _identity_p0,
        _identity_bounds,
        "y = a * x + b",
    ),
    "sigmoid_2p": LinkSpec(
        "sigmoid_2p",
        link_sigmoid_2p,
        2,
        ("L", "U"),
        _sigmoid2_p0,
        _sigmoid2_bounds,
        "y = L + (U - L) / (1 + exp(-x)); raw Metzger form",
    ),
    "sigmoid_4p": LinkSpec(
        "sigmoid_4p",
        link_sigmoid_4p,
        4,
        ("L", "U", "k", "x0"),
        _sigmoid4_p0,
        _sigmoid4_bounds,
        "y = L + (U - L) / (1 + exp(-k (x - x0))); 4PL",
    ),
    "sigmoid_5p": LinkSpec(
        "sigmoid_5p",
        link_sigmoid_5p,
        5,
        ("L", "U", "k", "x0", "m"),
        _sigmoid5_p0,
        _sigmoid5_bounds,
        "Richards / asymmetric 5PL",
    ),
    "tanh_4p": LinkSpec(
        "tanh_4p",
        link_tanh_4p,
        4,
        ("a", "b", "k", "x0"),
        _tanh4_p0,
        _tanh4_bounds,
        "y = a tanh(k (x - x0)) + b",
    ),
    "erf_6p": LinkSpec(
        "erf_6p",
        link_erf_6p,
        6,
        ("a", "b", "c", "d", "e", "f"),
        _erf6_p0,
        _erf6_bounds,
        "ORCHID-paper dual-affine erf blend (error_function3)",
    ),
    "bounded_linear_4p": LinkSpec(
        "bounded_linear_4p",
        link_bounded_linear_4p,
        4,
        ("a", "b", "L", "U"),
        _bounded_linear_p0,
        _bounded_linear_bounds,
        "y = softclip(a x + b, L, U); smooth phenotype-bounding",
    ),
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    name: str
    spec: LinkSpec
    status: str                        # "ok" or "failed: ..."
    params: np.ndarray | None
    param_dict: dict[str, float]
    transformed: np.ndarray | None     # f(x; theta) at the data
    n: int
    n_params: int
    rss: float
    r2: float                          # sklearn r2_score(y, transformed)
    pearson_r2: float                  # corr(y, transformed) ** 2
    aic: float
    bic: float


def _aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    rss = max(rss, 1e-300)
    twoll = -n * (np.log(2 * np.pi) + 1.0 + np.log(rss / n))
    aic = 2 * k - twoll
    bic = k * np.log(max(n, 1)) - twoll
    return float(aic), float(bic)


def _failed(spec: LinkSpec, n: int, status: str) -> FitResult:
    return FitResult(
        name=spec.name,
        spec=spec,
        status=status,
        params=None,
        param_dict={},
        transformed=None,
        n=n,
        n_params=spec.n_params,
        rss=float("nan"),
        r2=float("nan"),
        pearson_r2=float("nan"),
        aic=float("nan"),
        bic=float("nan"),
    )


def fit_method(
    spec: LinkSpec,
    x: np.ndarray,
    y: np.ndarray,
    *,
    maxfev: int = 200_000,
) -> FitResult:
    """Fit one link function and return a :class:`FitResult`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < spec.n_params + 1:
        return _failed(spec, n, f"failed: n={n} < n_params+1={spec.n_params+1}")

    p0 = spec.p0_fn(x, y)
    lower, upper = spec.bounds_fn(x, y)
    # Clamp p0 strictly inside the bounds so curve_fit doesn't reject.
    p0 = [
        min(max(p0_i, low + 1e-9 * abs(low) - 1e-9), high - 1e-9 * abs(high) - 1e-9)
        if np.isfinite(low) and np.isfinite(high) else p0_i
        for p0_i, low, high in zip(p0, lower, upper)
    ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, _ = curve_fit(
                spec.func,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=maxfev,
            )
    except Exception as exc:  # noqa: BLE001
        return _failed(spec, n, f"failed: {type(exc).__name__}: {exc}")

    transformed = spec.func(x, *popt)
    if not np.all(np.isfinite(transformed)):
        return _failed(spec, n, "failed: non-finite transformed predictions")

    rss = float(np.sum((y - transformed) ** 2))
    r2 = float(r2_score(y, transformed))
    if np.std(transformed) < 1e-12 or np.std(y) < 1e-12:
        pearson_r2 = float("nan")
    else:
        pearson_r2 = float(stats.pearsonr(y, transformed)[0] ** 2)
    aic, bic = _aic_bic(rss, n, spec.n_params)

    return FitResult(
        name=spec.name,
        spec=spec,
        status="ok",
        params=np.asarray(popt, dtype=float),
        param_dict=dict(zip(spec.param_names, [float(v) for v in popt])),
        transformed=transformed,
        n=n,
        n_params=spec.n_params,
        rss=rss,
        r2=r2,
        pearson_r2=pearson_r2,
        aic=aic,
        bic=bic,
    )


def fit_all_methods(
    x: np.ndarray,
    y: np.ndarray,
    methods: Iterable[str] | None = None,
) -> dict[str, FitResult]:
    """Fit every method in :data:`LINK_SPECS` (or the named subset)."""
    if methods is None:
        names = list(LINK_SPECS.keys())
    else:
        names = list(methods)
    results: dict[str, FitResult] = {}
    for name in names:
        if name not in LINK_SPECS:
            raise ValueError(
                f"Unknown method {name!r}. Known: {sorted(LINK_SPECS.keys())}"
            )
        results[name] = fit_method(LINK_SPECS[name], x, y)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_dataframe(results: dict[str, FitResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "method": name,
                "n_params": r.n_params,
                "n": r.n,
                "status": r.status,
                "r2": r.r2,
                "pearson_r2": r.pearson_r2,
                "rss": r.rss,
                "aic": r.aic,
                "bic": r.bic,
                "params": (
                    "; ".join(f"{k}={v:.6g}" for k, v in r.param_dict.items())
                    if r.param_dict
                    else ""
                ),
                "description": r.spec.description,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(by="r2", ascending=False, na_position="last").reset_index(drop=True)
    return df


def best_method(results: dict[str, FitResult], criterion: str = "r2") -> str:
    """Name of the best method under one of ``r2`` (max), ``aic`` (min), ``bic`` (min)."""
    valid = [(n, r) for n, r in results.items() if r.status == "ok" and np.isfinite(getattr(r, criterion))]
    if not valid:
        raise ValueError("No method fitted successfully.")
    if criterion == "r2":
        valid.sort(key=lambda nr: nr[1].r2, reverse=True)
    elif criterion in {"aic", "bic"}:
        valid.sort(key=lambda nr: getattr(nr[1], criterion))
    else:
        raise ValueError(f"criterion must be 'r2', 'aic', or 'bic', got {criterion!r}")
    return valid[0][0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth_curve(
    spec: LinkSpec, params: np.ndarray, x_min: float, x_max: float, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    span = x_max - x_min
    pad = 0.05 * span if span > 0 else 1.0
    grid = np.linspace(x_min - pad, x_max + pad, n)
    return grid, spec.func(grid, *params)


def plot_per_method(
    x: np.ndarray,
    y: np.ndarray,
    results: dict[str, FitResult],
    outdir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    x_min, x_max = float(np.min(x)), float(np.max(x))

    for name, r in results.items():
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(x, y, s=8, alpha=0.5, color="C0", label="data")
        if r.status == "ok" and r.params is not None:
            grid, ysmooth = _smooth_curve(r.spec, r.params, x_min, x_max)
            ax.plot(grid, ysmooth, "r-", lw=2.0, label=f"fit (R\u00b2={r.r2:.4f})")
        else:
            ax.text(0.5, 0.5, r.status, ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_xlabel("first-order prediction (x)")
        ax.set_ylabel("observed phenotype (y)")
        ax.set_title(f"{name}\n{r.spec.description}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / f"fit_{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths[name] = path
    return paths


def plot_combined(
    x: np.ndarray,
    y: np.ndarray,
    results: dict[str, FitResult],
    outdir: Path,
    filename: str = "fit_comparison.png",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.scatter(x, y, s=8, alpha=0.4, color="grey", label="data")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    cmap = plt.get_cmap("tab10")
    valid = [(n, r) for n, r in results.items() if r.status == "ok" and r.params is not None]
    valid.sort(key=lambda nr: -nr[1].r2)
    for i, (name, r) in enumerate(valid):
        grid, ysmooth = _smooth_curve(r.spec, r.params, x_min, x_max)
        ax.plot(grid, ysmooth, lw=1.6, color=cmap(i % 10), label=f"{name} (R\u00b2={r.r2:.4f})")
    ax.set_xlabel("first-order prediction (x)")
    ax.set_ylabel("observed phenotype (y)")
    ax.set_title("Linearisation candidates")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
