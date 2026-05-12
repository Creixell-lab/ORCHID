<p align="center">
  <img src="orchid_1280x640.png" alt="ORCHID cover art" width="100%">
</p>

# ORCHID: ORigin-independent and Context-exHaustIve analysis of CDMS libraries

This is the official repository of ORCHID for the following paper:
Identifying Protein Superbinders and Molecular Determinants of Epistasis with Combinatorial Deep Mutational Scanning (CDMS) Libraries

Mingxuan Jiang<sup>1</sup>, Mohan Sun<sup>1</sup>, Nuo Cheng<sup>1</sup>, Mihkel Örd<sup>1</sup>, Teresa L. Augustin<sup>1</sup>,  
Allyson Li<sup>2</sup>, Neel H. Shah<sup>2</sup>, Jesse Rinehart<sup>3,4</sup>, Helen R. Mott<sup>5</sup>, Pau Creixell<sup>1</sup><sup>*</sup>

## Affiliations

<sup>1</sup> Cancer Research UK Cambridge Institute, University of Cambridge, Li Ka Shing Centre, Robinson Way, Cambridge CB2 0RE, UK  
<sup>2</sup> Department of Chemistry, Columbia University, 3000 Broadway, New York, NY 10027, USA  
<sup>3</sup> Department of Cellular & Molecular Physiology, Yale School of Medicine, New Haven, CT 06520, USA  
<sup>4</sup> Systems Biology Institute, Yale University, West Haven, CT 06516, USA  
<sup>5</sup> Department of Biochemistry, University of Cambridge, Tennis Court Road, Cambridge CB2 1QW, UK  

<sup>*</sup> Corresponding author

## Description of the Method

Code for WT independent Epistasis Calculations
![image](https://github.com/user-attachments/assets/828ab39c-3dbe-401e-9115-7d4a017cb8cb)





[![Open example notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creixell-lab/ORCHID/blob/main/Epistasis_20250821_PIN136.ipynb)

Example notebook:  
[`Epistasis_20250821_PIN136.ipynb`](https://github.com/Creixell-lab/ORCHID/blob/main/Epistasis_20250821_PIN136.ipynb)

This notebook provides an interactive walkthrough of the ORCHID epistasis analysis pipeline, including phenotype processing, Walsh–Hadamard-based epistasis coefficient calculation, model interpretation, and example analysis of the PIN1 36-variant CDMS library.


ORCHID is a WT-agnostic, Walsh-Hadamard based framework that can treats phenotype-genotype datasets using a CDMS (combinatorial deep mutational scanning) approach with interpretable epistasis modelling to generate 1st order, 2nd order, 3rd order and higher order of epistasis coefficients. With this coefficients, we can quantify protein interaction landscapes and show interactions that can be explained with single mutations (1st order) and interactions that rely on cumulative, additive interactions of amino acids. Using these epistasis coefficients can also help to predict out of sample phenotypes for missing or unsampled variants, with high degrees of confidence with as sparse as 1% of the fitness landscape. This approach expands on the exisiting Walsh-Hadamard matrices, but are now configured to be wild-type free without need for referencing to a particular backround, and is able to analytically generate all mutational paths involved in a statistical epistasis calculation to explain the contributions leading to a wild-type independent epistasis coefficient value. 

## ORCHID Epistasis Code Installation & Usage Guide


ORCHID epistasis can also be run locally using the following pip installation guide

# 
This guide details how to install the ORCHID package locally for development and analysis.

## Prerequisites
* **Python >= 3.8+** 

## Installation Steps

### 1. Clone the Repository
First, download the code to your local machine:

```bash
git clone https://github.com/Creixell-lab/ORCHID.git
cd ORCHID
```

### 2. Create environment

```bash
python3 -m venv .venv # or any other environment name
```

### 3. Activate environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

### 5. Verify installation

```bash
orchid-epistasis-pba --help
```

### 6. Run Example

```bash
orchid-epistasis-pba \
  --input "example_files/210825_PIN1_36_library.csv" \
  --outdir "./orchid_output" \
  --variant-col pep_encoded \
  --phenotype-col PD_input_mean \
  --n 6 --k 3 --max-order 3 \
  --alphabet a,b,c
```

### Usage Guide

**Input Parameters:**

* `--input`: Path to the input `.csv` file (can be tab or comma delimited). 
* **Example:** [`example_files/210825_PIN1_36_library.csv`](https://github.com/Creixell-lab/ORCHID/blob/main/example_files/210825_PIN1_36_library.csv)
* `--variant-col`: Column name containing the variant sequences (genotypes). These can be encoded or actual amino acids.
* `--phenotype-col`: Column name containing the experimental or computational phenotype measurement.
* `--n`: The length of the sequence (number of positions).
* `--k`: The size of the alphabet (number of possible amino acids/characters per position).
* `--max-order`: The maximum order of epistasis interactions to include in the model.
* `--alphabet`: A comma-separated list of characters used in the genotypes (e.g., `a,b,c` or `A,C,D,E...`).

**Output Files:**

The script generates two files in the specified output directory:

1.  `predicted.csv`: Contains the predicted phenotypes calculated by adding epistatic terms from the 1st order up to the specified `max-order`.
2.  `epistasis_terms.csv`: Contains the calculated coefficients (weights) for the epistatic terms used in the model.

## Regression-based comparison benchmarks

Two additional commands are bundled to compare the WH-based PBA pipeline against
ElasticNet polynomial regression on the same 6-position, k=3 PIN1 example
library. Inputs (the bundled `210825_PIN1_36_library.csv`), the alphabet,
phenotype column and hyperparameter grids are all fixed — these commands are
reproducible reference benchmarks rather than general-purpose tools.

Both commands run the same nested cross-validated ElasticNet pipeline
(20 random repeats × 5-fold outer / 5-fold inner CV across 12 training-set
truncation levels from 10 % → 100 %), parallelised across all available CPU
cores via `joblib`. They differ only in the basis matrix used to encode the
genotype:

| Command | Basis | Description |
|---|---|---|
| `orchid-epistasis-regression-benchmark` | `V_PHYS` | Physics-correct equilateral simplex contrast matrix (centred, equal pairwise distances, unit-variance columns). |
| `wh-extension-regression`               | `V1_INV` | Direct Fauré marginal contrast matrix — the Walsh–Hadamard extension to k=3 used elsewhere in ORCHID. |

Run either of them from any directory after installing the package:

```bash
orchid-epistasis-regression-benchmark      # writes ./orchid_epistasis_regression_benchmark_output/
wh-extension-regression                    # writes ./wh_extension_regression_output/
```

Both commands accept a single optional flag `-j` / `--n-jobs` to cap the
joblib worker pool (default `-1` = all CPU cores). Useful when you want to
leave headroom on a laptop:

```bash
orchid-epistasis-regression-benchmark -j 4   # use 4 CPU cores instead of all
```

Note: `-j` (rather than `-n`) is used here so it does not collide with
`orchid-epistasis-pba --n`, which means "number of sequence positions".

Each command writes three artefacts to its output directory:

1.  `results.csv` — summary table with columns `Frac, N, mean_R2, std_R2, n_folds`
2.  `raw_folds.csv` — every (truncation, seed, fold) row, for downstream plotting
3.  `r2_vs_fraction.png` — R² vs training fraction with error bars

Expect the benchmark to take on the order of an hour per command on a typical
laptop with all cores in use; CPU is the bottleneck.

## General QR-derived regression: `orchid-regression-general`

For arbitrary `n` (positions) and `K` (alphabet size), `orchid-regression-general`
runs a fully general ORCHID/Helmert-style ElasticNet regression. Unlike the
two fixed benchmarks above, it accepts your own `--input` CSV, `--variant-col`,
`--phenotype-col`, `--n`, `--k` and `--max-order`, and:

1. Builds the redundant one-site interpretable matrix `H_redundant` and its
   reduced square form `H'`.
2. Runs **QR decomposition** on `H'` (with a deterministic sign convention)
   to obtain the orthonormal basis `Phi`, the coefficient-extraction matrix
   `G_tilde = Phi / K`, and the inverse/design matrix `V = G_tilde⁻¹`.
3. Lifts to `n` sites via Kronecker power: `V_full = V^⊗n`,
   `T_full = T^⊗n` where `T = G_original · V` maps the QR-derived `ε̃`
   coefficients to the redundant ORCHID-style `ε` coefficients.
4. Fits a nested-CV ElasticNet model **per epistasis order**, **un-scales**
   the StandardScaler-transformed coefficients, and writes both `ε̃` and
   `ε_original_redundant` to disk along with predictions and R² values.

```bash
orchid-regression-general \
  --input "example_files/210825_PIN1_36_library.csv" \
  --outdir "./orchid_general_output" \
  --variant-col pep_encoded \
  --phenotype-col PD_input_mean \
  --n 6 --k 3 --max-order 3 \
  --orders 1,2,3 \
  --alphabet a,b,c \
  --cv-folds 5 --cv-repeats 5 \
  -j 4
```

The output directory contains:

| File | Contents |
|---|---|
| `predictions_by_order.csv` | In-sample (`pred_order_e`) and out-of-fold (`cv_pred_order_e`) predictions for every fitted order |
| `epsilon_tilde_by_order.csv` | Coefficients in the QR-derived ORCHID basis, per order |
| `epsilon_original_redundant_by_order.csv` | Intuitive redundant ORCHID-style coefficients (`T_order @ β̃`), per order |
| `r2_by_order.csv` | Per-order summary R² (CV mean ± std, OOF, and full-fit) |
| `cv_folds_by_order.csv` | Every individual outer-fold result, with the chosen `alpha`/`l1_ratio` |
| `G_one_site_*.csv`, `V_one_site_*.csv`, `Phi_one_site_*.csv`, `H_one_site_*.csv`, `T_one_site_*.csv` | Basis-construction diagnostics for inspection |

`-j` / `--n-jobs` parallelises across epistasis orders. As with the two
benchmark commands above, `-n` is **not** the cores flag — `--n` already
means number of sequence positions in this CLI.

> **Memory note.** This is a full Kronecker construction: `V_full` has shape
> `K^n × K^n`, so the actual memory cost scales as `K^(2n)` float64 entries.
> It is well suited to small / mid-sized CDMS alphabets (typical PIN1-style
> libraries with `K = 3`, `K = 4`, and any reasonable `K ≤ 20` at short `n`).
> A soft cap is provided via `--max-genotype-space` (default `20**6 =
> 64_000_000`); raise it explicitly for a bigger run. Above that scale NumPy
> will simply raise `MemoryError` at allocation time. For genuinely large
> alphabets prefer `orchid-epistasis-pba` (the WH-PBA pipeline), which
> scales much better.

## Phenotype linearisation: `orchid-linearise`

Saturating dynamic range and other forms of nonspecific epistasis often
distort the relationship between the underlying genetic-score and the
measured phenotype, exactly as discussed in
[Park, Metzger & Thornton 2024](https://www.nature.com/articles/s41467-024-51895-5).
`orchid-linearise` fits a small **catalog of nonlinear link functions**
`y = f(x; θ)` between an observed phenotype `y` and a first-order ORCHID
prediction `x`, ranks them by R² / AIC / BIC, and tells you which one
best explains the residual nonlinearity.

| Method | Formula | # params | Notes |
|---|---|---|---|
| `identity` | `y = a·x + b` | 2 | Baseline. If this wins, your data is already linear. |
| `sigmoid_2p` | `y = L + (U − L)/(1 + e^(−x))` | 2 | Raw Metzger sigmoid; assumes `x` is already centred at half-occupancy. |
| `sigmoid_4p` | `y = L + (U − L)/(1 + e^(−k(x−x₀)))` | 4 | 4PL — the workhorse generalisation of the Metzger sigmoid. |
| `sigmoid_5p` (Richards) | `y = L + (U − L)/(1 + e^(−k(x−x₀)))^m` | 5 | Asymmetric 5PL. |
| `tanh_4p` | `y = a·tanh(k(x − x₀)) + b` | 4 | Reparametrised sigmoid; sometimes more numerically stable. |
| `erf_6p` | `(a·x+b) + (c·x+d) − ((a·x+b)−(c·x+d))·erf((x−e)·f)` | 6 | The dual-affine erf blend used in the original ORCHID paper. |
| `bounded_linear_4p` | `softclip(a·x + b, L, U)` | 4 | The simplest "phenotype-bounding" model with a smooth knee. |

You either pass a column that already contains a first-order prediction:

```bash
orchid-linearise \
  --input "example_files/210825_PIN1_36_library.csv" \
  --outdir "./linearise_output" \
  --observed-col PD_input_mean \
  --predicted-col my_first_order_prediction
```

…or omit `--predicted-col` and let `orchid-linearise` run the existing
ORCHID first-order pipeline on the input for you:

```bash
orchid-linearise \
  --input "example_files/210825_PIN1_36_library.csv" \
  --outdir "./linearise_output" \
  --observed-col PD_input_mean \
  --variant-col pep_encoded \
  --n 6 \
  --alphabet a,b,c
```

Outputs in the `--outdir`:

- `summary.csv` — every method ranked by R² with R², Pearson r², AIC, BIC, parameters, and status
- `transformed_predictions.csv` — observed and per-method `f(x; θ)` for every row
- `best_method.txt` — the winner under your chosen `--criterion` (`r2` / `aic` / `bic`)
- `fit_comparison.png` — combined plot overlaying every fitted curve on the data scatter
- `plots/fit_<method>.png` — a focused scatter+fit plot for each method

A `--methods method1,method2,...` flag lets you restrict the fit to a
subset (useful in CI), and `--no-plots` skips the PNG outputs.

## WT-dependent (variant-specific) epistasis: `orchid-wt-epistasis`

The other ORCHID commands compute *reference-free* / ensemble-averaged
epistasis. Sometimes you want the opposite: the **classical, variant-specific,
WT-dependent (reference-based)** epistasis between a particular reference
sequence and a particular destination, computed by the standard finite-
difference / inclusion–exclusion sum over the 2ⁿ corners of the n-dimensional
hypercube spanned by the differing sites.

For Hamming distance `n = 2` (the "epistasis square"):

```
ε⁽²⁾ = y(AB) − y(A) − y(B) + y(WT)
```

For Hamming distance `n = 3` (the "epistasis cube"):

```
ε⁽³⁾ = y(ABC) − (y(AB) + y(AC) + y(BC)) + (y(A) + y(B) + y(C)) − y(WT)
```

Higher orders are supported automatically as long as all 2ⁿ intermediate
variants exist in the dataset.

```bash
orchid-wt-epistasis \
  --input "example_files/210825_PIN1_36_library.csv" \
  --variant-col pep_encoded \
  --phenotype-col PD_input_mean \
  --reference   aaaaaa \
  --destination bbaaaa \
  --out ./pairwise_breakdown.csv
```

Output is a printed summary and an optional CSV breakdown listing every
intermediate variant, its inclusion–exclusion sign and contribution.

You can also call it from Python:

```python
from orchid_epistasis_pba import compute_wt_epistasis
import pandas as pd

df = pd.read_csv("example_files/210825_PIN1_36_library.csv")
result = compute_wt_epistasis(
    df,
    variant_col="pep_encoded",
    phenotype_col="PD_input_mean",
    reference="aaaaaa",
    destination="bbcaaa",          # Hamming distance 3 -> epistasis cube
)
print(result.epistasis)            # the order-3 term
result.breakdown_dataframe()       # one row per corner of the cube
```
