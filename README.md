<p align="center">
  <img src="orchid_github_social_preview_1280x640.png" alt="ORCHID cover art" width="100%">
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


### Future Updates

 1.  Incoporate data linearisation from the .ipynb into the python script, `orchid-epistasis-pba` assumes data is linearised already
 2.  `orchid-epistasis-pba` relies on Partial Background Averaging (PBA) based on the WH transform; an alternate script that uses ridge regression with automatic alpha optimisation is planned
 3.  output should generate more data visualisations, images and r2 values seen in the .ipynb

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
