from setuptools import setup, find_packages

setup(
    name="orchid_epistasis_pba",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.0.0",
        "biopython>=1.80",
        "logomaker>=0.8",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
    ],
)
