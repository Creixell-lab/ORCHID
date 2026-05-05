# __init__.py

__version__ = "0.1.0"
__author__ = "Mingxuan Jiang"

from .io import add_position_columns, read_library_csv

from .model import (
    epistasis_code,
    epistasis_value,
    y_from_epi,
    find_matching_wildcards,
    string_indices,
    learn_positionwise_map_strict,
    decode_with_position_map,
)


from .pipeline import run_epistasis_pipeline

from .regression import (
    V1_INV,
    V_PHYS,
    BenchmarkResult,
    LoadedLibrary,
    build_design_matrix,
    load_example_library,
    map_to_design,
    plot_benchmark,
    run_benchmark,
)

__all__ = [
    "read_library_csv",
    "add_position_columns",
    "epistasis_code",
    "epistasis_value",
    "y_from_epi",
    "find_matching_wildcards",
    "string_indices",
    "run_epistasis_pipeline",
    "learn_positionwise_map_strict",
    "decode_with_position_map",
    "V_PHYS",
    "V1_INV",
    "BenchmarkResult",
    "LoadedLibrary",
    "load_example_library",
    "build_design_matrix",
    "map_to_design",
    "run_benchmark",
    "plot_benchmark",
]
