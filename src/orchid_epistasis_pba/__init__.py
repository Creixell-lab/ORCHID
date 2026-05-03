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
]
