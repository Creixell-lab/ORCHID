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

from .cli_general_regression import (
    OrderFitResult,
    build_design_for_order,
    encode_variants,
    extract_unscaled_coefficients,
    fit_one_order,
    genotype_row_indices,
    kron_power,
    make_orchid_raw_H,
    make_original_orchid_G_one_site,
    make_orthonormal_G_one_site,
    make_tilde_to_original_one_site,
    original_coefficient_orders,
    tilde_coefficient_orders,
)

from .linearisation import (
    LINK_SPECS,
    FitResult,
    LinkSpec,
    best_method,
    fit_all_methods,
    fit_method,
    link_bounded_linear_4p,
    link_erf_6p,
    link_identity,
    link_sigmoid_2p,
    link_sigmoid_4p,
    link_sigmoid_5p,
    link_tanh_4p,
    plot_combined,
    plot_per_method,
    summary_dataframe,
)

from .wt_epistasis import (
    WTEpistasisResult,
    compute_pairwise_wt_epistasis,
    compute_triplet_wt_epistasis,
    compute_wt_epistasis,
    find_differing_positions,
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
    "OrderFitResult",
    "build_design_for_order",
    "encode_variants",
    "extract_unscaled_coefficients",
    "fit_one_order",
    "genotype_row_indices",
    "kron_power",
    "make_orchid_raw_H",
    "make_original_orchid_G_one_site",
    "make_orthonormal_G_one_site",
    "make_tilde_to_original_one_site",
    "original_coefficient_orders",
    "tilde_coefficient_orders",
    "LINK_SPECS",
    "FitResult",
    "LinkSpec",
    "best_method",
    "fit_all_methods",
    "fit_method",
    "link_bounded_linear_4p",
    "link_erf_6p",
    "link_identity",
    "link_sigmoid_2p",
    "link_sigmoid_4p",
    "link_sigmoid_5p",
    "link_tanh_4p",
    "plot_combined",
    "plot_per_method",
    "summary_dataframe",
    "WTEpistasisResult",
    "compute_pairwise_wt_epistasis",
    "compute_triplet_wt_epistasis",
    "compute_wt_epistasis",
    "find_differing_positions",
]
