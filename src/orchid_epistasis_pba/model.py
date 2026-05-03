from __future__ import annotations

from itertools import combinations
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, special, stats
from scipy.optimize import root_scalar

from itertools import combinations
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, special, stats
from scipy.optimize import root_scalar


def convert_nary_to_alpha(index: str, mappings: dict[int, list[str]]) -> str:
    """
    Convert an N-nary index (with digits) to an alphabetical index based on position mappings.
    """
    alpha_index = list(index)
    for pos, mapping in mappings.items():
        if index[pos].isdigit():
            digit = int(index[pos])
            if digit < len(mapping):
                alpha_index[pos] = mapping[digit]
    return "".join(alpha_index)


def string_indices(strings: Iterable[str], alphabet: Sequence[str], length: int = 4) -> np.ndarray:
    # converts an alphabetical order into a numerical order.
    # i.e. AAAA is 0, AAAB is 1 etc.
    strings = list(strings)
    if not all(len(s) == length for s in strings):
        raise ValueError(f"All strings must be {length} characters long")

    char_to_index = {char: idx for idx, char in enumerate(alphabet)}
    indices = np.array([[char_to_index[char] for char in s] for s in strings], dtype=int)

    base = len(alphabet)
    powers = base ** np.arange(length - 1, -1, -1)
    return np.sum(indices * powers, axis=1)


def e_scaling(original_matrix: np.ndarray, n: int = 3, k: int = 3, e: int = 1) -> np.ndarray:
    dim_scaler = original_matrix.shape[-1]
    a_r = np.arange(k)
    arange_matrix = np.vstack([a_r for _ in range(2**e)])
    output_terms = np.kron(original_matrix, np.ones(k, dtype=int)) + k ** (n - 1) * np.kron(
        np.ones(dim_scaler, dtype=int), arange_matrix
    )
    return output_terms


def kron_string(a: list[str], b: list[str]) -> list[str]:
    output = []
    for i in range(len(a)):
        for j in range(len(b)):
            output.append(a[i] + b[j])
    return output


def recrusive_kron(base_matrix: np.ndarray, instances: int) -> np.ndarray:
    matrix = base_matrix
    while instances >= 2:
        matrix = np.kron(base_matrix, matrix)
        instances -= 1
    return matrix


def recrusive_kron_string(base_matrix: list[str], instances: int) -> list[str]:
    matrix = base_matrix
    while instances >= 2:
        matrix = kron_string(base_matrix, matrix)
        instances -= 1
    return matrix


def epistasis_equation_writer(e: int, first: np.ndarray, last: np.ndarray, *, k: int) -> np.ndarray:
    p = 0
    BB = last
    BA = last - ((last // k**p % k - first // k**p % k) * k**p)
    last_1 = np.array([BA, BB])
    last_2 = last_1

    while p < (e - 1):
        p = p + 1
        last_2 = last_1 - ((last // k**p % k - first // k**p % k) * k**p)
        last_2 = np.array(last_2.tolist() + last_1.tolist())
        last_1 = last_2

    return last_2


def find_matching_wildcards(string: str, wildcards: Iterable[str]) -> list[str]:
    matching_wildcards = []
    for wildcard in wildcards:
        match = True
        for i, char in enumerate(wildcard):
            if char != "*" and char != string[i]:
                match = False
                break
        if match:
            matching_wildcards.append(wildcard)
    return matching_wildcards


def epistasis_value(
    *,
    n: int,
    e: int,
    k: int,
    df_tang: pd.DataFrame,
    pheno: str,
    H2_terms_initial: np.ndarray,
    alphabet: Sequence[str],
) -> tuple[list[str], list[np.ndarray]]:
    """
    Notebook epistasis_value, with globals removed.
    Returns:
      epi_terms: list of wildcard strings (length = C(n,e) * k^e)
      E_2_list: list of arrays, one per combination (each length k^e)
    """
    E_2_list: list[np.ndarray] = []

    HA1 = np.array(([1, -1], [-1, 1]))
    HA2 = recrusive_kron(HA1, e)

    positions_ = [str(i) for i in range(n)]
    combinations_list = list(combinations(positions_, e))

    wildcard_string = "*" * n
    new_index = np.arange(k**n)

    epi_terms: list[str] = []
    # Epi strings must use the SAME symbols as pep_encoded (e.g. a/b/c)
    variants_list_1 = list(alphabet)
    if len(variants_list_1) != k:
        raise ValueError(f"k={k} but alphabet has {len(variants_list_1)} symbols")
    variants_list_2 = recrusive_kron_string(variants_list_1, e)


    # Work on a copy to avoid clobbering caller columns like 'pep_' / 'string_index'
    df_tang = df_tang.copy()

    for elements_to_remove in combinations_list:
        # name generator
        for characters in variants_list_2:
            strings_ = list(wildcard_string)
            for i, chars in zip(elements_to_remove, characters):
                strings_[int(i)] = chars
            epi_terms.append("".join(strings_))

        # main code to get epistasis value
        positions = [str(i) for i in range(n)]
        positions = [elem for elem in positions if elem not in elements_to_remove]
        positions = positions + list(elements_to_remove)

        df_tang["pep_"] = df_tang[positions].apply(lambda row: "".join(row), axis=1)
        df_tang["string_index"] = string_indices(df_tang["pep_"], alphabet, length=n)
        df_tang_ = df_tang.set_index("string_index")

        df_dict = df_tang_[pheno].to_dict()
        new_dict = {idx: df_dict.get(idx, np.nan) for idx in new_index}

        order_XXX = 0
        for row, sign in zip(H2_terms_initial, HA2[0]):
            order_XXX += sign * np.array([new_dict[key] for key in row])

        repeats = int(order_XXX.shape[0] / k**e)
        E_2_list.append(np.nanmean(order_XXX.reshape(k**e, repeats), axis=1))

    return epi_terms, E_2_list


def epistasis_code(
    *,
    k: int,
    n: int,
    e: int,
    df: pd.DataFrame,
    pheno: str,
    alphabet: Sequence[str],
) -> pd.DataFrame:
    """
    Notebook epistasis_code, with globals removed.
    Output columns: ['Epi', 'E_partial']
    """
    # notebook z1
    z1 = np.ones([k, k])
    np.fill_diagonal(z1, 0)

    max_distance = recrusive_kron(z1, e)
    first, last = sparse.csr_matrix(max_distance).nonzero()
    last_2 = epistasis_equation_writer(e, first, last, k=k)

    pp = e
    H2_terms_initial = last_2
    while pp < n:
        pp = pp + 1
        H2_terms_initial = e_scaling(H2_terms_initial, n=pp, e=e, k=k)

    zeroth_order_scale = k**n * (k) ** 0
    eth_order_scale = k ** (n - e) * (k - 1) ** e

    epi_terms, E_2_list = epistasis_value(
        n=n,
        e=e,
        k=k,
        df_tang=df,
        pheno=pheno,
        H2_terms_initial=H2_terms_initial,
        alphabet=alphabet,
    )

    E2_df = pd.DataFrame(
        [epi_terms, np.array(E_2_list).reshape(-1) * eth_order_scale / zeroth_order_scale]
    ).T
    E2_df.columns = ["Epi", "E_partial"]
    return E2_df


def y_from_epi(En_df: pd.DataFrame, test_data: pd.DataFrame, variant_name: str) -> list[float]:
    predicted_y_values = []
    for string in test_data.dropna(subset=[variant_name])[variant_name].astype(str):
        matching_wildcards = find_matching_wildcards(string, En_df["Epi"])
        df_wildcard = pd.DataFrame(matching_wildcards, columns=["Epi"])
        df_wildcard2 = pd.merge(df_wildcard, En_df, on="Epi")
        predicted_y_values.append(float(df_wildcard2.E_partial.sum()))
    return predicted_y_values


# --- Global-epistasis helper functions from notebook (kept as-is) ---

def error_function3(x, a, b, c, d, e, f):
    t1 = a * x + b
    t2 = c * x + d
    output = t1 + t2 - (t1 - t2) * special.erf((x - e) * f)
    return output


DEFAULT_ERROR_FUNCTION_BOUNDS = (
    [0.001, -np.inf, 0.001, -np.inf, -np.inf, -np.inf],
    [np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf],
)


def inverse_error_function3(y, a, b, c, d, e, f):
    difference_function = lambda x: error_function3(x, a, b, c, d, e, f) - y
    root = root_scalar(difference_function, bracket=[-100, 100])
    return root.root


def clip_values(arr: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(arr, a_min=threshold, a_max=None)


def calculate_pvalue_mean_and_error(row: pd.Series) -> pd.Series:
    all_values = []
    for cell in row.dropna():
        if isinstance(cell, (list, np.ndarray, pd.Series)):
            all_values.extend(cell)
        else:
            all_values.append(cell)
    all_values = np.array(all_values, dtype=np.float64)

    if len(all_values) > 0:
        _, p_value = stats.ttest_1samp(all_values, 0)
        mean_value = np.nanmean(all_values)
        error_value = np.nanstd(all_values) / np.sqrt(len(all_values))
        return pd.Series({"p_value": p_value, "nanmean": mean_value, "nan_err": error_value})
    else:
        return pd.Series({"p_value": np.nan, "nanmean": np.nan, "nan_err": np.nan})


def epi_to_matrix(df: pd.DataFrame, char_col="Epi_char", pos_col="Epi_pos0", value_col="E_partial") -> pd.DataFrame:
    d = df.copy()
    d[char_col] = d[char_col].replace({"b": "*", "B": "*"})
    mat = d.pivot_table(index=char_col, columns=pos_col, values=value_col, aggfunc="first")
    mat = mat.reindex(sorted(d[pos_col].unique()), axis=1)
    chars = list(d[char_col].unique())
    if "*" in chars:
        ordered_rows = ["*"] + sorted([c for c in chars if c != "*"])
        mat = mat.reindex(ordered_rows)
    else:
        mat = mat.sort_index()
    return mat


# --- decoding helpers copied from notebook cell (made parameterized) ---

def learn_positionwise_map_strict(
    df: pd.DataFrame,
    *,
    ref_col: str = "pep_",
    enc_col: str = "pep_encoded",
    placeholders: set[str] | None = None,
):
    if placeholders is None:
        # default: treat digits as placeholders
        placeholders = set("0123456789")

    pos_map = defaultdict(dict)
    max_len = 0
    conflicts = []

    for enc, ref in zip(df[enc_col].astype(str), df[ref_col].astype(str)):
        n = len(enc) if len(enc) < len(ref) else len(ref)
        if n > max_len:
            max_len = n
        for i in range(n):
            sym = enc[i]
            if sym in placeholders:
                aa = ref[i]
                if sym in pos_map[i] and pos_map[i][sym] != aa:
                    conflicts.append((i, sym, pos_map[i][sym], aa))
                else:
                    pos_map[i][sym] = aa

    if conflicts:
        msg = ["Conflicts found:"]
        for i, sym, prev, new in conflicts[:10]:
            msg.append(f" pos {i}, '{sym}': {prev} vs {new}")
        raise ValueError("\n".join(msg))

    return pos_map, max_len


def decode_with_position_map(
    series: pd.Series,
    pos_map,
    *,
    expected_len: int | None = None,
    unknown_char: str = "?",
    placeholders: set[str] | None = None,
) -> pd.Series:
    if placeholders is None:
        placeholders = set("0123456789")

    decoded = []
    for enc in series.astype(str):
        n = expected_len if expected_len and expected_len < len(enc) else len(enc)
        out = []
        for i in range(n):
            ch = enc[i]
            if ch in placeholders:
                out.append(pos_map.get(i, {}).get(ch, unknown_char))
            else:
                out.append(ch)
        if len(enc) > n:
            out.append(enc[n:])
        decoded.append("".join(out))
    return pd.Series(decoded, index=series.index)

