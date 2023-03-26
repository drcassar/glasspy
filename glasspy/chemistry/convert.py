"""Conversion and rescaling of objects containing chemical information.

This module offers some helper functions to convert and rescale objects that
hold chemical information, ChemArrays.

See the function to_array for an easy way to convert strings, dictionaries, and
pandas DataFrames to ChemArray.

Check the function wt_to_mol and mol_to_wt to easily convert a ChemArray from
wt% to mol% and vice versa.

"""
from typing import Union, List, Dict, Tuple

from chemparse import parse_formula
import numpy as np
import pandas as pd

from .types import CompositionLike, ChemArray
from .data import elementmass


def rescale_array(
    x: Union[np.ndarray, ChemArray],
    rescale_to_sum: Union[float, int, bool] = 100,
) -> Union[np.ndarray, ChemArray]:
    """Rescale all rows of an array to have the same sum.

    This function does nothing if rescale_to_sum is False.

    Args:
      x:
        A 2D array. Each row is a chemical substance. See the docstring of the
        function to_array for more information.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.

    Raises:
      AssertionError:
        Raised when rescale_to_sum is negative.
    """

    if rescale_to_sum:
        assert rescale_to_sum > 0, "Negative sum value to normalize"
        sum_ = x.sum(axis=1)
        sum_[sum_ == 0] = 1
        x = x * rescale_to_sum / sum_.reshape(-1, 1)
    return x


def to_array(
    x: CompositionLike,
    input_cols: List[str] = [],
    output_cols: Union[str, List[str]] = "default",
    rescale_to_sum: Union[float, int, bool] = False,
) -> ChemArray:
    """Convert the input object to an array.

    Most of the operations in this module receive an array as the argument. This
    design choice was made because numpy arrays are fast. These arrays may be
    called "chemical arrays", but they are still instances of numpy array.
    Chemical arrays must follow three rules:
      i. each row of the array is one chemical substance;
      ii. each column of the array represents a chemical element or chemical
        molecule;
      iii. chemical arrays must be 2D arrays.

    Say, for example, that you have a chemical array x. The value stored in
    x[i][j] is the amount of the element/molecule j that the substance i has. It
    is up to the user to define what this amount means. Perhaps it is the mole
    fraction of the element/molecule j; perhaps it is the weight percentage of
    the element/molecule j. It doesn't matter what it means, as long as this
    definition is the same for all values stored in this chemical array x. Check
    the function wt_to_mol and mol_to_wt to easily convert array from wt% to
    mol% and vice versa. Note: even if your array hold only one substance, it
    must still meet condition iii, that is: it must still be a 2D array (in this
    case with only one row).

    Args:
      x:
        Any composition like object.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x. Necessary only when x is a list or array, ignored otherwise.
      output_cols:
        List of strings of chemical compounds or chemical elements. The columns
        of the output array will be arranged in this order. If 'default' is
        passed to this argument, then the output array will have the same order
        as the input. Caution: this function does not convert compounds to
        chemical elements, see to_element_array if this is what you are looking
        for.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      The converted ChemArray.

    Raises:
      ValueError:
        Raised when x is a list or an array that is not 1D or 2D.
    """

    if isinstance(x, list):
        x = np.array(x)
        cols = input_cols

    elif isinstance(x, str):
        x = parse_formula(x)
        cols = list(x.keys())
        x = np.array(list(x.values())).T

    elif isinstance(x, dict):
        cols = list(x.keys())
        x = np.array(list(x.values())).T

    elif isinstance(x, pd.DataFrame):
        cols = x.columns.tolist()
        x = x.values

    elif isinstance(x, ChemArray):
        cols = x.cols

    else:
        cols = input_cols

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    elif len(x.shape) != 2:
        raise ValueError("x must be a 1D or a 2D array")

    x = rescale_array(x, rescale_to_sum)

    if output_cols != "default":
        x_reorder = np.zeros((len(x), len(output_cols)))
        idx = [
            (i, cols.index(c)) for i, c in enumerate(output_cols) if c in cols
        ]
        for i, j in idx:
            x_reorder[:, i] = x[:, j]
        x, cols = x_reorder, output_cols

    return ChemArray(x, cols)


def to_element_array(
    x: CompositionLike,
    input_cols: List[str] = [],
    output_element_cols: Union[str, List[str]] = "default",
    rescale_to_sum: Union[float, int, bool] = False,
) -> ChemArray:
    """Convert x to an element array.

    Args:
      x:
        Any composition like object.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x. Necessary only when x is a list or array, ignored otherwise.
      output_element_cols:
        List of strings of chemical element symbols. The columns of the output
        array will be arranged in this order. If 'all' is passed to this
        argument, then the output array will be a sequence from hydrogen to
        plutonium sorted by chemical number. If 'default' is passed, then the
        output array will have only the chemical elements that are present in x,
        sorted alphabetically.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      The converted ChemArray.

    Raises:
      AssertionError:
        Raised when input_cols has lenght of zero and x is a list or an array
      AssertionError:
        Raised when the lenght of input_cols is different than the lenght of x
        along the column axis.
    """

    x = to_array(x, input_cols)

    assert len(x.cols) > 0, "You forgot to pass the list of input_cols."
    assert len(x.cols) == x.shape[1], "Invalid lenght of input_cols."

    if output_element_cols == "default":
        output_element_cols = list(
            sorted(set([el for c in x.cols for el in parse_formula(c)]))
        )
    elif output_element_cols == "all":
        output_element_cols = list(elementmass.keys())

    x_element = np.zeros((len(x), len(output_element_cols)))
    for i, c in enumerate(x.cols):
        for el, n in parse_formula(c).items():
            x_element[:, output_element_cols.index(el)] += x[:, i] * n
    x_element = rescale_array(x_element, rescale_to_sum)

    return ChemArray(x_element, output_element_cols)


def wt_to_mol(
    x: Union[np.ndarray, ChemArray],
    input_cols: List[str],
    rescale_to_sum: Union[float, int, bool] = False,
) -> Union[np.ndarray, ChemArray]:
    """Convert an array from weight% to mol%.

    Args:
      x:
        A 2D array. Each row is a chemical substance. See the docstring of the
        function to_array for more information.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.
    """

    inv_molar_mass = np.diag(
        [
            1 / sum([elementmass[el] * n for el, n in parse_formula(comp).items()])
            for comp in input_cols
        ]
    )
    x = x @ inv_molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x


def mol_to_wt(
    x: Union[np.ndarray, ChemArray],
    input_cols: List[str],
    rescale_to_sum: Union[float, int, bool] = False,
) -> Union[np.ndarray, ChemArray]:
    """Convert an array from mol% to weight%.

    Args:
      x:
        A 2D array. Each row is a chemical substance. See the docstring of the
        function to_array for more information.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.
    """

    molar_mass = np.diag(
        [
            sum([elementmass[el] * n for el, n in parse_formula(comp).items()])
            for comp in input_cols
        ]
    )
    x = x @ molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x
