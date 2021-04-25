"""Convertion and rescaling of objects containing chemical information.

This module offers some helper functions to convert and rescale objects that
hold chemical information. 

Most of the operations in this module recieve an array as the argument. This
design choice was made because numpy arrays are fast. These arrays may be called
"chemical arrays", but they are still instances of numpy array.  Chemical arrays
must follow three rules:
    i. each row of the array is one chemical substance;
    ii. each column of the array represents a chemical element or chemical
    molecule;
    iii. chemical arrays must be 2D arrays.

Say, for example, that you have a chemical array x. The value stored in x[i][j]
is the amount of the element/molecule j that the substance i has. It is up to
the user to define what this amount means. Perhaps it is the mole fraction of
the element/molecule j; perhaps it is the weight percentage of the
element/molecule j. It doesn't matter what it means, as long as this definition
is the same for all values stored in this chemical array x. Check the function
wt_to_mol and mol_to_wt to easily convert array from wt% to mol% and vice versa.
Note: even if your array hold only one substance, it must still meet condition
iii, that is: it must still be a 2D array (in this case with only one row).

See the function to_array for an easy way to convert strings, dictionaries, and
pandas DataFrames to a valid chemical array.

"""
from typing import Union, List, Dict, Tuple
import re

from chemparse import parse_formula
import numpy as np
import pandas as pd

from .types import CompositionLike


_digits = re.compile("\d")

# data from https://github.com/lmmentel/mendeleev/
_elementmass = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 97.90721,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 144.91276,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,
    "Fr": 223.0,
    "Ra": 226.0,
    "Ac": 227.0,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.0,
    "Pu": 244.0,
}

_allelements = list(_elementmass.keys())

import numpy as np


class ChemArray(np.ndarray):
    """Numpy array to use for storing chemical composition data.

    Notes:
      Code based on https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    """

    def __new__(
        cls,
        chem_composition: np.ndarray,
        chem_columns: List[str],
    ):
        obj = np.asarray(chem_composition).view(cls)
        obj.cols = chem_columns
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.cols = getattr(obj, "cols", None)

    def __repr__(self):
        print("ChemArray")
        print(self.cols)
        print(self.view())
        return ""


def rescale_array(
    x: np.ndarray,
    rescale_to_sum: Union[float, int, bool] = 100,
) -> np.ndarray:
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
) -> Tuple[np.ndarray, List[str]]:
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
      x:
        A 2D array. Each row is a chemical substance.
      cols;
        A list of strings containing the chemical substance related to each
        column of the array.

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

    else:
        cols = input_cols

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    elif len(x.shape) != 2:
        raise ValueError("x must be a 1D or a 2D array")

    x = rescale_array(x, rescale_to_sum)

    if output_cols != "default":
        x_reorder = np.zeros((len(x), len(output_cols)))
        idx = [(i, cols.index(c)) for i, c in enumerate(output_cols) if c in cols]
        for i, j in idx:
            x_reorder[:, i] = x[:, j]
        x, cols = x_reorder, output_cols

    return x, cols


def to_element_array(
    x: CompositionLike,
    input_cols: List[str] = [],
    output_element_cols: Union[str, List[str]] = "default",
    rescale_to_sum: Union[float, int, bool] = False,
) -> Tuple[np.ndarray, List[str]]:
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
      x_element:
        A 2D array. Each row is a chemical substance.
      cols;
        A list of strings containing the symbol of chemical elements related to
        each column of the array.

    Raises:
      AssertionError:
        Raised when input_cols has lenght of zero and x is a list or an array
      AssertionError:
        Raised when the lenght of input_cols is different than the lenght of x
        along the column axis.

    """
    x, cols = to_array(x, input_cols)

    assert len(cols) > 0, "You forgot to pass the list of input_cols."
    assert len(cols) == x.shape[1], "Invalid lenght of input_cols."

    if output_element_cols == "default":
        output_element_cols = list(
            sorted(set([el for c in cols for el in parse_formula(c)]))
        )
    elif output_element_cols == "all":
        output_element_cols = _allelements

    x_element = np.zeros((len(x), len(output_element_cols)))
    for i, c in enumerate(cols):
        for el, n in parse_formula(c).items():
            x_element[:, output_element_cols.index(el)] += x[:, i] * n
    x_element = rescale_array(x_element, rescale_to_sum)

    return x_element, output_element_cols


def wt_to_mol(
    x: np.ndarray,
    input_cols: List[str],
    rescale_to_sum: Union[float, int, bool] = False,
) -> np.ndarray:
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
            1 / sum([_elementmass[el] * n for el, n in parse_formula(comp).items()])
            for comp in input_cols
        ]
    )
    x = x @ inv_molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x


def mol_to_wt(
    x: np.ndarray,
    input_cols: List[str],
    rescale_to_sum: Union[float, int, bool] = False,
) -> np.ndarray:
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
            sum([_elementmass[el] * n for el, n in parse_formula(comp).items()])
            for comp in input_cols
        ]
    )
    x = x @ molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x
