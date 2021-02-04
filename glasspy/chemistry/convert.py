"""Convertion and rescaling of objects containing chemical information.

This module offers some helper functions to convert and rescale objects that
hold chemical information. 

  Typical usage example:

  TODO
  foo = ClassFoo()
  bar = foo.FunctionBar()

"""
from typing import Union, List, Dict, Tuple
import re

from chemparse import parse_formula
import numpy as np
import pandas as pd

from .types import composition_like


_digits = re.compile('\d')

_allelements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
                'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
                'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

_elementmass = {
    'H': 1.008,
    'He': 4.002602,
    'Li': 6.94,
    'Be': 9.0121831,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998403163,
    'Ne': 20.1797,
    'Na': 22.98976928,
    'Mg': 24.305,
    'Al': 26.9815385,
    'Si': 28.085,
    'P': 30.973761998,
    'S': 32.06,
    'Cl': 35.45,
    'Ar': 39.948,
    'K': 39.0983,
    'Ca': 40.078,
    'Sc': 44.955908,
    'Ti': 47.867,
    'V': 50.9415,
    'Cr': 51.9961,
    'Mn': 54.938044,
    'Fe': 55.845,
    'Co': 58.933194,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.63,
    'As': 74.921595,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.4678,
    'Sr': 87.62,
    'Y': 88.90584,
    'Zr': 91.224,
    'Nb': 92.90637,
    'Mo': 95.95,
    'Tc': 97.90721,
    'Ru': 101.07,
    'Rh': 102.9055,
    'Pd': 106.42,
    'Ag': 107.8682,
    'Cd': 112.414,
    'In': 114.818,
    'Sn': 118.71,
    'Sb': 121.76,
    'Te': 127.6,
    'I': 126.90447,
    'Xe': 131.293,
    'Cs': 132.90545196,
    'Ba': 137.327,
    'La': 138.90547,
    'Ce': 140.116,
    'Pr': 140.90766,
    'Nd': 144.242,
    'Pm': 144.91276,
    'Sm': 150.36,
    'Eu': 151.964,
    'Gd': 157.25,
    'Tb': 158.92535,
    'Dy': 162.5,
    'Ho': 164.93033,
    'Er': 167.259,
    'Tm': 168.93422,
    'Yb': 173.045,
    'Lu': 174.9668,
    'Hf': 178.49,
    'Ta': 180.94788,
    'W': 183.84,
    'Re': 186.207,
    'Os': 190.23,
    'Ir': 192.217,
    'Pt': 195.084,
    'Au': 196.966569,
    'Hg': 200.592,
    'Tl': 204.38,
    'Pb': 207.2,
    'Bi': 208.9804,
    'Po': 209.0,
    'At': 210.0,
    'Rn': 222.0,
    'Fr': 223.0,
    'Ra': 226.0,
    'Ac': 227.0,
    'Th': 232.0377,
    'Pa': 231.03588,
    'U': 238.02891,
    'Np': 237.0,
    'Pu': 244.0,
}


def rescale_array(
        x: np.ndarray,
        rescale_to_sum: Union[float,int,bool] = 100,
) -> np.ndarray:
    '''Rescale all rows of an array to have the same sum.

    This function does nothing if rescale_to_sum is False.

    Args:
      x:
        A 2D array. Each row is a chemical substance.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.

    '''
    if rescale_to_sum:
        assert rescale_to_sum > 0, 'Negative sum value to normalize'
        sum_ = x.sum(axis=1)
        sum_[sum_ == 0] = 1
        x = x * rescale_to_sum / sum_.reshape(-1,1)
    return x


def wt_to_mol(
        x: np.ndarray,
        input_cols: List[str],
        rescale_to_sum: Union[float,int,bool] = False,
) -> np.ndarray:
    '''Convert an array from weight% to mol%.

    Args:
      x:
        A 2D array. Each row is a chemical substance.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.

    '''
    inv_molar_mass = np.diag([
        1 / sum([_elementmass[el] * n for el, n in parse_formula(comp).items()])
        for comp in input_cols
    ])
    x = x @ inv_molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x


def mol_to_wt(
        x: np.ndarray,
        input_cols: List[str],
        rescale_to_sum: Union[float,int,bool] = False,
) -> np.ndarray:
    '''Convert an array from mol% to weight%.

    Args:
      x:
        A 2D array. Each row is a chemical substance.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.

    Returns:
      A 2D array. Each row is a chemical substance.

    '''
    molar_mass = np.diag([
        sum([_elementmass[el] * n for el, n in parse_formula(comp).items()])
        for comp in input_cols
    ])
    x = x @ molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x


def to_array(x: Union[str, dict, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
    '''Convert the input object to an array.

    The input object can be a string, a dictionary, or a pandas DataFrame.
    Examples of possible strings are: "SiO2" or "Li2O(SiO2)2".
    Example of a dictionary is: {"SiO2": 2, "LiO2": 1}.
    Pandas DataFrames must only contain chemical information. The columns must
    contain only valid chemical strings, they can be either individual chemical
    elements or any valid chemical substance.

    Args:
      x:
        The chemical object can be either a string, a dictionary, or a pandas
        DataFrame. See the docstring for more info.

    Returns:
      A tuple containing the 2D array (where every line is a chemical substance)
      and a list of strings containing the chemical substance related to each
      column of the array.

    '''
    if isinstance(x, str):
        x = parse_formula(x)
        cols = list(x.keys())
        x = np.array(list(x.values())).T

    elif isinstance(x, dict):
        cols = list(x.keys())
        x = np.array(list(x.values())).T

    elif isinstance(x, pd.DataFrame):
        cols = x.columns.tolist()
        x = x.values

    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    elif len(x.shape) != 2:
        raise ValueError('x must be a 1D or a 2D array')

    return x, cols


def any_to_element_array(
        x: composition_like,
        input_cols: List[str] = [],
        output_element_cols='default',
        rescale_to_sum=False,
):
    '''Convert x to an element array.

    Parameters
    ----------
    input_cols : list of the column labels of the input data. Necessary only
        when x is a list or array. Ignored otherwise.

    '''
    if isinstance(x, list):
        x = np.array(x)

    if not isinstance(x, np.ndarray):
        x, input_cols = to_array(x)
    else:
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        elif len(x.shape) != 2:
            raise ValueError('x must be a 1D or a 2D array')

    assert len(input_cols) > 0, 'You forgot to pass the list of input_cols.'
    assert len(input_cols) == x.shape[1], 'Invalid lenght of input_cols.'

    if output_element_cols == 'default':
        output_element_cols = _allelements

    x_element = np.zeros((len(x), len(output_element_cols)))
    for i, c in enumerate(input_cols):
        for el, n in parse_formula(c).items():
            x_element[:,output_element_cols.index(el)] += x[:,i] * n
    x_element = rescale_array(x_element, rescale_to_sum)

    return x_element, output_element_cols

