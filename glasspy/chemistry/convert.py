import re
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from mendeleev import element
from chemparse import parse_formula


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

_elementmass = {el: element(el).mass for el in _allelements}

composition_like = Union[
    str,                     # 'SiO2'
    List[float],             # [0.3, 0.7], the column labels must also be given
    List[List[float]],       # same as above, but with more than one composition
    np.ndarray,              # same as above, can be 1d or 2d      
    Dict[str, float],        # {'SiO2': 0.3, 'Li2O': 0.7}
    Dict[str, List[float]],  # {'SiO2': [0.3, 0.5], 'LiO2': [0.7, 0.5]}
    Dict[str, np.ndarray],   # same as above
    pd.DataFrame,            # can only have chemical column labels
]


def rescale_array(
        array: np.ndarray,
        rescale_to_sum: Union[float,bool] = 100,
) -> np.ndarray:
    '''Rescale all rows of an array to have the same sum.

    '''
    if rescale_to_sum:
        assert rescale_to_sum > 0, 'Negative sum value to normalize'
        sum_ = array.sum(axis=1)
        sum_[sum_ == 0] = 1
        array = array * rescale_to_sum / sum_.reshape(-1,1)
    return array


def wt_to_mol(
        x: np.ndarray,
        input_cols: List[str],
        rescale_to_sum: Union[float,bool] = False,
) -> np.ndarray:
    '''Convert an array from wt% to mol%.

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
        rescale_to_sum: Union[float,bool] = False,
) -> np.ndarray:
    '''Convert an array from mol% to wt%.

    '''
    molar_mass = np.diag([
        sum([_elementmass[el] * n for el, n in parse_formula(comp).items()])
        for comp in input_cols
    ])
    x = x @ molar_mass
    x = rescale_array(x, rescale_to_sum)
    return x

def to_array(x: Union[str, dict, pd.DataFrame]):
    '''Convert x to an array.

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

