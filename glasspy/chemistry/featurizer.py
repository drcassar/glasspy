from itertools import product
from typing import List, Tuple, Union
import os

import numpy as np

from .convert import any_to_element_array
from .types import CompositionLike


__cur_path = os.path.dirname(__file__)
__chem_prop_path = os.path.join(__cur_path, 'data/chemical_properties.csv')

__num_chem_features = 56

_elements = np.genfromtxt(
    __chem_prop_path,
    skip_header=1,
    delimiter=',',
    usecols=0,
    dtype=str,
)


_prop = np.genfromtxt(
    __chem_prop_path,
    delimiter=',',
    usecols=list(range(1, __num_chem_features + 1)),
    dtype=str,
    max_rows=1,
)

_data = np.genfromtxt(
    __chem_prop_path,
    skip_header=1,
    delimiter=',',
    usecols=list(range(1, __num_chem_features + 1)),
    dtype=[(p, 'float') for p in _prop],
)

_all_aggregate_functions = ['sum', 'mean', 'min', 'max', 'std']

all_chem_features = [(p,a) for p,a in product(_prop, _all_aggregate_functions)]

def _aggregate(array: np.array, function_name: str) -> np.ndarray:
    '''Apply an aggregator function to an array.

    Args:
      array:
        Array to which the aggregator function will by applied along the column
        axis.
      function_name:
        Name of the aggregator function to be applied on the array. Possible
        names are mean, sum for summation, min for minimum, max for maximum, and
        std for standard deviation.

    Returns:
      Array with the aggregation function applied.

    Raises:
      ValueError:
        Raises this error if the function_name is not valid.

    '''
    if function_name == 'sum':
        return np.nansum(array, axis=1)
    elif function_name == 'mean':
        return np.nanmean(array, axis=1)
    elif function_name == 'max':
        return np.nanmax(array, axis=1)
    elif function_name == 'min':
        return np.nanmin(array, axis=1)
    elif function_name == 'std':
        return np.nanstd(array, axis=1)
    else:
        raise ValueError('Invalid function name')


def extract_chem_feats(
        x: CompositionLike,
        input_cols: List[str] = [],
        absolute_features: List[Tuple[str,str]] = [],
        weighted_features: List[Tuple[str,str]] = [],
        rescale_to_sum: Union[float,int,bool] = 1,
        sep: str = '|',
) -> Tuple[np.ndarray, List[str]]:
    '''Extract chemical features from a chemical object.

    For a list of all possible features that can be extracted, check the
    variable all_chem_features.

    Args:
      x:
        Any composition like object.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x. Necessary only when x is a list or array, ignored otherwise.
      absolute_features:
        List of tuples containing the name of the feature to be extracted and
        the aggregator function. Features computed from this list are absolute.
      weighted_features:
        List of tuples containing the name of the feature to be extracted and
        the aggregator function. Features computed from this list are weighted.
      rescale_to_sum:
        A positive number representing the total sum of each chemical substance.
        If False then the same input array is returned.
      sep:
        String used to separate the information of the name of each extracted
        feature.

    Returns:
      features:
        A 2D array. Each row is a chemical substance.
      feature_columns;
        A list of strings containing the name of the extracted chemical feature.
        Strings starting with "A" are absolute features and strings starting
        with "W" are weighted.

    Raises:
      AssertionError:
        Raised when rescale_to_sum is negative.
      ValueError:
        Raised when the input composition has chemical elements that cannot be
        used to extract features.

    '''
    msg = '"rescale_to_sum" must be a positive number, try 1 or 100'
    assert rescale_to_sum > 0, msg

    if len(input_cols) > 0 and not set(input_cols).issubset(set(_elements)):
        outofdomain = set(input_cols) - set(_elements)
        raise ValueError(
            f'Cannot featurize compositions with these elements: {outofdomain}'
        )

    array, _ = any_to_element_array(x, input_cols, list(_elements),
                                    rescale_to_sum)

    pos = 0
    feature_columns = []
    features = np.zeros(
        (len(array), len(absolute_features) + len(weighted_features)),
        dtype=float,
    )

    array[~(array > 0)] = np.nan
    for feat, stat in weighted_features:
        features[:,pos] = _aggregate(array * _data[feat], stat)
        feature_columns.append(f'W{sep}{feat}{sep}{stat}')
        pos += 1

    array[array > 0] = 1
    for feat, stat in absolute_features:
        features[:,pos] = _aggregate(array * _data[feat], stat)
        feature_columns.append(f'A{sep}{feat}{sep}{stat}')
        pos += 1

    return features, feature_columns
