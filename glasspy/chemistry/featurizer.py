from itertools import product
from typing import List, Tuple, Union
from copy import deepcopy
import os

import numpy as np

from .convert import to_element_array
from .types import CompositionLike


__cur_path = os.path.dirname(__file__)
__chem_prop_path = os.path.join(__cur_path, "data/chemical_properties.csv")

__num_chem_features = 86

_elements = np.genfromtxt(
    __chem_prop_path,
    skip_header=1,
    delimiter=",",
    usecols=0,
    dtype=str,
)

_prop = np.genfromtxt(
    __chem_prop_path,
    delimiter=",",
    usecols=list(range(1, __num_chem_features + 1)),
    dtype=str,
    max_rows=1,
)

_data = np.genfromtxt(
    __chem_prop_path,
    skip_header=1,
    delimiter=",",
    usecols=list(range(1, __num_chem_features + 1)),
)

_all_aggregate_functions = ["sum", "mean", "min", "max", "std", "std1"]

prop_idx = {p: i for i, p in enumerate(_prop)}

all_features = [
    (p, a) for p, a in product(_prop, _all_aggregate_functions)
]


def _aggregate(array: np.array, function_name: str) -> np.ndarray:
    """Apply an aggregator function to an array.

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
    """

    if function_name == "sum":
        return np.sum(array, axis=1)
    elif function_name == "mean":
        return np.mean(array, axis=1)
    elif function_name == "max":
        return np.max(array, axis=1)
    elif function_name == "min":
        return np.min(array, axis=1)
    elif function_name == "std":
        return np.std(array, axis=1, ddof=0)
    elif function_name == "std1":
        return np.std(array, axis=1, ddof=1)
    else:
        raise ValueError("Invalid function name")


def physchem_featurizer(
    x: CompositionLike,
    input_cols: List[str] = [],
    elemental_features: List[str] = [],
    weighted_features: List[Tuple[str, str]] = [],
    absolute_features: List[Tuple[str, str]] = [],
    rescale_to_sum: Union[float, int, bool] = 1,
    sep: str = "|",
    check_invalid: bool = True,
    order: str = "ewa",
) -> Tuple[np.ndarray, List[str]]:
    """Extract features from a chemical object.

    For a list of all possible features that can be extracted, check the
    variable `all_features`.

    Args:
      x:
        Any composition like object.
      input_cols:
        List of strings representing the chemical entities related to each
        column of x. Necessary only when x is a list or array, ignored otherwise.
      absolute_features:
        List of chemical elements that will be part of the features.
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
      check_invalid:
        Checks if there are invalid features that cannot be computed. Invalid
        features are those with missing values in the desired chemical domain.
        The function still works even with invalid features. However, it is not
        recommended to use it in this case. Only disable this check if you are
        sure that no invalid features exist in your chemical domain.
      order:
        String containing the order of the features in the final array. Use the
        letter `e` for elemental features, `w` for weighted features, and `a`
        for absolute features.

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
      ValueError:
        Raised when invalid features are present and check_invalid is True.
    """

    msg = '"rescale_to_sum" must be a positive number, try 1 or 100'
    assert rescale_to_sum > 0, msg

    o_elements = to_element_array(
        x, input_cols, output_element_cols="default"
    ).cols

    if not set(o_elements).issubset(set(_elements)):
        outofdomain = set(o_elements) - set(_elements)
        raise ValueError(
            f"Cannot featurize compositions with these elements: {outofdomain}"
        )

    if check_invalid:
        unavailable_features = []
        el_idx = tuple([i for i, v in enumerate(_elements) if v in o_elements])
        for feat, _ in weighted_features:
            if any(np.isnan(_data[el_idx, prop_idx[feat]])):
                unavailable_features.append(feat)
        for feat, _ in absolute_features:
            if any(np.isnan(_data[el_idx, prop_idx[feat]])):
                unavailable_features.append(feat)
        if len(unavailable_features) > 0:
            raise ValueError(f"Invalid features: {set(unavailable_features)}")

    chemarray = to_element_array(
        x, input_cols, list(_elements), rescale_to_sum
    )

    # Weighted wfeatures
    if "w" in order:
        pos = 0
        wfeatures = np.empty(
            (len(chemarray), len(weighted_features)),
            dtype=float,
        )
        wfeatures.fill(np.nan)
        chemarray_ = np.ma.masked_equal(chemarray, 0)
        feats = {feat for feat, _ in weighted_features}
        cache = {feat: chemarray_ * _data[:, prop_idx[feat]] for feat in feats}
        for feat, stat in weighted_features:
            donthavenan = np.logical_not(np.isnan(cache[feat]).sum(axis=1))
            wfeatures[donthavenan, pos] = _aggregate(
                cache[feat][donthavenan], stat
            )
            pos += 1
        wfeatures_columns = [
            f"W{sep}{feat}{sep}{stat}" for feat, stat in weighted_features
        ]

    # Absolute afeatures
    if "a" in order:
        pos = 0
        afeatures = np.empty(
            (len(chemarray), len(absolute_features)),
            dtype=float,
        )
        afeatures.fill(np.nan)
        chemarray_ = np.ma.masked_equal(chemarray, 0)
        chemarray_ = chemarray_.astype(bool)
        feats = {feat for feat, _ in absolute_features}
        cache = {feat: chemarray_ * _data[:, prop_idx[feat]] for feat in feats}
        for feat, stat in absolute_features:
            donthavenan = np.logical_not(np.isnan(cache[feat]).sum(axis=1))
            afeatures[donthavenan, pos] = _aggregate(
                cache[feat][donthavenan], stat
            )
            pos += 1
        afeatures_columns = [
            f"A{sep}{feat}{sep}{stat}" for feat, stat in absolute_features
        ]

    # Elemental features
    if "e" in order:
        els = list(_elements)
        colidx = [els.index(e) for e in elemental_features]
        efeatures = chemarray[:, colidx]
        efeatures_columns = deepcopy(elemental_features)

    if order.startswith("e"):
        features = efeatures
        feature_columns = efeatures_columns
    elif order.startswith("w"):
        features = wfeatures
        feature_columns = wfeatures_columns
    elif order.startswith("a"):
        features = afeatures
        feature_columns = afeatures_columns
    else:
        raise ValueError("Invalid value for the `order` argument.")

    for letter in order[1:]:
        if letter == "e":
            features = np.hstack((features, efeatures))
            feature_columns.extend(efeatures_columns)
        elif letter == "w":
            features = np.hstack((features, wfeatures))
            feature_columns.extend(wfeatures_columns)
        elif letter == "a":
            features = np.hstack((features, afeatures))
            feature_columns.extend(afeatures_columns)
        else:
            raise ValueError("Invalid value for the `order` argument.")

    return features, feature_columns
