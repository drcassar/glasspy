"""Provides CompositionLike to check if an object is a valid chemical object and
the ChemArray class.

There are many ways to represent a chemical substance using Python objects.
GlassPy accepts 8 different types:

* String: Any string that can be parsed by the parse_formula of chemparse
  (https://pypi.org/project/chemparse/) is allowed. Examples: "SiO2",
  "CaMgSi2O5", "(Li2O)1(SiO2)2", "C1.5O3". See the `chemparse` documentation for
  more information.

* List of floats: A simple list of floats can represent a chemical substance. To
  represent SiO2, you could write [1, 2, 0], where the first element represents
  the amount of silicon, the second the amount of oxygen, and the third the
  amount of lithium. Another way to represent this substance would be the list
  [1], where the only element present is silica itself. As you can see, it is up
  to the user to know which element of the list is associated with each chemical
  substance.

* List of lists of floats: similar to the above, here the user can store more
  than one chemical substance in the same variable. To represent both SiO2 and
  Li2O, you can write "substances = [[1, 2, 0], [0, 1, 2]]". Note that the first
  index of substances is associated with a single chemical substance, so
  substances[0] contains information about SiO2 and substances[1] contains
  information about Li2O. The second substance index is associated with chemical
  elements or molecules. In this case, the values stored in substances[0][0] and
  substances[1][0] store the amount of silicon that SiO2 and Li2O have.

* Numpy array: 1D numpy arrays follow the same logic as a list of floats, and 2D
  numpy arrays follow the same logic as a list of lists of floats.

* Dictionary with string keys and float values: in this case, the keys are the
  chemical elements or molecules and the corresponding values are the amount of
  these elements or molecules. SiO2 can be written as {'Si': 1, 'O': 2} or
  {'SiO2': 1}.

* Dictionary with string keys and list of float values: similar to the above,
  but the user can store more than one chemical substance in the same
  dictionary. To store SiO2 and Li2O in the same dictionary, you can write
  {'Si': [1, 0], 'O': [2, 1], 'Li': [0, 2]}. Note that the first element of each
  list stores information about one substance (SiO2) and the second element
  stores information about the other (Li2O).

* Dictionary with string keys and numpy array values: behaves the same as the
  above.

* Pandas DataFrame: each row of the DataFrame represents a chemical substance.
  The columns represent the chemical elements or molecules that make up the
  substance. Elements and molecules that are not present must be zero. Only
  information related to the chemical composition of the substances can be
  present in the DataFrame.
"""

from typing import Union, List, Dict

import numpy as np
import pandas as pd


class ChemArray(np.ndarray):
    """Numpy array for storing chemical composition data.

    ChemArrays must obey three rules:
        i. each row of the array is a chemical substance;
        ii. each column of the array represents a chemical element or molecule;
        iii. ChemArrays must be 2D arrays.

    For example, suppose you have a ChemArray x. The value stored in x[i][j] is
    the amount of element/molecule j that substance i has. It is up to the user
    to define what this amount means. Maybe it's the mole fraction of the
    element/molecule j; maybe it's the weight percentage of the element/molecule
    j. It doesn't matter what it means, as long as this definition is the same
    for all values stored in this ChemArray.

    Note that even if your array contains only one substance, it must still
    satisfy condition iii, that is, it must still be a 2D array (in this case,
    with only one row). The recommended way to create ChemArrays is to use the
    `to_array` or `to_element_array` function from GlassPy's `chemistry.convert`
    submodule.

    Args:
      chem_composition:
        A 2D array. Each row is a chemical element. See the docstring of the
        to_array function for more information.
      chem_columns:
        A list of strings containing the chemical substance associated with each
        column in the array.

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


CompositionLike = Union[
    str,
    List[float],
    List[List[float]],
    np.ndarray,
    Dict[str, float],
    Dict[str, List[float]],
    Dict[str, np.ndarray],
    pd.DataFrame,
    ChemArray,
]
