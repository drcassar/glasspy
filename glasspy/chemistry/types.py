"""Provides CompositionLike to check if an object is a valid chemical object and
the ChemArray class.

You can represent a chemical substance in many ways using Python objects.
GlassPy accepts 8 different types:

* String: any string that can be parsed by the parse_formula of chemparse
  (https://pypi.org/project/chemparse/) is allowed. Examples: 'SiO2',
  'CaMgSi2O5', '(Li2O)1(SiO2)2', 'C1.5O3'. Check the `chemparse` documentation for
  additional information.

* List of floats: a simple list of floats can represent a chemical substance. To
  represent SiO2 one could write [1, 2, 0] with the first element representing
  the amount of silicon, the second the amount of oxygen, and the third element
  the amount of lithium. Another way to represent this substance would be the
  list [1], where the only element present is silica itself. As you can see, the
  knowledge of which element of the list is associated with each chemical
  substance is left to the user.

* List of lists of floats: similar to the item above, here the user can store
  more than one chemical substance in the same variable. To represent both SiO2
  and Li2O one can write 'substances = [[1, 2, 0], [0, 1, 2]]'. Observe that the
  first index of substances is associated with an individual chemical substance,
  so substances[0] contains information on SiO2 and substances[1] contains
  information on Li2O. The second index of substances is associated with
  chemical elements or molecules. In this case, the values stored in
  substances[0][0] and substances[1][0] store the amount of silicon that SiO2
  and Li2O have.

* Numpy array: 1D numpy arrays follow the same logic as the list of floats and
  2D numpy arrays follow the same logic as list of lists of float.

* Dictionary with string keys and float values: in this case, the keys are the
  chemical elements or molecules and the associated values are the amount of
  these elements or molecules. SiO2 can be writen as {'Si': 1, 'O': 2} or
  {'SiO2': 1}.

* Dictionary with string keys and list of floats values: similar to the item
  above, but the user can store more than one chemical substance in the same
  dictionary. To store SiO2 and Li2O in the same dictionary you can write
  {'Si': [1, 0], 'O': [2, 1], 'Li': [0, 2]}. Observe that the first element of
  each list stores information of one substance (SiO2) and the second element
  stores information of the other (Li2O).

* Dictionary with string keys and numpy array values: behaves the same as the
  item above.

* Pandas DataFrame: each row of the DataFrame is considered one chemical
  substance. The columns represent the chemical elements or molecules that make
  the substances. Elements and molecules that are not present must be zero. Only
  information related to the chemical composition of the substances can be
  present in the DataFrame.

"""
from typing import Union, List, Dict

import numpy as np
import pandas as pd


class ChemArray(np.ndarray):
    """Numpy array for storing chemical composition data.

    ChemArrays must follow three rules:
      i. each row of the array is one chemical substance;
      ii. each column of the array represents a chemical element or chemical
          molecule;
      iii. ChemArrays must be 2D arrays.

    Say, for example, that you have a ChemArray x. The value stored in x[i][j]
    is the amount of the element/molecule j that the substance i has. It is up to
    the user to define what this amount means. Perhaps it is the mole fraction of
    the element/molecule j; perhaps it is the weight percentage of the
    element/molecule j. It doesn't matter what it means, as long as this definition
    is the same for all values stored in this ChemArray.

    Note that even if your array holds only one substance, it must still meet
    condition iii, that is: it must still be a 2D array (in this case, with only
    one row). The recommended way to create ChemArrays is using the function
    `to_array` or `to_element_array` from GlassPy's `chemistry.convert`
    submodule.

    Args:
      chem_composition:
        A 2D array. Each row is a chemical substance. See the docstring of the
        function to_array for more information.
      chem_columns:
        A list of strings containing the chemical substance related to each
        column of the array.

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
