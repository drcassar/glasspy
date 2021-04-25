"""Provides CompositionLike to check if an object is a valid chemical object.

You can represent a chemical substance in many ways using Python objects.
GlassPy accepts 8 different:

* String: any string that can be parsed by the parse_formula of chemparse
  (https://pypi.org/project/chemparse/) is allowed. Examples: 'SiO2',
  'CaMgSi2O5', '(Li2O)1(SiO2)2', 'C1.5O3'. Check the chemparse documentation for
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


CompositionLike = Union[
    str,
    List[float],
    List[List[float]],
    np.ndarray,
    Dict[str, float],
    Dict[str, List[float]],
    Dict[str, np.ndarray],
    pd.DataFrame,
]
