from typing import Union, List, Dict

import numpy as np
import pandas as pd


CompositionLike = Union[
    str,                     # 'SiO2'
    List[float],             # [0.3, 0.7], the column labels must also be given
    List[List[float]],       # same as above, but with more than one composition
    np.ndarray,              # same as above, can be 1d or 2d      
    Dict[str, float],        # {'SiO2': 0.3, 'Li2O': 0.7}
    Dict[str, List[float]],  # {'SiO2': [0.3, 0.5], 'LiO2': [0.7, 0.5]}
    Dict[str, np.ndarray],   # same as above
    pd.DataFrame,            # can only have chemical column labels
]
