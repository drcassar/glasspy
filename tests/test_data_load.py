import pandas as pd
from glasspy.data.load import SciGlass


def test_sciglass_dataload():
    sg = SciGlass()
    assert isinstance(sg.data, pd.DataFrame)
