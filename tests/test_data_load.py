import pandas as pd
from glasspy.data.load import SciGlass
from glasspy.predict.base import _create_data_glassnet, _load_data_glassnet
from pandas.testing import assert_frame_equal


def test_sciglass_dataload():
    sg = SciGlass()
    assert isinstance(sg.data, pd.DataFrame)


def test_glasspy_data_creator():
    loaded_df = _load_data_glassnet()
    created_df = _create_data_glassnet()
    # So close... I have no clue why it is not the same...
    assert_frame_equal(created_df, loaded_df)
