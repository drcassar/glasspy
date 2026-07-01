import numpy as np
import pandas as pd
from glasspy.predict.models import VITRIFY, GlassNet, ViscNet

data = [
    [1, 0, 2],
    [0, 1, 2],
    [1, 1, 2],
]

comp1 = "SiO2"
comp2 = "Li2O(SiO2)2"
comp3 = {
    "SiO2": 2,
    "Li2O": 1,
}
comp4 = pd.DataFrame(data, columns=["Li2O", "Na2O", "SiO2"])

all_comps = [comp1, comp2, comp3, comp4]


def test_viscnet():
    model = ViscNet()
    for comp in all_comps:
        log10_viscosity = model.predict(T=1000, composition=comp)
        assert isinstance(log10_viscosity, np.ndarray)


def test_glassnet():
    model = GlassNet()
    for comp in all_comps:
        prediction = model.predict(comp)
        assert isinstance(prediction, pd.DataFrame)


def test_vitrify():
    model = VITRIFY()
    for comp in all_comps:
        prediction = model.predict(comp)
        assert isinstance(prediction, np.ndarray)
