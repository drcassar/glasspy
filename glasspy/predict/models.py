from .base import MLP, MTL
from glasspy.chemistry import physchem_featurizer, CompositionLike
from glasspy.viscosity.equilibrium_log import myega_alt
from glasspy.data import SciGlass

import os
import pickle
from typing import Dict, List, Tuple, NamedTuple, Union, Any
from pathlib import Path
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable

import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from compress_pickle import load
from scipy.stats import theilslopes
from scipy.optimize import least_squares
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# fmt: off
try:
    from glasspy_extra import load_rf_model
    _HAS_GLASSPY_EXTRA = True
except ModuleNotFoundError:
    _HAS_GLASSPY_EXTRA = False
# fmt: on

_BASEMODELPATH = Path(os.path.dirname(__file__)) / "models"

_VISCOSITY_COLUMNS_FOR_REGRESSION = [
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "Viscosity773K",
    "Viscosity873K",
    "Viscosity973K",
    "Viscosity1073K",
    "Viscosity1173K",
    "Viscosity1273K",
    "Viscosity1373K",
    "Viscosity1473K",
    "Viscosity1573K",
    "Viscosity1673K",
    "Viscosity1773K",
    "Viscosity1873K",
    "Viscosity2073K",
    "Viscosity2273K",
    "Tg",
    "TLittletons",
]


def _myega_residuals(x, T, y):
    return myega_alt(T, *x) - y


class _BaseViscNet(MLP):
    """Base class for creating ViscNet-like models.

    References:
      [1] D.R. Cassar, ViscNet: Neural network for predicting the fragility
        index and the temperature-dependency of viscosity, Acta Materialia. 206
        (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
        https://arxiv.org/abs/2007.03719
    """

    def __init__(self, parameters_range, hparams={}, x_mean=0, x_std=1):
        super().__init__(hparams)

        self.hparams = hparams
        self.x_mean = x_mean
        self.x_std = x_mean
        self.parameters_range = parameters_range

        input_dim = int(
            self.hparams.get(
                f'layer_{self.hparams.get("num_layers",1)}_size', 10
            )
        )

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, len(self.parameters_range)),
            nn.Sigmoid(),
        )

    @abstractmethod
    def log_viscosity_fun(self) -> torch.tensor:
        pass

    @abstractmethod
    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ) -> np.ndarray:
        pass

    def viscosity_parameters_from_tensor(
        self,
        feature_tensor,
        return_tensor=False,
    ):
        """Predicts the viscosity parameters from a feature tensor.

        Consider using other methods for predicting the viscosity parameters.
        """

        xf = self.hidden_layers((feature_tensor - self.x_mean) / self.x_std)
        xf = self.output_layer(xf)

        parameters = {}

        for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
            # Scaling the viscosity parameters to be within the parameter range
            parameters[p_name] = torch.add(
                torch.ones(xf.shape[0]).mul(p_range[0]),
                xf[:, i],
                alpha=p_range[1] - p_range[0],
            )

        if not return_tensor:
            parameters = {k: v.detach().numpy() for k, v in parameters.items()}

        return parameters

    def viscosity_parameters(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        return_tensor: bool = False,
    ):
        """Compute the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          return_tensor:
            If `True`, the dictionary values will be torch tensors. If `False`,
            the dictionary values will be numpy arrays.

        Returns:
          Dictionary with the viscosity parameters as keys and the viscosity
          parameters as values.
        """

        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()
        parameters = self.viscosity_parameters_from_tensor(
            features, return_tensor
        )
        return parameters

    def viscosity_parameters_dist(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        num_samples: int = 100,
    ):
        """Compute the distribution of the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Dictionary with the viscosity parameters as keys and an array with the
          the parameters distribution as values.
        """

        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()

        is_training = self.training
        if not is_training:
            self.train()

        dist = defaultdict(list)
        with torch.no_grad():
            for _ in range(num_samples):
                pdict = self.viscosity_parameters_from_tensor(features, False)
                for k, v in pdict.items():
                    dist[k].append(v)

        if not is_training:
            self.eval()

        dist = {k: np.array(v).T for k, v in dist.items()}

        return dist

    def viscosity_parameters_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Compute the confidence bands of the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Dictionary with the viscosity parameters as keys and an array with the
          the confidence band as values.
        """

        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]
        dist = self.viscosity_parameters_dist(
            composition, input_cols, num_samples
        )
        bands = {k: np.percentile(v, q, axis=1).T for k, v in dist.items()}
        return bands

    def forward(self, x):
        """Method used for training the neural network.

        Consider using the other methods for prediction.

        Args:
          x:
            Feature tensor with the last column being the temperature in Kelvin.

        Returns
          Tensor with the viscosity prediction.
        """

        T = x[:, -1].detach().clone()
        parameters = self.viscosity_parameters_from_tensor(x[:, :-1], True)
        log_viscosity = self.log_viscosity_fun(T, **parameters)
        return log_viscosity

    def predict(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the base-10 logarithm of viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the base-10 logarithm of viscosity.
        """

        parameters = self.viscosity_parameters(composition, input_cols, True)
        num_compositions = len(list(parameters.values())[0])

        with torch.no_grad():
            if isinstance(T, Iterable):
                if len(T) == num_compositions and table_mode:
                    T = torch.tensor(T).float()
                    params = [p for p in parameters.values()]
                    log_viscosity = self.log_viscosity_fun(T, *params).numpy()

                else:
                    T = torch.tensor(T).float()
                    params = [
                        p.expand(len(T), len(p)).T for p in parameters.values()
                    ]
                    log_viscosity = self.log_viscosity_fun(T, *params).numpy()

            else:
                T = torch.tensor(T).float()
                params = [p for p in parameters.values()]
                log_viscosity = self.log_viscosity_fun(T, *params).numpy()

        return log_viscosity

    def predict_log10_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the base-10 logarithm of viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the base-10 logarithm of viscosity.
        """

        return self.predict(T, composition, input_cols, table_mode)

    def predict_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the viscosity.
        """

        return 10 ** self.predict(T, composition, input_cols, table_mode)

    def predict_fragility(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the liquid fragility.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the liquid fragility.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        fragility = parameters["m"]
        return fragility

    def predict_Tg(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the glass transition temperature.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the glass transition temperature.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        Tg = parameters["Tg"]
        return Tg

    def predict_log10_eta_infinity(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the base-10 logarithm of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the base-10 logarithm of the asymptotic viscosity.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        log_eta_inf = parameters["log_eta_inf"]
        return log_eta_inf

    def predict_eta_infinity(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the asymptotic viscosity.
        """

        return 10 ** self.predict_log10_eta_infinity(composition, input_cols)

    def predict_Tg_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ):
        """Confidence bands of the glass transition temperature.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the glass transition temperature.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        Tg = parameters_unc["Tg"]
        return Tg

    def predict_fragility_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ):
        """Confidence bands of the liquid fragility.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the liquid fragility.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        m = parameters["m"]
        return m

    def predict_log10_eta_infinity_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Confidence bands of the base-10 logarithm of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the base-10 logarithm of the asymptotic viscosity.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        log_eta_inf = parameters_unc["log_eta_inf"]
        return log_eta_inf

    def predict_viscosity_unc(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
        table_mode: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.array]]:
        """Compute the confidence bands of the viscosity prediction.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          bands:
            Confidence bands of the base-10 logarithm of viscosity. If more than
            one composition was given, then this is a 3-dimension numpy array.
            In this case, the first dimension is the composition, the second is
            the confidence band (lower and higher value, respectively), and the
            third is the temperature. Differently, if only one composition was
            given, then this is a 2-dimension array, with the first dimension
            being the confidence band and the second being the temperature. If
            the number of compositions is the same as the lenght of variable T
            and table_mode is True, then this variable is also a 2-dimension
            array.
          param_bands:
            Dictionary with the uncertainty of the viscosity parameters for each
            composition.
          pdistribution:
            Dictionary with the distribution of the viscosity parameters.
        """

        pdistribution = self.viscosity_parameters_dist(
            composition, input_cols, num_samples
        )
        parameters = []
        for param in self.parameters_range.keys():
            parameters.append(pdistribution[param])
        parameters = np.array(parameters)

        num_compositions = parameters.shape[1]

        if isinstance(T, Iterable):
            if len(T) == num_compositions and table_mode:
                all_curves = np.zeros((1, num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(p).float()
                        for p in parameters[:, :, i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:, :, i] = viscosity

            else:
                all_curves = np.zeros((len(T), num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(
                            np.broadcast_to(p, (len(T), len(p))).T
                        ).float()
                        for p in parameters[:, :, i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:, :, i] = viscosity.T

        else:
            all_curves = np.zeros((1, num_compositions, num_samples))
            T = torch.tensor(T).float()
            for i in range(num_samples):
                params = [
                    torch.from_numpy(p).float() for p in parameters[:, :, i]
                ]
                viscosity = self.log_viscosity_fun(T, *params).numpy()
                all_curves[:, :, i] = viscosity

        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]

        bands = np.percentile(all_curves, q, axis=2)
        bands = np.transpose(bands, (2, 0, 1))

        if num_compositions == 1:
            bands = bands[0, :, :]

        param_bands = {
            k: np.percentile(v, q, axis=1).T for k, v in pdistribution.items()
        }

        return bands, param_bands, pdistribution

    @staticmethod
    def citation(bibtex: bool = False) -> str:
        if bibtex:
            c = "@article{Cassar_2021, title={ViscNet: Neural network for "
            "predicting the fragility index and the temperature-dependency of "
            "viscosity}, volume={206}, ISSN={1359-6454}, "
            "DOI={10.1016/j.actamat.2020.116602}, journal={Acta Materialia}, "
            "author={Cassar, Daniel R.}, year={2021}, month={Mar}, "
            "pages={116602}}"
        else:
            c = "D.R. Cassar, ViscNet: Neural network for predicting the "
            "fragility index and the temperature-dependency of viscosity, "
            "Acta Materialia. 206 (2021) 116602. "
            "https://doi.org/10.1016/j.actamat.2020.116602."
        return c


class ViscNet(_BaseViscNet):
    """ViscNet predictor of viscosity and viscosity parameters.

    ViscNet is a physics-informed neural network that has the MYEGA [1]
    viscosity equation embedded in it. See Ref. [2] for the original
    publication.

    References:
      [1] J.C. Mauro, Y. Yue, A.J. Ellison, P.K. Gupta, D.C. Allan, Viscosity of
          glass-forming liquids., Proceedings of the National Academy of
          Sciences of the United States of America. 106 (2009) 19780–19784.
          https://doi.org/10.1073/pnas.0911705106.
      [2] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719
    """

    parameters_range = {
        "log_eta_inf": [-18, 5],
        "Tg": [400, 1400],
        "m": [10, 130],
    }

    hparams = {
        "batch_size": 64,
        "layer_1_activation": "ReLU",
        "layer_1_batchnorm": False,
        "layer_1_dropout": 0.07942161101271952,
        "layer_1_size": 192,
        "layer_2_activation": "Tanh",
        "layer_2_batchnorm": False,
        "layer_2_dropout": 0.05371454289414608,
        "layer_2_size": 48,
        "loss": "mse",
        "lr": 0.0011695226458761677,
        "max_epochs": 500,
        "n_features": 35,
        "num_layers": 2,
        "optimizer": "AdamW",
        "patience": 9,
    }

    absolute_features = [
        ("ElectronAffinity", "std1"),
        ("FusionEnthalpy", "std1"),
        ("GSenergy_pa", "std1"),
        ("GSmagmom", "std1"),
        ("NdUnfilled", "std1"),
        ("NfValence", "std1"),
        ("NpUnfilled", "std1"),
        ("atomic_radius_rahm", "std1"),
        ("c6_gb", "std1"),
        ("lattice_constant", "std1"),
        ("mendeleev_number", "std1"),
        ("num_oxistates", "std1"),
        ("nvalence", "std1"),
        ("vdw_radius_alvarez", "std1"),
        ("vdw_radius_uff", "std1"),
        ("zeff", "std1"),
    ]

    weighted_features = [
        ("FusionEnthalpy", "min"),
        ("GSbandgap", "max"),
        ("GSmagmom", "mean"),
        ("GSvolume_pa", "max"),
        ("MiracleRadius", "std1"),
        ("NValence", "max"),
        ("NValence", "min"),
        ("NdUnfilled", "max"),
        ("NdValence", "max"),
        ("NsUnfilled", "max"),
        ("SpaceGroupNumber", "max"),
        ("SpaceGroupNumber", "min"),
        ("atomic_radius", "max"),
        ("atomic_volume", "max"),
        ("c6_gb", "max"),
        ("c6_gb", "min"),
        ("max_ionenergy", "min"),
        ("num_oxistates", "max"),
        ("nvalence", "min"),
    ]

    x_mean = torch.tensor(
        [
            5.7542068e01,
            2.2090124e01,
            2.0236173e00,
            3.6860932e-02,
            3.2620981e-01,
            1.4419103e00,
            2.0164611e00,
            3.4407539e01,
            1.2352635e03,
            1.4792695e00,
            4.2045139e01,
            8.4130961e-01,
            2.3045063e00,
            4.7984661e01,
            5.6983612e01,
            1.1145602e00,
            9.2185795e-02,
            2.1363457e-01,
            2.2581025e-04,
            5.8149724e00,
            1.2964197e01,
            3.7008467e00,
            1.3742505e-01,
            1.8369867e-02,
            3.2303229e-01,
            7.1324579e-02,
            5.0019447e01,
            4.3720217e00,
            3.6445999e01,
            8.4036522e00,
            2.0281389e02,
            7.5613613e00,
            1.2259276e02,
            6.7182720e-01,
            1.0508456e-01,
        ]
    ).float()

    x_std = torch.tensor(
        [
            7.6420674e00,
            4.7180834e00,
            4.5827568e-01,
            1.6872969e-01,
            9.7033477e-01,
            2.7695282e00,
            3.3152765e-01,
            6.4520807e00,
            6.3392188e02,
            4.0605769e-01,
            1.1776709e01,
            2.8129578e-01,
            7.9213607e-01,
            7.5883222e00,
            1.1334700e01,
            2.8822860e-01,
            4.4787191e-02,
            1.1219133e-01,
            1.2392291e-03,
            1.1634388e00,
            2.9513965e00,
            4.7246245e-01,
            3.1958127e-01,
            8.8973150e-02,
            6.7548370e-01,
            6.2869355e-02,
            1.0003569e01,
            2.7434089e00,
            1.9245349e00,
            3.4734800e-01,
            1.2475176e02,
            3.2667663e00,
            1.5287421e02,
            7.3510885e-02,
            1.6187511e-01,
        ]
    ).float()

    state_dict_path = _BASEMODELPATH / "ViscNet_SD.p"

    def __init__(self):
        super().__init__(
            self.parameters_range, self.hparams, self.x_mean, self.x_std
        )

        state_dict = pickle.load(open(self.state_dict_path, "rb"))
        self.load_state_dict(state_dict)

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        """Computes the base-10 logarithm of viscosity using the MYEGA equation."""

        log_viscosity = (
            log_eta_inf
            + (12 - log_eta_inf)
            * (Tg / T)
            * ((m / (12 - log_eta_inf) - 1) * (Tg / T - 1)).exp()
        )
        return log_viscosity

    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ) -> np.ndarray:
        """Compute the chemical features used for viscosity prediction.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Array with the computed chemical features
        """

        feat_array, _ = physchem_featurizer(
            x=composition,
            input_cols=input_cols,
            weighted_features=self.weighted_features,
            absolute_features=self.absolute_features,
            rescale_to_sum=1,
            order="aw",
        )
        return feat_array


class ViscNetHuber(ViscNet):
    """ViscNet-Huber predictor of viscosity and viscosity parameters.

    ViscNet-Huber is a physics-informed neural network that has the MYEGA [1]
    viscosity equation embedded in it. The difference between this model and
    ViscNet is the loss function: this model has a robust smooth-L1 loss
    function, while ViscNet has a MSE (L2) loss function. See Ref. [2] for the
    original publication.

    References:
      [1] J.C. Mauro, Y. Yue, A.J. Ellison, P.K. Gupta, D.C. Allan, Viscosity of
          glass-forming liquids., Proceedings of the National Academy of
          Sciences of the United States of America. 106 (2009) 19780–19784.
          https://doi.org/10.1073/pnas.0911705106.
      [2] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719
    """

    def __init__(self):
        super().__init__()

        self.hparams = self.hparams.copy()
        self.hparams["loss"] = "huber"
        self.loss_fun = F.smooth_l1_loss

        state_dict_path = _BASEMODELPATH / "ViscNetHuber_SD.p"
        state_dict = pickle.load(open(state_dict_path, "rb"))
        self.load_state_dict(state_dict)


class ViscNetVFT(ViscNet):
    """ViscNet-VFT predictor of viscosity and viscosity parameters.

    ViscNet-VFT is a physics-informed neural network that has the VFT [1-3]
    viscosity equation embedded in it. See Ref. [4] for the original
    publication.

    References:
      [1] H. Vogel, Das Temperatureabhängigketsgesetz der Viskosität von
          Flüssigkeiten, Physikalische Zeitschrift. 22 (1921) 645–646.
      [2] G.S. Fulcher, Analysis of recent measurements of the viscosity of
          glasses, Journal of the American Ceramic Society. 8 (1925) 339–355.
          https://doi.org/10.1111/j.1151-2916.1925.tb16731.x.
      [3] G. Tammann, W. Hesse, Die Abhängigkeit der Viscosität von der
          Temperatur bie unterkühlten Flüssigkeiten, Z. Anorg. Allg. Chem. 156
          (1926) 245–257. https://doi.org/10.1002/zaac.19261560121.
      [4] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719
    """

    def __init__(self):
        super().__init__()

        state_dict_path = _BASEMODELPATH / "ViscNetVFT_SD.p"
        state_dict = pickle.load(open(state_dict_path, "rb"))
        self.load_state_dict(state_dict)

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        """Computes the base-10 logarithm of viscosity using the VFT equation.

        Reference:
          [1] H. Vogel, Das Temperatureabhängigketsgesetz der Viskosität von
            Flüssigkeiten, Physikalische Zeitschrift. 22 (1921) 645–646.
          [2] G.S. Fulcher, Analysis of recent measurements of the viscosity of
            glasses, Journal of the American Ceramic Society. 8 (1925) 339–355.
            https://doi.org/10.1111/j.1151-2916.1925.tb16731.x.
          [3] G. Tammann, W. Hesse, Die Abhängigkeit der Viscosität von der
            Temperatur bie unterkühlten Flüssigkeiten, Z. Anorg. Allg. Chem. 156
            (1926) 245–257. https://doi.org/10.1002/zaac.19261560121.
        """

        log_viscosity = log_eta_inf + (12 - log_eta_inf) ** 2 / (
            m * (T / Tg - 1) + (12 - log_eta_inf)
        )
        return log_viscosity


class GlassNet(MTL):
    """Multi-task neural network for predicting glass properties."""

    hparams = {
        "batch_size": 256,
        "layer_1_activation": "Softplus",
        "layer_1_batchnorm": True,
        "layer_1_dropout": 0.08118311665886885,
        "layer_1_size": 280,
        "layer_2_activation": "Mish",
        "layer_2_batchnorm": True,
        "layer_2_dropout": 0.0009472891190852595,
        "layer_2_size": 500,
        "layer_3_activation": "LeakyReLU",
        "layer_3_batchnorm": False,
        "layer_3_dropout": 0.08660291424886811,
        "layer_3_size": 390,
        "layer_4_activation": "PReLU",
        "layer_4_batchnorm": False,
        "layer_4_dropout": 0.16775047518280012,
        "layer_4_size": 480,
        "loss": "mse",
        "lr": 1.3252600209332101e-05,
        "max_epochs": 2000,
        "n_features": 98,
        "n_targets": 85,
        "num_layers": 4,
        "optimizer": "AdamW",
        "patience": 27,
    }

    targets = [
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "Viscosity773K",
        "Viscosity873K",
        "Viscosity973K",
        "Viscosity1073K",
        "Viscosity1173K",
        "Viscosity1273K",
        "Viscosity1373K",
        "Viscosity1473K",
        "Viscosity1573K",
        "Viscosity1673K",
        "Viscosity1773K",
        "Viscosity1873K",
        "Viscosity2073K",
        "Viscosity2273K",
        "Viscosity2473K",
        "Tg",
        "Tmelt",
        "Tliquidus",
        "TLittletons",
        "TAnnealing",
        "Tstrain",
        "Tsoft",
        "TdilatometricSoftening",
        "AbbeNum",
        "RefractiveIndex",
        "RefractiveIndexLow",
        "RefractiveIndexHigh",
        "MeanDispersion",
        "Permittivity",
        "TangentOfLossAngle",
        "TresistivityIs1MOhm.m",
        "Resistivity273K",
        "Resistivity373K",
        "Resistivity423K",
        "Resistivity573K",
        "Resistivity1073K",
        "Resistivity1273K",
        "Resistivity1473K",
        "Resistivity1673K",
        "YoungModulus",
        "ShearModulus",
        "Microhardness",
        "PoissonRatio",
        "Density293K",
        "Density1073K",
        "Density1273K",
        "Density1473K",
        "Density1673K",
        "ThermalConductivity",
        "ThermalShockRes",
        "CTEbelowTg",
        "CTE328K",
        "CTE373K",
        "CTE433K",
        "CTE483K",
        "CTE623K",
        "Cp293K",
        "Cp473K",
        "Cp673K",
        "Cp1073K",
        "Cp1273K",
        "Cp1473K",
        "Cp1673K",
        "TMaxGrowthVelocity",
        "MaxGrowthVelocity",
        "CrystallizationPeak",
        "CrystallizationOnset",
        "SurfaceTensionAboveTg",
        "SurfaceTension1173K",
        "SurfaceTension1473K",
        "SurfaceTension1573K",
        "SurfaceTension1673K",
    ]

    target_trans = {p: i for i, p in enumerate(targets)}

    # fmt: off
    element_features = [
        "Ag", "Al", "As", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce",
        "Cl", "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Eu", "Fe", "Ga", "Gd", "Ge",
        "H", "Hf", "Hg", "Ho", "I", "In", "K", "La", "Li", "Lu", "Mg", "Mn",
        "Mo", "N", "Na", "Nb", "Nd", "Ni", "P", "Pb", "Pr", "Rb", "S", "Sb",
        "Sc", "Se", "Sm", "Sn", "Sr", "Ta", "Tb", "Te", "Ti", "Tl", "V", "W",
        "Y", "Yb", "Zn", "Zr",
    ]
    # fmt: on

    weighted_features = [
        ("c6_gb", "min"),
        ("ElectronAffinity", "min"),
        ("NUnfilled", "std1"),
        ("NdValence", "min"),
        ("NfValence", "min"),
        ("NpUnfilled", "min"),
        ("NsUnfilled", "min"),
        ("nvalence", "max"),
        ("atomic_volume", "min"),
        ("GSbandgap", "min"),
        ("GSenergy_pa", "max"),
        ("FusionEnthalpy", "min"),
    ]

    absolute_features = [
        ("num_oxistates", "max"),
        ("NUnfilled", "min"),
        ("NdUnfilled", "max"),
        ("NdValence", "max"),
        ("NfUnfilled", "sum"),
        ("NfValence", "sum"),
        ("NpUnfilled", "sum"),
        ("NpUnfilled", "max"),
        ("NpValence", "max"),
        ("NsUnfilled", "sum"),
        ("NsValence", "min"),
        ("nvalence", "max"),
        ("vdw_radius_uff", "max"),
        ("atomic_radius_rahm", "max"),
        ("GSbandgap", "min"),
        ("GSestFCClatcnt", "std1"),
        ("GSmagmom", "sum"),
        ("en_Sanderson", "max"),
        ("en_Tardini_Organov", "min"),
        ("zeff", "std1"),
        ("boiling_point", "std1"),
        ("FusionEnthalpy", "max"),
    ]

    _visc_parameters = [
        "log10_eta_infinity (Pa.s)",
        "Tg_MYEGA (K)",
        "fragility",
    ]

    training_file = _BASEMODELPATH / "GlassNet.p"
    training_file_mh = _BASEMODELPATH / "GlassNetMH.p"
    scaler_x_file = _BASEMODELPATH / "GlassNet_scaler_x.joblib"
    scaler_y_file = _BASEMODELPATH / "GlassNet_scaler_y.joblib"

    def __init__(self, multihead=True, loadrf=False):
        msg = "To use `loadrf=True` you must have `glasspy_extra` installed"
        assert not loadrf or _HAS_GLASSPY_EXTRA, msg

        super().__init__(self.hparams)

        self.scaler_x = joblib.load(self.scaler_x_file)
        self.scaler_y = joblib.load(self.scaler_y_file)

        dim = int(self.hparams[f'layer_{self.hparams["num_layers"]}_size'])

        if multihead:
            self.multihead = True
            self.output_layer = nn.Identity()
            self.tasks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, 10), nn.ReLU(), nn.Linear(10, 1)
                    )
                    for n in range(self.hparams["n_targets"])
                ]
            )
            self.load_training(self.training_file_mh)
            self._rf_models = [
                "AbbeNum",
                "CrystallizationOnset",
                "CrystallizationPeak",
                "CTE433K",
                "CTEbelowTg",
                "Density293K",
                "MeanDispersion",
                "Microhardness",
                "RefractiveIndex",
                "Resistivity273K",
                "Resistivity373K",
                "Resistivity423K",
                "Resistivity573K",
                "TdilatometricSoftening",
                "Tg",
                "Tliquidus",
                "Tmelt",
                "Density1073K",
            ]

        else:
            self.multihead = False
            self.output_layer = nn.Sequential(
                nn.Linear(dim, self.hparams["n_targets"]),
            )
            self.load_training(self.training_file)
            self._rf_models = [
                "AbbeNum",
                "CrystallizationOnset",
                "CrystallizationPeak",
                "CTE433K",
                "CTEbelowTg",
                "Density293K",
                "MeanDispersion",
                "Microhardness",
                "RefractiveIndex",
                "Resistivity273K",
                "Resistivity373K",
                "Resistivity423K",
                "Resistivity573K",
                "TdilatometricSoftening",
                "Tg",
                "Tliquidus",
                "Tmelt",
                "TresistivityIs1MOhm.m",
                "Tsoft",
                "Tstrain",
                "YoungModulus",
            ]

        self.rf_models_loaded = False
        if loadrf:
            self._load_rf_models()
            self.rf_models_loaded = True

    def _load_data(self):
        """Loads the data used to train GlassNet.

        This method takes some time to run, that is why it is not run
        automatically when a new instance of the class is created. After running
        once, the instance will have a `data` attribute that is a DataFrame with
        all the data.

        Returns:
          DataFrame with the all the data used to train and validate
          GlassNet."""

        if not hasattr(self, "data"):
            remove_dupe_decimals = 3

            removed_compounds = [
                "",
                "Al2O3+Fe2O3",
                "MoO3+WO3",
                "CaO+MgO",
                "FeO+Fe2O3",
                "Li2O+Na2O+K2O",
                "Na2O+K2O",
                "F2O-1",
                "FemOn",
                "HF+H2O",
                "R2O",
                "R2O3",
                "R2O3",
                "RO",
                "RmOn",
            ]

            removed_elements = [
                "Ac",
                "Am",
                "Ar",
                "At",
                "Bk",
                "Cf",
                "Cm",
                "Es",
                "Fm",
                "Fr",
                "He",
                "Kr",
                "Ne",
                "Np",
                "Pa",
                "Pm",
                "Po",
                "Pu",
                "Ra",
                "Rn",
                "Th",
                "U",
                "Xe",
            ]

            treatment = {
                "AbbeNum": {
                    "max": 115,
                },
                "Cp293K": {
                    "max": 2000,
                },
                "Cp473K": {
                    "max": 2000,
                },
                "Cp673K": {
                    "max": 3000,
                },
                "Cp1073K": {
                    "min": 500,
                    "max": 2500,
                },
                "Cp1273K": {
                    "min": 500,
                    "max": 3000,
                },
                "Cp1473K": {
                    "min": 500,
                    "max": 3000,
                },
                "Cp1673K": {
                    "min": 500,
                    "max": 2250,
                },
                "CTE328K": {
                    "min": 10**-6.5,
                    "log": True,
                },
                "CTE373K": {
                    "min": 10**-6.5,
                    "log": True,
                },
                "CTE433K": {
                    "min": 10**-8,
                    "log": True,
                },
                "CTE483K": {
                    "min": 10**-7,
                    "log": True,
                },
                "CTE623K": {
                    "min": 10**-6.5,
                    "log": True,
                },
                "CTEbelowTg": {
                    "min": 0,
                    "log": True,
                },
                "Density293K": {
                    "min": 1,
                    "max": 10,
                },
                "CTE623K": {
                    "log": True,
                },
                "MaxGrowthVelocity": {
                    "min": 1e-10,
                    "log": True,
                },
                "MeanDispersion": {
                    "log": True,
                },
                "Microhardness": {
                    "max": 15,
                },
                "Permittivity": {
                    "max": 50,
                },
                "PoissonRatio": {
                    "min": 0,
                    "max": 1,
                },
                "RefractiveIndex": {
                    "max": 4,
                },
                "RefractiveIndexHigh": {
                    "min": 1.7,
                    "max": 3.5,
                },
                "Resistivity273K": {
                    "log": True,
                    "max": 1e40,
                },
                "Resistivity373K": {
                    "log": True,
                    "max": 1e28,
                },
                "Resistivity423K": {
                    "log": True,
                },
                "Resistivity573K": {
                    "log": True,
                },
                "Resistivity1073K": {
                    "max": 10**4,
                    "log": True,
                },
                "Resistivity1273K": {
                    "max": 10**5,
                    "log": True,
                },
                "Resistivity1473K": {
                    "log": True,
                },
                "Resistivity1673K": {
                    "log": True,
                },
                "SurfaceTension1473K": {
                    "max": 0.5,
                },
                "SurfaceTension1573K": {
                    "max": 0.7,
                },
                "SurfaceTension1673K": {
                    "max": 0.7,
                },
                "SurfaceTensionAboveTg": {
                    "max": 0.8,
                },
                "Viscosity773K": {
                    "log": True,
                },
                "Viscosity873K": {
                    "log": True,
                },
                "Viscosity973K": {
                    "log": True,
                },
                "Viscosity1073K": {
                    "log": True,
                },
                "Viscosity1173K": {
                    "log": True,
                },
                "Viscosity1273K": {
                    "log": True,
                },
                "Viscosity1373K": {
                    "log": True,
                },
                "Viscosity1473K": {
                    "log": True,
                },
                "Viscosity1573K": {
                    "log": True,
                },
                "Viscosity1673K": {
                    "log": True,
                },
                "Viscosity1773K": {
                    "max": 10**10,
                    "log": True,
                },
                "Viscosity1873K": {
                    "max": 10**10,
                    "log": True,
                },
                "Viscosity2073K": {
                    "max": 10**8,
                    "log": True,
                },
                "Viscosity2273K": {
                    "max": 10**8,
                    "log": True,
                },
                "Viscosity2473K": {
                    "log": True,
                },
                "T3": {
                    "max": 2350,
                },
                "T4": {
                    "max": 2000,
                },
                "TangentOfLossAngle": {
                    "min": 1e-4,
                    "max": 0.16,
                    "log": True,
                },
                "ThermalConductivity": {
                    "max": 6,
                },
                "TresistivityIs1MOhm.m": {
                    "max": 2000,
                },
                "YoungModulus": {
                    "max": 175,
                },
                "Tsoft": {
                    "max": 1600,
                },
            }

            min_cols = [
                col
                for col in treatment
                if treatment[col].get("min", None) is not None
            ]

            max_cols = [
                col
                for col in treatment
                if treatment[col].get("max", None) is not None
            ]

            log_cols = [
                ("property", col)
                for col in treatment
                if treatment[col].get("log", False)
            ]

            propconf = {"keep": self.targets}

            compconf = {
                "acceptable_sum_deviation": 1,
                "final_sum": 1,
                "return_weight": False,
                "dropline": removed_compounds,
                "drop_compound_with_element": removed_elements,
            }

            sg = SciGlass(False, propconf, compconf)

            for col in min_cols:
                logic = sg.data["property"][col] < treatment[col]["min"]
                sg.data.loc[logic, ("property", col)] = np.nan

            for col in max_cols:
                logic = sg.data["property"][col] > treatment[col]["max"]
                sg.data.loc[logic, ("property", col)] = np.nan

            sg.data[log_cols] = sg.data[log_cols].apply(np.log10, axis=1)

            sg.elements_from_compounds(final_sum=1, compounds_in_weight=False)

            sg.remove_duplicate_composition(
                scope="elements",
                decimals=remove_dupe_decimals,
                aggregator="median",
            )

            self.data = sg.data

        return self.data

    def _load_rf_models(self):
        if not self.rf_models_loaded:
            rf_dict = {}
            for target in self._rf_models:
                rf_dict[target] = load_rf_model(target)
            self.rf_dict = rf_dict

    def get_training_dataset(self):
        """Gets the training dataset used in the GlassNet paper.

        Returns:
          DataFrame containing the training dataset used in the GlassNet
          paper."""

        self._load_data()
        indices = self.data.index
        train_idx, _ = train_test_split(
            indices,
            test_size=0.1,
            random_state=61455,
            shuffle=True,
        )
        return self.data.loc[train_idx]

    def get_test_dataset(self):
        """Gets the holdout dataset used in the GlassNet paper.

        Returns:
          DataFrame containing the holdout dataset used in the GlassNet
          paper."""

        self._load_data()
        indices = self.data.index
        _, test_idx = train_test_split(
            indices,
            test_size=0.1,
            random_state=61455,
            shuffle=True,
        )
        return self.data.loc[test_idx]

    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        auto_scale: bool = True,
        return_cols: bool = False,
    ) -> np.ndarray:
        """Compute the features used for input in GlassNet.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          auto_scale:
            If `True`, then the features are automatically scaled and ready to
            use with GlassNet. If `False`, then the features will not be scaled
            and should not be used to make predictions with Glassnet. Default
            value is `True`.
          return_cols:
            If `True`, then a list with the name of the feature columns is also
            returned.

        Returns:
          Array with the computed features. Optionally, the name of the coulmns
          can also be returned if `return_cols` is True.
        """

        feat_array, cols = physchem_featurizer(
            x=composition,
            input_cols=input_cols,
            elemental_features=self.element_features,
            weighted_features=self.weighted_features,
            absolute_features=self.absolute_features,
            rescale_to_sum=1,
            order="eaw",
        )

        if auto_scale:
            feat_array = self.scaler_x.transform(feat_array)

        if return_cols:
            return feat_array, cols
        else:
            return feat_array

    def forward(self, x):
        """Method used for training the neural network.

        Consider using the other methods for prediction.

        Args:
          x:
            Feature tensor.

        Returns
          Tensor with the predictions.
        """

        if self.multihead:
            dense = self.hidden_layers(x)
            return torch.hstack([task(dense) for task in self.tasks])

        else:
            return self.output_layer(self.hidden_layers(x))

    def predict(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        return_dataframe: bool = True,
    ):
        """Makes prediction of properties using GlassNet.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          return_dataframe:
            If `True`, then returns a pandas DataFrame, else returns an array.
            Default value is `True`.

        Returns:
          Predicted values of properties. Will be a DataFrame if
          `return_dataframe` is True, otherwise will be an array.
        """

        is_training = self.training
        if is_training:
            self.eval()

        with torch.no_grad():
            features = self.featurizer(composition, input_cols)
            features = torch.from_numpy(features).float()
            y_pred = self.scaler_y.inverse_transform(self(features).detach())

        if is_training:
            self.train()

        if self.rf_models_loaded:
            for target in self.rf_dict:
                y_pred[:, self.target_trans[target]] = self.rf_dict[
                    target
                ].predict(features)

        if return_dataframe:
            return pd.DataFrame(y_pred, columns=self.targets)
        else:
            return y_pred

    def _viscosity_table_single(
        self,
        prediction,
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
    ):
        """Viscosity table prediction.

        Args:
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.

        Returns:
          Viscosity table.
        """

        data = []

        for col in columns:
            value = prediction[self.target_trans[col]]

            if col.startswith("Viscosity"):
                T = int(col[9:-1])
                log_visc = value

            elif col.startswith("T"):
                T = value

                if col[1].isdigit():
                    log_visc = int(col[1:])

                else:
                    if col == "Tg":
                        log_visc = 12
                    elif col == "TLittletons":
                        log_visc = 6.6
                    elif col == "TAnnealing":
                        log_visc = 12.4
                    elif col == "Tstrain":
                        log_visc = 13.5
                    elif col == "TdilatometricSoftening":
                        log_visc = 10
                    elif col == "Tsoft":
                        log_visc = 6.6
                    elif col == "Tmelt":
                        log_visc = 1
                    else:
                        log_visc = np.nan

            if log_visc > log_visc_limit:
                log_visc = np.nan

            if np.isfinite(log_visc):
                data.append((T, log_visc))

        data = np.array(sorted(data))

        return data

    def _viscosity_param_single(
        self,
        prediction,
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
    ):
        """Generator of viscosity tables.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          return_dataframe:
            If `True`, returns a pandas DataFrame. Else returns a numpy array.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.

        Return:
          Viscosity parameters.
        """

        data = self._viscosity_table_single(
            prediction, log_visc_limit, columns
        )

        try:
            slope, _, _, _ = theilslopes(
                data[:n_points_low, 1],
                1 / data[:n_points_low, 0],
            )

            closest_to_Tg = np.argmin(np.abs(data[:, 1] - 12))

            guess_Tg = data[closest_to_Tg, 0]
            guess_m = slope / guess_Tg
            guess_ninf = data[-1, 1]

            regression = least_squares(
                _myega_residuals,
                [guess_ninf, guess_Tg, guess_m],
                loss="cauchy",
                f_scale=0.5,  # inlier residuals are lower than this value
                args=(data[:, 0], data[:, 1]),
                bounds=([-np.inf, 0, 10], [11, np.inf, np.inf]),
            )

            log_eta_inf, Tg_MYEGA, fragility = regression.x

            return log_eta_inf, Tg_MYEGA, fragility

        except ValueError:
            return np.nan, np.nan, np.nan

    def viscosity_parameters(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_dataframe: bool = True,
    ):
        """Predict the MYEGA viscosity parameters.

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the parameters returned will be NaN.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.
          return_dataframe:
            If `True`, then returns a pandas DataFrame, else returns an array.
            Default value is `True`.

        Returns:
          parameters:
            Numpy array or DataFrame of the viscosity parameters. The first
            column is the asymptotic viscosity (in base-10 logarithm of Pa.s);
            the second column is the temperature where the viscosity is 10^12
            Pa.s; and the third column is the fragility index.
        """

        predictions = self.predict(composition, input_cols, False)
        parameters = [
            self._viscosity_param_single(
                i, log_visc_limit, columns, n_points_low
            )
            for i in predictions
        ]

        if return_dataframe:
            return pd.DataFrame(parameters, columns=self._visc_parameters)
        else:
            return np.array(parameters)

    def predict_log10_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_parameters: bool = False,
    ):
        """Predict the base-10 logarithm of viscosity (viscosity in Pa.s).

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the predicted viscosity will be NaN.

        Args:
          T:
            Temperature (in Kelvin) to compute the viscosity. If this is a numpy
            array, it must have only one dimension.
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable for the regression. Predictions of values greater than 12
            for the log10 of viscosity are prone to higher errors. Default value
            is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.
          return_parameters:
            If `True`, also returns the viscosity parameters array.

        Returns:
          log_10_viscosity:
            Numpy array of the predicted base-10 logarithm of viscosity
          parameters:
            Numpy array of the viscosity parameters. The first column is the
            asymptotic viscosity (in base-10 logarithm of Pa.s); the second
            column is the temperature where the viscosity is 10^12 Pa.s; and the
            third column is the fragility index. Only returned if
            `return_parameters` is `True`.
        """

        parameters = self.viscosity_parameters(
            composition,
            input_cols,
            log_visc_limit,
            columns,
            n_points_low,
            return_dataframe=False,
        )

        log10_viscosity = myega_alt(
            T, parameters[:, 0], parameters[:, 1], parameters[:, 2]
        )

        if return_parameters:
            return log10_viscosity, parameters
        else:
            return log10_viscosity

    def predict_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_parameters: bool = False,
    ):
        """Predict the viscosity in Pa.s.

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the predicted viscosity will be NaN.

        Args:
          T:
            Temperature (in Kelvin) to compute the viscosity. If this is a numpy
            array, it must have only one dimension.
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable for the regression. Predictions of values greater than 12
            for the log10 of viscosity are prone to higher errors. Default value
            is 12.
          columns:
            Which targets of GlassNet that will be considered to perform the
            regression.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index for the regression. Must be a positive integer. Default value
            is 10.
          return_parameters:
            If `True`, also returns the viscosity parameters array.

        Returns:
          viscosity:
            Numpy array of the predicted viscosity in Pa.s.
          parameters:
            Numpy array of the viscosity parameters. The first column is the
            asymptotic viscosity (in base-10 logarithm of Pa.s); the second
            column is the temperature where the viscosity is 10^12 Pa.s; and the
            third column is the fragility index. Only returned if
            `return_parameters` is `True`.
        """

        log10_viscosity, parameters = self.predict_log10_viscosity(
            T,
            composition,
            input_cols,
            log_visc_limit,
            columns,
            n_points_low,
            return_parameters=True,
        )

        if return_parameters:
            return 10**log10_viscosity, parameters
        else:
            return 10**log10_viscosity

    @staticmethod
    def citation(bibtex: bool = False) -> str:
        raise NotImplementedError("Not implemented yet.")
